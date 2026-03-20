"""Simulation tool: power flow, optimization, and advanced modes."""

import textwrap
from contextlib import contextmanager

from pypsamcp.core import (
    mcp,
    get_energy_model,
    convert_to_serializable,
    stdout_to_stderr,
)

import numpy as np
import pandas as pd


def _patch_single_column_pnl(network):
    """Work around PyPSA 1.1.2 MGA bug with single-column pnl DataFrames.

    MGA post-processing calls _set_dynamic_data which does:
        c.dynamic[attr].loc[idx, cols] = df
    When df has shape (N,1), pandas raises 'setting an array element with a
    sequence' due to a broadcasting mismatch. This only affects components
    with exactly one member that has time-varying data.

    Fix: for each single-column pnl DataFrame, add a zero-filled dummy column.
    Returns a cleanup function that removes the dummies after MGA completes.
    """
    patches = []  # list of (pnl_accessor, attr, dummy_col_name) to undo

    for comp_name in ["loads", "generators", "storage_units", "stores", "links", "lines"]:
        pnl = getattr(network, f"{comp_name}_t", None)
        if pnl is None:
            continue
        for attr in list(pnl.keys()):
            df = pnl[attr]
            if df.empty or df.shape[1] != 1:
                continue
            # Single-column — add a zero-filled dummy
            dummy_col = f"__mga_pad_{comp_name}_{attr}__"
            pnl[attr][dummy_col] = 0.0
            patches.append((pnl, attr, dummy_col))

    def cleanup():
        for pnl, attr, dummy_col in patches:
            if dummy_col in pnl[attr].columns:
                pnl[attr] = pnl[attr].drop(columns=[dummy_col])

    return cleanup


@contextmanager
def _patch_set_dynamic_data():
    """Patch PyPSA's _set_dynamic_data to handle single-column DataFrames.

    PyPSA 1.1.2 has a bug where assigning a (N,1) DataFrame via
    .loc[idx, cols] to another (M,1) DataFrame raises 'setting an array
    element with a sequence' due to a pandas broadcasting mismatch.

    This fix squeezes single-column solution DataFrames to Series before
    assignment, which avoids the shape mismatch.
    """
    import pypsa.optimization.common as _common
    import pypsa.optimization.optimize as _optimize

    _original_common = _common._set_dynamic_data
    _original_optimize = _optimize._set_dynamic_data

    def _patched(n, component, attr, df):
        c = n.components[component]
        if (attr not in c.dynamic) or (c.dynamic[attr].empty):
            c.dynamic[attr] = df.reindex(n.snapshots)
        else:
            if df.shape[1] == 1:
                # Squeeze to Series to avoid (N,1) vs (N,) broadcast mismatch
                col = df.columns[0]
                c.dynamic[attr].loc[df.index, col] = df[col]
            else:
                c.dynamic[attr].loc[df.index, df.columns] = df

        result = c.dynamic[attr].reindex(n.snapshots, level="snapshot", axis=0)
        if n.has_scenarios:
            expected_columns = pd.MultiIndex.from_product(
                [n.scenarios, c.names], names=["scenario", "name"]
            )
            result = result.reindex(columns=expected_columns)
        else:
            result = result.reindex(c.names, level="name", axis=1)

        c.dynamic[attr] = result.fillna(0.0)

    # Patch both the source module and the importing module
    _common._set_dynamic_data = _patched
    _optimize._set_dynamic_data = _patched
    try:
        yield
    finally:
        _common._set_dynamic_data = _original_common
        _optimize._set_dynamic_data = _original_optimize


VALID_MODES = {
    "pf",
    "lpf",
    "optimize",
    "mga",
    "security_constrained",
    "rolling_horizon",
    "transmission_expansion_iterative",
    "optimize_and_pf",
}


def _collect_pf_results(network) -> dict:
    """Collect power flow results: bus voltages, angles, line flows."""
    summary = {}
    if not network.buses_t.v_mag_pu.empty:
        summary["bus_v_mag_pu"] = convert_to_serializable(network.buses_t.v_mag_pu)
    if not network.buses_t.v_ang.empty:
        summary["bus_v_ang"] = convert_to_serializable(network.buses_t.v_ang)
    if not network.lines_t.p0.empty:
        summary["line_p0"] = convert_to_serializable(network.lines_t.p0)
    if not network.lines_t.p1.empty:
        summary["line_p1"] = convert_to_serializable(network.lines_t.p1)
    return summary


def _is_infeasible(termination_condition: str) -> bool:
    """Check if the optimization result indicates infeasibility."""
    return termination_condition in ("infeasible", "infeasible_or_unbounded")


def _build_infeasible_response(mode: str, status: str, termination_condition: str,
                               compute_infeasibilities: bool = False) -> dict:
    """Build a response dict for an infeasible optimization result.

    Returns clean data instead of stale results from a prior solve.
    """
    resp = {
        "mode": mode,
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": None,
        "message": f"Optimization infeasible ({termination_condition}). "
                   "No valid dispatch or expansion results available.",
        "summary": {},
    }
    if compute_infeasibilities:
        resp["infeasibility_note"] = (
            "compute_infeasibilities=True requires the Gurobi solver in PyPSA 1.x. "
            "With HiGHS, use load-shedding generators (high marginal_cost VoLL dummy "
            "generators on each bus) to identify and quantify infeasible buses/hours."
        )
    return resp


def _collect_optimization_results(network) -> dict:
    """Collect optimization results: dispatch, expansion, shadow prices."""
    summary = {}
    # Generator dispatch
    if not network.generators_t.p.empty:
        summary["generator_dispatch"] = convert_to_serializable(
            network.generators_t.p
        )
    # Storage unit dispatch
    if not network.storage_units_t.p.empty:
        summary["storage_unit_dispatch"] = convert_to_serializable(
            network.storage_units_t.p
        )
    # Optimal capacities
    if "p_nom_opt" in network.generators.columns:
        gen_expanded = network.generators[
            network.generators.p_nom_opt != network.generators.p_nom
        ]
        if not gen_expanded.empty:
            summary["generator_expansion"] = convert_to_serializable(
                gen_expanded[["p_nom", "p_nom_opt"]]
            )
    # Bus shadow prices
    if not network.buses_t.marginal_price.empty:
        summary["bus_marginal_price"] = convert_to_serializable(
            network.buses_t.marginal_price
        )
    return summary


def _build_extra_functionality(code_string: str):
    """Compile a code string into a callable extra_functionality function.

    The code_string can use either `network` or `n` to refer to the PyPSA
    network (both are available for compatibility with PyPSA conventions).
    `snapshots` is also available.
    """

    def extra_functionality(network, snapshots):
        indented = textwrap.indent(code_string, "    ")
        func_code = f"def _user_func(network, n, snapshots):\n{indented}"
        local_ns = {}
        exec(func_code, {}, local_ns)  # noqa: S102
        local_ns["_user_func"](network, network, snapshots)

    return extra_functionality


@mcp.tool()
async def run_simulation(
    model_id: str,
    mode: str = "optimize",
    # pf/lpf
    snapshots: list[str] | None = None,
    distribute_slack: bool = False,
    slack_weights: str = "p_set",
    x_tol: float = 1e-6,
    # optimize
    solver_name: str = "highs",
    formulation: str = "kirchhoff",
    multi_investment_periods: bool = False,
    transmission_losses: int | bool = False,
    linearized_unit_commitment: bool = False,
    assign_all_duals: bool = False,
    compute_infeasibilities: bool = False,
    solver_options: dict | None = None,
    extra_functionality: str | None = None,
    # mga
    slack: float = 0.05,
    sense: str = "min",
    weights: dict | None = None,
    # security_constrained
    branch_outages: list[str] | None = None,
    # rolling_horizon
    horizon: int = 100,
    overlap: int = 0,
    # transmission_expansion_iterative
    msq_threshold: float = 0.05,
    min_iterations: int = 1,
    max_iterations: int = 100,
) -> dict:
    """Run a simulation on a PyPSA network.

    Supports 8 modes: pf, lpf, optimize, mga, security_constrained,
    rolling_horizon, transmission_expansion_iterative, optimize_and_pf.
    """
    if mode not in VALID_MODES:
        return {
            "error": f"Invalid mode '{mode}'. Valid modes: {sorted(VALID_MODES)}"
        }

    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        if mode in ("pf", "lpf"):
            return await _run_power_flow(network, mode, snapshots, distribute_slack, slack_weights, x_tol)
        elif mode == "optimize":
            return await _run_optimize(
                network, mode, solver_name, formulation, multi_investment_periods,
                transmission_losses, linearized_unit_commitment, assign_all_duals,
                compute_infeasibilities, solver_options, extra_functionality,
            )
        elif mode == "mga":
            return await _run_mga(
                network, solver_name, formulation, multi_investment_periods,
                transmission_losses, linearized_unit_commitment, assign_all_duals,
                compute_infeasibilities, solver_options, extra_functionality,
                slack, sense, weights,
            )
        elif mode == "security_constrained":
            return await _run_security_constrained(
                network, solver_name, solver_options,
                extra_functionality, branch_outages,
            )
        elif mode == "rolling_horizon":
            return await _run_rolling_horizon(
                network, solver_name, solver_options,
                extra_functionality, horizon, overlap,
            )
        elif mode == "transmission_expansion_iterative":
            return await _run_transmission_expansion(
                network, solver_name, solver_options,
                extra_functionality, msq_threshold, min_iterations, max_iterations,
            )
        elif mode == "optimize_and_pf":
            return await _run_optimize_and_pf(
                network, solver_name, solver_options,
                extra_functionality,
            )
    except Exception as e:
        return {"error": f"Simulation failed: {e}"}


async def _run_power_flow(network, mode, snapshots, distribute_slack, slack_weights, x_tol):
    """Run PF or LPF."""
    kwargs = {}
    if snapshots is not None:
        kwargs["snapshots"] = snapshots
    if mode == "pf":
        kwargs["distribute_slack"] = distribute_slack
        kwargs["slack_weights"] = slack_weights
        kwargs["x_tol"] = x_tol

    with stdout_to_stderr():
        if mode == "pf":
            network.pf(**kwargs)
        else:
            network.lpf(**kwargs)

    summary = _collect_pf_results(network)
    return {
        "mode": mode,
        "message": f"{mode.upper()} completed successfully.",
        "summary": summary,
    }


async def _run_optimize(
    network, mode, solver_name, formulation, multi_investment_periods,
    transmission_losses, linearized_unit_commitment, assign_all_duals,
    compute_infeasibilities, solver_options, extra_functionality_code,
):
    """Run standard optimization."""
    kwargs = {
        "solver_name": solver_name,
        "formulation": formulation,
        "multi_investment_periods": multi_investment_periods,
        "transmission_losses": transmission_losses,
        "linearized_unit_commitment": linearized_unit_commitment,
        "assign_all_duals": assign_all_duals,
        "compute_infeasibilities": compute_infeasibilities,
    }
    if solver_options:
        kwargs["solver_options"] = solver_options
    if extra_functionality_code:
        kwargs["extra_functionality"] = _build_extra_functionality(extra_functionality_code)

    with stdout_to_stderr():
        status, termination_condition = network.optimize(**kwargs)

    if _is_infeasible(termination_condition):
        return _build_infeasible_response(
            mode, status, termination_condition, compute_infeasibilities,
        )

    summary = _collect_optimization_results(network)
    return {
        "mode": mode,
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(network.objective),
        "message": f"Optimization completed: {status} ({termination_condition})",
        "summary": summary,
    }


async def _run_mga(
    network, solver_name, formulation, multi_investment_periods,
    transmission_losses, linearized_unit_commitment, assign_all_duals,
    compute_infeasibilities, solver_options, extra_functionality_code,
    slack, sense, weights,
):
    """Run MGA: first optimize baseline, then run MGA."""
    opt_kwargs = {
        "solver_name": solver_name,
        "formulation": formulation,
        "multi_investment_periods": multi_investment_periods,
        "transmission_losses": transmission_losses,
        "linearized_unit_commitment": linearized_unit_commitment,
        "assign_all_duals": assign_all_duals,
        "compute_infeasibilities": compute_infeasibilities,
    }
    if solver_options:
        opt_kwargs["solver_options"] = solver_options
    if extra_functionality_code:
        opt_kwargs["extra_functionality"] = _build_extra_functionality(extra_functionality_code)

    with stdout_to_stderr():
        status, termination_condition = network.optimize(**opt_kwargs)

    mga_kwargs = {
        "slack": slack,
        "sense": sense,
    }
    if weights:
        mga_kwargs["weights"] = weights

    # Workaround for PyPSA 1.1.2 bug: MGA post-processing crashes when any
    # component's pnl DataFrame has exactly 1 column (shape (N,1) vs (N,)
    # broadcasting failure in pandas). We add a temporary dummy component to
    # pad single-column pnl DataFrames, then remove it after MGA completes.
    _mga_cleanup = _patch_single_column_pnl(network)

    try:
        with stdout_to_stderr():
            network.optimize.optimize_mga(**mga_kwargs)
    finally:
        _mga_cleanup()

    summary = _collect_optimization_results(network)
    return {
        "mode": "mga",
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(network.objective),
        "message": f"MGA completed: {status} ({termination_condition})",
        "summary": summary,
    }


async def _run_security_constrained(
    network, solver_name, solver_options,
    extra_functionality_code, branch_outages,
):
    """Run security-constrained optimization."""
    kwargs = {
        "solver_name": solver_name,
    }
    if solver_options:
        kwargs["solver_options"] = solver_options
    if extra_functionality_code:
        kwargs["extra_functionality"] = _build_extra_functionality(extra_functionality_code)
    if branch_outages:
        kwargs["branch_outages"] = branch_outages

    with stdout_to_stderr():
        status, termination_condition = network.optimize.optimize_security_constrained(**kwargs)

    summary = _collect_optimization_results(network)
    return {
        "mode": "security_constrained",
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(network.objective),
        "message": f"Security-constrained optimization completed: {status} ({termination_condition})",
        "summary": summary,
    }


async def _run_rolling_horizon(
    network, solver_name, solver_options,
    extra_functionality_code, horizon, overlap,
):
    """Run rolling horizon optimization."""
    kwargs = {
        "solver_name": solver_name,
        "horizon": horizon,
        "overlap": overlap,
    }
    if solver_options:
        kwargs["solver_options"] = solver_options
    if extra_functionality_code:
        kwargs["extra_functionality"] = _build_extra_functionality(extra_functionality_code)

    # Workaround for PyPSA 1.1.2 bug: assign_solution crashes with
    # "setting an array element with a sequence" when any component's pnl
    # DataFrame has exactly 1 column (shape (N,1) vs (N,) broadcasting
    # mismatch in pandas). Patch _set_dynamic_data to squeeze single-column
    # DataFrames before .loc assignment.
    with _patch_set_dynamic_data(), stdout_to_stderr():
        # optimize_with_rolling_horizon returns the Network, not (status, condition)
        network.optimize.optimize_with_rolling_horizon(**kwargs)

    # Infer status from network state after rolling horizon completes
    obj = network.objective
    if obj is not None and not (isinstance(obj, float) and obj != obj):
        status, termination_condition = "ok", "optimal"
    else:
        status, termination_condition = "warning", "unknown"

    summary = _collect_optimization_results(network)
    return {
        "mode": "rolling_horizon",
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(obj) if obj is not None else None,
        "message": f"Rolling horizon optimization completed: {status} ({termination_condition})",
        "summary": summary,
    }


async def _run_transmission_expansion(
    network, solver_name, solver_options,
    extra_functionality_code, msq_threshold, min_iterations, max_iterations,
):
    """Run iterative transmission expansion optimization."""
    kwargs = {
        "solver_name": solver_name,
        "msq_threshold": msq_threshold,
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
    }
    if solver_options:
        kwargs["solver_options"] = solver_options
    if extra_functionality_code:
        kwargs["extra_functionality"] = _build_extra_functionality(extra_functionality_code)

    with stdout_to_stderr():
        status, termination_condition = network.optimize.optimize_transmission_expansion_iteratively(**kwargs)

    summary = _collect_optimization_results(network)
    return {
        "mode": "transmission_expansion_iterative",
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(network.objective),
        "message": f"Transmission expansion optimization completed: {status} ({termination_condition})",
        "summary": summary,
    }


async def _run_optimize_and_pf(
    network, solver_name, solver_options,
    extra_functionality_code,
):
    """Run optimization followed by non-linear power flow."""
    kwargs = {
        "solver_name": solver_name,
    }
    if solver_options:
        kwargs["solver_options"] = solver_options
    if extra_functionality_code:
        kwargs["extra_functionality"] = _build_extra_functionality(extra_functionality_code)

    with stdout_to_stderr():
        status, termination_condition = network.optimize.optimize_and_run_non_linear_powerflow(**kwargs)

    summary = {
        **_collect_optimization_results(network),
        **_collect_pf_results(network),
    }
    return {
        "mode": "optimize_and_pf",
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(network.objective),
        "message": f"Optimize + PF completed: {status} ({termination_condition})",
        "summary": summary,
    }
