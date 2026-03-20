"""Simulation tool: power flow, optimization, and advanced modes."""

import textwrap

from pypsamcp.core import (
    mcp,
    get_energy_model,
    convert_to_serializable,
    stdout_to_stderr,
)

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

    try:
        with stdout_to_stderr():
            network.optimize.optimize_mga(**mga_kwargs)
    except ValueError as mga_err:
        err_msg = str(mga_err)
        if "setting an array element with a sequence" in err_msg:
            return {
                "error": "MGA post-processing failed due to a known PyPSA 1.1.2 bug "
                "when loads have time-varying p_set stored as dynamic data. "
                "Workaround: set load p_set as a static scalar value, or use the "
                "'optimize' mode which is not affected.",
                "mode": "mga",
                "baseline_status": status,
                "baseline_objective": float(network.objective),
            }
        raise

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

    with stdout_to_stderr():
        status, termination_condition = network.optimize.optimize_with_rolling_horizon(**kwargs)

    summary = _collect_optimization_results(network)
    return {
        "mode": "rolling_horizon",
        "status": status,
        "termination_condition": termination_condition,
        "objective_value": float(network.objective),
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
