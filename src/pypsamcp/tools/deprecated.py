"""Deprecated tool aliases for backwards compatibility.

These tools delegate to their replacements and will be removed in a future release.
"""

from pypsamcp.core import mcp
from pypsamcp.tools.simulation import run_simulation as _run_simulation
from pypsamcp.tools.time_config import configure_time as _configure_time

_DEPRECATION_NOTICE_SNAPSHOTS = (
    "set_snapshots is deprecated. Use configure_time(mode='snapshots') instead."
)
_DEPRECATION_NOTICE_PF = (
    "run_powerflow is deprecated. Use run_simulation(mode='pf') instead."
)
_DEPRECATION_NOTICE_OPT = (
    "run_optimization is deprecated. Use run_simulation(mode='optimize') instead."
)


@mcp.tool()
async def set_snapshots(model_id: str, snapshots: list[str]) -> dict:
    """[Deprecated] Set time snapshots. Use configure_time(mode='snapshots') instead.

    Args:
        model_id: The model ID
        snapshots: List of datetime strings
    """
    result = await _configure_time(model_id, "snapshots", snapshots=snapshots)
    result["deprecation_notice"] = _DEPRECATION_NOTICE_SNAPSHOTS
    return result


@mcp.tool()
async def run_powerflow(model_id: str, snapshot: str | None = None) -> dict:
    """[Deprecated] Run power flow. Use run_simulation(mode='pf') instead.

    Args:
        model_id: The model ID
        snapshot: Specific snapshot to run for (optional)
    """
    snap_list = [snapshot] if snapshot else None
    result = await _run_simulation(model_id, mode="pf", snapshots=snap_list)
    result["deprecation_notice"] = _DEPRECATION_NOTICE_PF
    return result


@mcp.tool()
async def run_optimization(
    model_id: str,
    solver_name: str = "highs",
    formulation: str = "kirchhoff",
    extra_functionality: str | None = None,
) -> dict:
    """[Deprecated] Run optimization. Use run_simulation(mode='optimize') instead.

    Args:
        model_id: The model ID
        solver_name: Solver to use
        formulation: Network formulation
        extra_functionality: Custom constraint code
    """
    result = await _run_simulation(
        model_id,
        mode="optimize",
        solver_name=solver_name,
        formulation=formulation,
        extra_functionality=extra_functionality,
    )
    result["deprecation_notice"] = _DEPRECATION_NOTICE_OPT
    return result
