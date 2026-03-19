"""Time and investment structure configuration."""

import pandas as pd

from pypsamcp.core import get_energy_model, mcp


@mcp.tool()
async def configure_time(
    model_id: str,
    mode: str,
    # mode="snapshots"
    snapshots: list[str] | None = None,
    weightings: dict | None = None,
    # mode="investment_periods"
    periods: list[int] | None = None,
    # mode="scenarios"
    scenarios: dict | list | None = None,
    # mode="risk_preference"
    alpha: float | None = None,
    omega: float | None = None,
) -> dict:
    """Configure time snapshots, investment periods, scenarios, or risk preference.

    Args:
        model_id: The model to configure
        mode: One of 'snapshots', 'investment_periods', 'scenarios', 'risk_preference'
        snapshots: ISO datetime strings (mode='snapshots')
        weightings: Snapshot weightings dict (mode='snapshots')
        periods: Investment years e.g. [2025, 2030] (mode='investment_periods')
        scenarios: Scenario definition (mode='scenarios')
        alpha: CVaR confidence level (mode='risk_preference')
        omega: Weight on CVaR term 0-1 (mode='risk_preference')
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        if mode == "snapshots":
            if snapshots is None:
                return {"error": "snapshots parameter is required for mode='snapshots'."}
            network.set_snapshots(pd.to_datetime(snapshots))
            if weightings:
                for key, val in weightings.items():
                    if key in network.snapshot_weightings.columns:
                        if isinstance(val, list):
                            network.snapshot_weightings[key] = val
                        else:
                            network.snapshot_weightings[key] = val
            msg = f"Snapshots configured for model '{model_id}'."

        elif mode == "investment_periods":
            if periods is None:
                return {"error": "periods parameter is required for mode='investment_periods'."}
            network.set_investment_periods(periods)
            msg = f"Investment periods {periods} configured for model '{model_id}'."

        elif mode == "scenarios":
            if scenarios is None:
                return {"error": "scenarios parameter is required for mode='scenarios'."}
            network.set_scenarios(scenarios)
            msg = f"Scenarios configured for model '{model_id}'."

        elif mode == "risk_preference":
            if alpha is None or omega is None:
                return {"error": "alpha and omega are required for mode='risk_preference'."}
            if not getattr(network, "has_scenarios", False):
                return {"error": "Scenarios must be set before configuring risk preference."}
            network.set_risk_preference(alpha, omega)
            msg = f"Risk preference (alpha={alpha}, omega={omega}) configured for model '{model_id}'."

        else:
            return {
                "error": f"Invalid mode '{mode}'. Must be one of: "
                "'snapshots', 'investment_periods', 'scenarios', 'risk_preference'."
            }

        # Build response
        has_ip = bool(getattr(network, "has_investment_periods", False))
        has_sc = bool(getattr(network, "has_scenarios", False))
        has_rp = bool(getattr(network, "has_risk_preference", False))

        result = {
            "mode": mode,
            "message": msg,
            "has_investment_periods": has_ip,
            "has_scenarios": has_sc,
            "has_risk_preference": has_rp,
            "snapshot_count": len(network.snapshots),
        }

        if has_ip:
            ipw = network.investment_period_weightings
            result["investment_period_weightings"] = ipw.to_dict()

        return result

    except Exception as e:
        return {"error": str(e)}
