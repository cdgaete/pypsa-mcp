"""Statistics and results tool."""

from pypsamcp.core import convert_to_serializable, get_energy_model, mcp

VALID_METRICS = [
    "system_cost", "capex", "opex", "fom", "overnight_cost",
    "installed_capacity", "optimal_capacity", "expanded_capacity",
    "installed_capex", "expanded_capex", "capacity_factor", "curtailment",
    "energy_balance", "supply", "withdrawal", "revenue", "market_value",
    "prices", "transmission",
]


import inspect


def _call_metric(network, metric, kwargs):
    """Call a single statistics metric and return the serialized result.

    Some metrics (e.g. prices, transmission) don't accept all shared kwargs
    like aggregate_across_components or certain groupby values. We try with
    full kwargs first, then progressively strip incompatible ones on error.
    """
    method = getattr(network.statistics, metric)
    try:
        result = method(**kwargs)
        return convert_to_serializable(result)
    except TypeError:
        # Strip aggregate_across_components and retry
        stripped = {k: v for k, v in kwargs.items() if k != "aggregate_across_components"}
        try:
            result = method(**stripped)
            return convert_to_serializable(result)
        except (TypeError, ValueError):
            # Last resort: call with no kwargs
            result = method()
            return convert_to_serializable(result)


@mcp.tool()
async def get_statistics(
    model_id: str,
    metric: str = "system_cost",
    components: list[str] | None = None,
    carrier: list[str] | None = None,
    bus_carrier: list[str] | None = None,
    groupby: str | list[str] = "carrier",
    aggregate_across_components: bool = False,
    nice_names: bool | None = None,
    drop_zero: bool | None = None,
) -> dict:
    """Get statistics from a solved PyPSA model.

    Args:
        model_id: The model to query
        metric: Metric name or 'all'. One of: system_cost, capex, opex, fom,
                overnight_cost, installed_capacity, optimal_capacity,
                expanded_capacity, installed_capex, expanded_capex,
                capacity_factor, curtailment, energy_balance, supply,
                withdrawal, revenue, market_value, prices, transmission
        components: Filter to specific component types
        carrier: Filter by carrier name
        bus_carrier: Filter by bus carrier
        groupby: Aggregation grouping (e.g. 'carrier', 'bus')
        aggregate_across_components: Merge component types into one row per carrier
        nice_names: Use human-readable carrier names
        drop_zero: Omit zero-value rows
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    # Check if solved — n.objective is None on unsolved networks
    if network.objective is None:
        return {
            "error": f"Model '{model_id}' has not been solved. "
            "Run run_simulation(mode='optimize') first."
        }

    if metric != "all" and metric not in VALID_METRICS:
        return {
            "error": f"Invalid metric '{metric}'. Must be one of: {VALID_METRICS + ['all']}"
        }

    # Build shared kwargs, only including non-None values
    kwargs = {"groupby": groupby, "aggregate_across_components": aggregate_across_components}
    if components is not None:
        kwargs["components"] = components
    if carrier is not None:
        kwargs["carrier"] = carrier
    if bus_carrier is not None:
        kwargs["bus_carrier"] = bus_carrier
    if nice_names is not None:
        kwargs["nice_names"] = nice_names
    if drop_zero is not None:
        kwargs["drop_zero"] = drop_zero

    try:
        if metric == "all":
            combined = {}
            for m in VALID_METRICS:
                try:
                    combined[m] = _call_metric(network, m, kwargs)
                except Exception:
                    combined[m] = None
            return {
                "metric": "all",
                "model_id": model_id,
                "result": combined,
            }
        else:
            result = _call_metric(network, metric, kwargs)
            return {
                "metric": metric,
                "model_id": model_id,
                "result": result,
                "unit_note": "Currency units depend on the cost inputs used when building the model.",
            }
    except Exception as e:
        return {"error": str(e)}
