"""Statistics and results tool."""

from pypsamcp.core import convert_to_serializable, get_energy_model, mcp

VALID_METRICS = [
    "system_cost", "capex", "opex", "fom", "overnight_cost",
    "installed_capacity", "optimal_capacity", "expanded_capacity",
    "installed_capex", "expanded_capex", "capacity_factor", "curtailment",
    "energy_balance", "supply", "withdrawal", "revenue", "market_value",
    "prices", "transmission",
]

# Metrics that do NOT accept aggregate_across_components
_NO_AGGREGATE_ACROSS = {"prices"}

# Metrics that only accept specific groupby values
_RESTRICTED_GROUPBY = {
    "prices": {"bus_carrier", False},
}


def _build_kwargs_for_metric(metric, shared_kwargs):
    """Build metric-specific kwargs, rejecting incompatible options up front."""
    kwargs = dict(shared_kwargs)

    if metric in _NO_AGGREGATE_ACROSS and kwargs.get("aggregate_across_components"):
        raise ValueError(
            f"Metric '{metric}' does not support aggregate_across_components. "
            f"Call get_statistics(metric='{metric}', aggregate_across_components=False) instead."
        )
    if metric in _NO_AGGREGATE_ACROSS:
        kwargs.pop("aggregate_across_components", None)

    if metric in _RESTRICTED_GROUPBY:
        allowed = _RESTRICTED_GROUPBY[metric]
        groupby = kwargs.get("groupby")
        if groupby not in allowed:
            allowed_str = ", ".join(repr(v) for v in sorted(allowed, key=str))
            raise ValueError(
                f"Metric '{metric}' only supports groupby={allowed_str}. "
                f"Got groupby={groupby!r}. "
                f"Call get_statistics(metric='{metric}', groupby='bus_carrier') instead."
            )

    return kwargs


def _call_metric(network, metric, shared_kwargs):
    """Call a single statistics metric and return the serialized result.

    Raises ValueError with an actionable message if the metric does not
    accept the provided kwargs.
    """
    kwargs = _build_kwargs_for_metric(metric, shared_kwargs)
    method = getattr(network.statistics, metric)
    result = method(**kwargs)
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
    shared_kwargs = {
        "groupby": groupby,
        "aggregate_across_components": aggregate_across_components,
    }
    if components is not None:
        shared_kwargs["components"] = components
    if carrier is not None:
        shared_kwargs["carrier"] = carrier
    if bus_carrier is not None:
        shared_kwargs["bus_carrier"] = bus_carrier
    if nice_names is not None:
        shared_kwargs["nice_names"] = nice_names
    if drop_zero is not None:
        shared_kwargs["drop_zero"] = drop_zero

    try:
        if metric == "all":
            combined = {}
            errors = {}
            for m in VALID_METRICS:
                try:
                    combined[m] = _call_metric(network, m, shared_kwargs)
                except Exception as e:
                    combined[m] = None
                    errors[m] = str(e)
            result = {
                "metric": "all",
                "model_id": model_id,
                "result": combined,
            }
            if errors:
                result["metric_errors"] = errors
            return result
        else:
            result = _call_metric(network, metric, shared_kwargs)
            return {
                "metric": metric,
                "model_id": model_id,
                "result": result,
                "unit_note": "Currency units depend on the cost inputs used when building the model.",
            }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
