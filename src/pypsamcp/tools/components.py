"""Component CRUD tools: add, update, remove, query."""

import warnings

import pandas as pd

from pypsamcp.core import (
    VALID_COMPONENT_TYPES,
    convert_to_serializable,
    get_energy_model,
    mcp,
    validate_component_type,
)

# Component types that require bus references
_SINGLE_BUS = {"Generator", "Load", "StorageUnit", "Store", "ShuntImpedance"}
_DUAL_BUS = {"Line", "Transformer"}
_MULTI_BUS = {"Link"}  # bus0, bus1 required; bus2..busN optional
_NO_BUS_CHECK = {"Carrier", "GlobalConstraint", "LineType", "TransformerType", "Bus"}


def _get_component_attrs(network, canonical_type):
    """Get the attrs DataFrame for a component type, suppressing deprecation warnings."""
    comp = network.components[canonical_type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return comp.attrs


import re

# Pattern for Link's dynamically expanded port attributes (bus2, bus3, efficiency2, etc.)
_LINK_DYNAMIC_PORT_RE = re.compile(r"^(bus|efficiency|marginal_cost_quadratic)\d+$")


def _validate_params_are_input(attrs, params, canonical_type=None):
    """Check that all param keys are Input attributes. Returns list of bad keys.

    For Link components, bus2+/efficiency2+ are dynamically expanded by PyPSA
    and won't appear in the static attrs DataFrame — allow them through.
    """
    bad_keys = []
    for key in params:
        if key not in attrs.index:
            # Allow Link's dynamic port attributes (bus2, efficiency2, etc.)
            if canonical_type == "Link" and _LINK_DYNAMIC_PORT_RE.match(key):
                continue
            bad_keys.append(key)
        elif not attrs.loc[key, "status"].startswith("Input"):
            bad_keys.append(key)
    return bad_keys


def _get_bus_names(network):
    """Get bus names, handling both flat and MultiIndex (scenarios) indexes."""
    idx = network.buses.index
    if isinstance(idx, pd.MultiIndex) and "name" in idx.names:
        return set(idx.get_level_values("name"))
    return set(idx)


def _validate_bus_references(network, canonical_type, params):
    """Check that bus references in params resolve to existing buses."""
    if canonical_type in _NO_BUS_CHECK:
        return None

    bus_names = _get_bus_names(network)

    if canonical_type in _SINGLE_BUS:
        bus = params.get("bus")
        if bus and bus not in bus_names:
            return f"Bus '{bus}' does not exist."
    elif canonical_type in _DUAL_BUS:
        for attr in ("bus0", "bus1"):
            bus = params.get(attr)
            if bus and bus not in bus_names:
                return f"Bus '{bus}' (from {attr}) does not exist."
    elif canonical_type in _MULTI_BUS:
        for attr in ("bus0", "bus1"):
            bus = params.get(attr)
            if bus and bus not in bus_names:
                return f"Bus '{bus}' (from {attr}) does not exist."
        for key, val in params.items():
            if key.startswith("bus") and key[3:].isdigit() and int(key[3:]) >= 2:
                if val not in bus_names:
                    return f"Bus '{val}' (from {key}) does not exist."
    return None


def _validate_time_series_keys(attrs, time_series):
    """Check that all time_series keys are varying Input attributes."""
    bad_keys = []
    for key in time_series:
        if key not in attrs.index:
            bad_keys.append(key)
        elif not attrs.loc[key, "status"].startswith("Input"):
            bad_keys.append(key)
        elif not attrs.loc[key, "varying"]:
            bad_keys.append(key)
    return bad_keys


@mcp.tool()
async def add_component(
    model_id: str,
    component_type: str,
    component_id: str,
    params: dict | None = None,
    time_series: dict | None = None,
) -> dict:
    """Add any PyPSA component to a model.

    Args:
        model_id: The model to add the component to
        component_type: Component type (e.g. 'Generator', 'Bus', 'Line')
        component_id: Unique ID for the new component
        params: Static/scalar input parameters
        time_series: Time-varying parameters as {attr: [values...]}
    """
    params = params or {}
    time_series = time_series or {}

    # 1. Model exists
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    # 2. Valid component type
    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    attrs = _get_component_attrs(network, canonical)

    # 3. All params are Input attributes
    bad_keys = _validate_params_are_input(attrs, params, canonical)
    if bad_keys:
        return {
            "error": f"Invalid parameters {bad_keys}. These are not Input attributes. "
            f"Use describe_component('{canonical}') to see valid parameters."
        }

    # 4. Bus references resolve
    bus_error = _validate_bus_references(network, canonical, params)
    if bus_error:
        return {"error": bus_error}

    # 5. Component ID not already in use
    df = getattr(network, list_name)
    has_scenarios = isinstance(df.index, pd.MultiIndex) and "name" in df.index.names
    if has_scenarios:
        existing_names = df.index.get_level_values("name")
        already_exists = component_id in existing_names
    else:
        already_exists = component_id in df.index
    if already_exists:
        return {"error": f"{canonical} '{component_id}' already exists in model '{model_id}'."}

    # 6. Time series keys are varying Input attributes
    if time_series:
        bad_ts_keys = _validate_time_series_keys(attrs, time_series)
        if bad_ts_keys:
            return {
                "error": f"Invalid time_series keys {bad_ts_keys}. "
                f"These are not varying Input attributes. "
                f"Use describe_component('{canonical}') to see varying parameters."
            }

    # 7. Time series requires explicitly configured snapshots (not the default "now")
    if time_series:
        if not isinstance(network.snapshots, (pd.DatetimeIndex, pd.MultiIndex)):
            return {
                "error": "Cannot set time series: no snapshots configured. "
                "Use configure_time(mode='snapshots') or configure_time(mode='investment_periods') first."
            }

    # Execute
    try:
        network.add(canonical, component_id, **params)

        # Assign time series
        for attr, values in time_series.items():
            ts_df = getattr(network, f"{list_name}_t")
            series = pd.Series(values, index=network.snapshots[: len(values)])
            ts_df[attr][component_id] = series

        df = getattr(network, list_name)
        # With scenarios, the index is a MultiIndex (scenario, name) —
        # use xs to select by component name across scenarios.
        if isinstance(df.index, pd.MultiIndex) and "name" in df.index.names:
            component_row = df.xs(component_id, level="name")
            total = len(df.index.get_level_values("name").unique())
        else:
            component_row = df.loc[component_id]
            total = len(df)
        return {
            "message": f"{canonical} '{component_id}' added to model '{model_id}'.",
            "component_data": convert_to_serializable(component_row),
            "total_count": total,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def update_component(
    model_id: str,
    component_type: str,
    component_id: str,
    params: dict | None = None,
    time_series: dict | None = None,
) -> dict:
    """Update parameters of an existing component.

    Args:
        model_id: The model containing the component
        component_type: Component type (e.g. 'Generator', 'Bus')
        component_id: ID of the component to update
        params: Static/scalar parameters to update
        time_series: Time-varying parameters to update as {attr: [values...]}
    """
    params = params or {}
    time_series = time_series or {}

    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    attrs = _get_component_attrs(network, canonical)

    # Validate params are Input
    bad_keys = _validate_params_are_input(attrs, params, canonical)
    if bad_keys:
        return {
            "error": f"Invalid parameters {bad_keys}. "
            f"Use describe_component('{canonical}') to see valid parameters."
        }

    # Component must exist
    df = getattr(network, list_name)
    if component_id not in df.index:
        return {"error": f"{canonical} '{component_id}' not found in model '{model_id}'."}

    # Validate time series
    if time_series:
        bad_ts_keys = _validate_time_series_keys(attrs, time_series)
        if bad_ts_keys:
            return {"error": f"Invalid time_series keys {bad_ts_keys}."}
        if not isinstance(network.snapshots, pd.DatetimeIndex):
            return {"error": "Cannot set time series: no snapshots configured."}

    try:
        # Update static params
        for key, val in params.items():
            df.loc[component_id, key] = val

        # Update time series
        for attr, values in time_series.items():
            ts_df = getattr(network, f"{list_name}_t")
            series = pd.Series(values, index=network.snapshots[: len(values)])
            ts_df[attr][component_id] = series

        return {
            "message": f"{canonical} '{component_id}' updated in model '{model_id}'.",
            "component_data": convert_to_serializable(df.loc[component_id]),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def remove_component(
    model_id: str,
    component_type: str,
    component_id: str,
) -> dict:
    """Remove a component from a model.

    Args:
        model_id: The model containing the component
        component_type: Component type (e.g. 'Generator', 'Bus')
        component_id: ID of the component to remove
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    df = getattr(network, list_name)

    if component_id not in df.index:
        return {"error": f"{canonical} '{component_id}' not found in model '{model_id}'."}

    try:
        network.remove(canonical, component_id)
        df = getattr(network, list_name)
        return {
            "message": f"{canonical} '{component_id}' removed from model '{model_id}'.",
            "remaining_count": len(df),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def query_components(
    model_id: str,
    component_type: str,
    filters: dict | None = None,
) -> dict:
    """Query components of a given type, optionally filtered.

    Args:
        model_id: The model to query
        component_type: Component type (e.g. 'Generator', 'Bus')
        filters: Optional {attr: value} dict to filter rows
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    df = getattr(network, list_name)

    if filters:
        for attr, val in filters.items():
            if attr in df.columns:
                df = df[df[attr] == val]

    return {
        "component_type": canonical,
        "count": len(df),
        "components": convert_to_serializable(df),
    }
