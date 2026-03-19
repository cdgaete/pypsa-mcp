"""Discovery tools: list_component_types, describe_component."""

import warnings

import pypsa

from pypsamcp.core import (
    MODELS,
    VALID_COMPONENT_TYPES,
    mcp,
    validate_component_type,
)

COMPONENT_DESCRIPTIONS = {
    "Bus": "Electrical node connecting components",
    "Generator": "Power source with cost and capacity parameters",
    "Load": "Power demand at a bus",
    "Line": "AC transmission line between two buses",
    "Link": "Controllable branch: HVDC, heat pumps, electrolyzers, sector coupling",
    "StorageUnit": "Coupled power-energy storage: battery, pumped hydro, reservoir",
    "Store": "Energy reservoir decoupled from power: H2 tank, heat store",
    "Transformer": "Voltage-level coupling with tap ratio and phase shift",
    "ShuntImpedance": "Shunt conductance/susceptance at a bus",
    "Carrier": "Energy carrier with CO2 emissions, color, growth limits",
    "GlobalConstraint": "System-wide constraint: CO2 cap, primary energy limit, capacity target",
    "LineType": "Standard AC line type library (r/x/c per km)",
    "TransformerType": "Standard transformer type library",
}


@mcp.tool()
async def list_component_types() -> dict:
    """List all available PyPSA component types with descriptions.

    Returns the catalog of 13 user-facing component types. No model required.
    """
    return {
        "component_types": [
            {
                "type": type_name,
                "list_name": list_name,
                "description": COMPONENT_DESCRIPTIONS[type_name],
            }
            for type_name, list_name in VALID_COMPONENT_TYPES.items()
        ]
    }


@mcp.tool()
async def describe_component(
    component_type: str,
    include_defaults: bool = True,
) -> dict:
    """Describe the full input-parameter schema for a PyPSA component type.

    Args:
        component_type: Component type name (e.g. 'Generator', 'Bus', or 'generators')
        include_defaults: Whether to include default values in the response
    """
    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    # Use an existing model or create a throwaway one
    if MODELS:
        network = next(iter(MODELS.values()))
    else:
        network = pypsa.Network()

    # Get component attrs (suppress deprecation warning)
    comp = network.components[canonical]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        attrs = comp.attrs

    # Filter to Input attributes only
    input_mask = attrs["status"].str.startswith("Input")
    input_attrs = attrs[input_mask]

    required = []
    static = []
    varying = []

    for attr_name, row in input_attrs.iterrows():
        if attr_name == "name":
            continue  # Skip 'name' — it's the component_id

        entry = {
            "attr": attr_name,
            "unit": str(row.get("unit", "")),
            "description": str(row.get("description", "")),
            "typ": str(row.get("typ", "")),
        }
        if include_defaults:
            default_val = row.get("default", None)
            entry["default"] = str(default_val) if default_val is not None else None

        status = row["status"]
        if status == "Input (required)":
            required.append(entry)
        elif row.get("varying", False):
            varying.append(entry)
        else:
            static.append(entry)

    return {
        "component_type": canonical,
        "list_name": VALID_COMPONENT_TYPES[canonical],
        "required": required,
        "static": static,
        "varying": varying,
        "note": (
            "Pass 'static' and 'varying' params to add_component(params={}). "
            "Pass time series for 'varying' params to add_component(time_series={})."
        ),
    }
