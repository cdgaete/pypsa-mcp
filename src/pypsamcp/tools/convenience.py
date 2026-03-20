"""Deprecated convenience wrappers: add_bus, add_generator, add_load, add_line.

These tools delegate to add_component internally. They will be removed in a
future release. Use add_component directly instead.
"""

from pypsamcp.core import mcp
from pypsamcp.tools.components import add_component as _add_component

_DEPRECATION_NOTICE = (
    "This tool is deprecated and will be removed in a future release. "
    "Use add_component() instead."
)


@mcp.tool()
async def add_bus(
    model_id: str,
    bus_id: str,
    v_nom: float,
    x: float = 0.0,
    y: float = 0.0,
    carrier: str = "AC",
    country: str | None = None,
) -> dict:
    """[Deprecated] Add a bus to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        bus_id: The ID for the new bus
        v_nom: Nominal voltage in kV
        x: x-coordinate for plotting
        y: y-coordinate for plotting
        carrier: Energy carrier (e.g., "AC", "DC")
        country: Country code if applicable
    """
    params = {"v_nom": v_nom, "x": x, "y": y, "carrier": carrier}
    if country is not None:
        params["location"] = country  # PyPSA 1.x renamed 'country' to 'location'
    result = await _add_component(model_id, "Bus", bus_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result


@mcp.tool()
async def add_generator(
    model_id: str,
    generator_id: str,
    bus: str,
    p_nom: float = 0.0,
    p_nom_extendable: bool = False,
    capital_cost: float | None = None,
    marginal_cost: float | None = None,
    carrier: str | None = None,
    efficiency: float = 1.0,
) -> dict:
    """[Deprecated] Add a generator to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        generator_id: The ID for the new generator
        bus: The bus ID to connect to
        p_nom: Nominal power capacity in MW
        p_nom_extendable: Whether capacity can be expanded
        capital_cost: Investment cost in currency/MW
        marginal_cost: Operational cost in currency/MWh
        carrier: Energy carrier (e.g., "wind", "solar")
        efficiency: Generator efficiency (0 to 1)
    """
    params = {
        "bus": bus,
        "p_nom": p_nom,
        "p_nom_extendable": p_nom_extendable,
        "efficiency": efficiency,
    }
    if capital_cost is not None:
        params["capital_cost"] = capital_cost
    if marginal_cost is not None:
        params["marginal_cost"] = marginal_cost
    if carrier is not None:
        params["carrier"] = carrier
    result = await _add_component(model_id, "Generator", generator_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result


@mcp.tool()
async def add_load(
    model_id: str,
    load_id: str,
    bus: str,
    p_set: float = 0.0,
    q_set: float = 0.0,
) -> dict:
    """[Deprecated] Add a load to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        load_id: The ID for the new load
        bus: The bus ID to connect to
        p_set: Active power demand in MW
        q_set: Reactive power demand in MVAr
    """
    params = {"bus": bus, "p_set": p_set, "q_set": q_set}
    result = await _add_component(model_id, "Load", load_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result


@mcp.tool()
async def add_line(
    model_id: str,
    line_id: str,
    bus0: str,
    bus1: str,
    x: float,
    r: float = 0.0,
    s_nom: float = 0.0,
    s_nom_extendable: bool = False,
    capital_cost: float | None = None,
    length: float | None = None,
) -> dict:
    """[Deprecated] Add a transmission line to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        line_id: The ID for the new line
        bus0: The ID of the first bus
        bus1: The ID of the second bus
        x: Reactance in ohm
        r: Resistance in ohm
        s_nom: Nominal apparent power capacity in MVA
        s_nom_extendable: Whether capacity can be expanded
        capital_cost: Investment cost in currency/MW
        length: Line length in km
    """
    params = {
        "bus0": bus0,
        "bus1": bus1,
        "x": x,
        "r": r,
        "s_nom": s_nom,
        "s_nom_extendable": s_nom_extendable,
    }
    if capital_cost is not None:
        params["capital_cost"] = capital_cost
    if length is not None:
        params["length"] = length
    result = await _add_component(model_id, "Line", line_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result
