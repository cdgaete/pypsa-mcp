"""
PyPSA MCP Core functionality

This module provides the core functionality for the PyPSA MCP server.
It includes the MCP object and global model storage.
"""

import numpy as np
import pandas as pd
import pypsa
import textwrap
from fastmcp import FastMCP


mcp = FastMCP(
    "pypsa-mcp",
    on_duplicate_tools="error",
    description="Create, analyze, and optimize energy system models using PyPSA",
    dependencies=["pypsa", "pandas", "numpy"]
)

MODELS = {}


def get_model(model_id: str) -> pypsa.Network:
    """Get a model by ID from the global models dictionary."""
    if model_id not in MODELS:
        raise ValueError(f"Model with ID '{model_id}' not found. Available models: {list(MODELS.keys())}")
    return MODELS[model_id]

def generate_network_summary(network: pypsa.Network) -> dict:
    """Generate a summary of a PyPSA network."""
    return {
        "name": network.name,
        "buses": len(network.buses),
        "generators": len(network.generators),
        "storage_units": len(network.storage_units),
        "links": len(network.links),
        "lines": len(network.lines),
        "transformers": len(network.transformers),
        "snapshots": len(network.snapshots) if hasattr(network, "snapshots") and network.snapshots is not None else 0,
        "components": list(network.components.keys())
    }

def convert_to_serializable(data):
    """Convert PyPSA data to JSON-serializable format."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    elif isinstance(data, pd.Series):
        return data.to_dict()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.int64 | np.float64):
        return data.item()
    else:
        return data


# --- Basic model management tools ---

@mcp.tool()
async def create_energy_model(
    model_id: str,
    name: str | None = None,
    override: bool = False
) -> dict:
    """
    Create a new PyPSA energy model with the given ID.
    
    Args:
        model_id: A unique identifier for the model
        name: A descriptive name for the model (defaults to model_id if not provided)
        override: Whether to override an existing model with the same ID
        
    Returns:
        Information about the created model
    """
    if model_id in MODELS and not override:
        return {
            "error": f"Model with ID '{model_id}' already exists. Use override=True to replace it.",
            "available_models": list(MODELS.keys())
        }
    
    # Create a new PyPSA network
    network = pypsa.Network()
    network.name = name if name else model_id
    
    # Store the network in the global dictionary
    MODELS[model_id] = network
    
    return {
        "model_id": model_id,
        "name": network.name,
        "message": f"PyPSA energy model '{model_id}' created successfully.",
        "model_summary": generate_network_summary(network)
    }

@mcp.tool()
async def list_models() -> dict:
    """
    List all currently available PyPSA models.
    
    Returns:
        A dictionary containing the list of models and their summaries
    """
    model_list = []
    for model_id, network in MODELS.items():
        model_list.append({
            "model_id": model_id,
            "name": network.name,
            "summary": generate_network_summary(network)
        })
    
    return {
        "count": len(model_list),
        "models": model_list
    }

@mcp.tool()
async def delete_model(model_id: str) -> dict:
    """
    Delete a PyPSA model by ID.
    
    Args:
        model_id: The ID of the model to delete
        
    Returns:
        Status message
    """
    if model_id not in MODELS:
        return {
            "error": f"Model with ID '{model_id}' not found. Available models: {list(MODELS.keys())}"
        }
    
    # Delete the model
    del MODELS[model_id]
    
    return {
        "message": f"Model '{model_id}' deleted successfully.",
        "remaining_models": list(MODELS.keys())
    }

# --- Component creation tools ---

@mcp.tool()
async def add_bus(
    model_id: str,
    bus_id: str,
    v_nom: float,
    x: float | None = 0.0,
    y: float | None = 0.0,
    carrier: str | None = "AC",
    country: str | None = None
) -> dict:
    """
    Add a bus to a PyPSA model.
    
    Args:
        model_id: The ID of the model to add the bus to
        bus_id: The ID for the new bus
        v_nom: Nominal voltage of the bus in kV
        x: x-coordinate for plotting
        y: y-coordinate for plotting
        carrier: Energy carrier (e.g., "AC", "DC")
        country: Country code if applicable
        
    Returns:
        Information about the added bus
    """
    try:
        network = get_model(model_id)
        
        # Check if bus already exists
        if bus_id in network.buses.index:
            return {"error": f"Bus '{bus_id}' already exists in model '{model_id}'."}
        
        # Add the bus
        network.add("Bus", 
                    bus_id,
                    v_nom=v_nom,
                    x=x,
                    y=y,
                    carrier=carrier,
                    country=country)
        
        return {
            "message": f"Bus '{bus_id}' added successfully to model '{model_id}'.",
            "bus_data": convert_to_serializable(network.buses.loc[bus_id]),
            "total_buses": len(network.buses)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def add_generator(
    model_id: str,
    generator_id: str,
    bus: str,
    p_nom: float = 0.0,
    p_nom_extendable: bool = False,
    capital_cost: float | None = None,
    marginal_cost: float | None = None,
    carrier: str | None= None,
    efficiency: float | None = 1.0
) -> dict:
    """
    Add a generator to a PyPSA model.
    
    Args:
        model_id: The ID of the model to add the generator to
        generator_id: The ID for the new generator
        bus: The bus ID to connect the generator to
        p_nom: Nominal power capacity in MW
        p_nom_extendable: Whether the capacity can be expanded in optimization
        capital_cost: Investment cost in currency/MW
        marginal_cost: Operational cost in currency/MWh
        carrier: Energy carrier (e.g., "wind", "solar", "coal")
        efficiency: Generator efficiency (from 0 to 1)
        
    Returns:
        Information about the added generator
    """
    try:
        network = get_model(model_id)
        
        # Check if bus exists
        if bus not in network.buses.index:
            return {"error": f"Bus '{bus}' does not exist in model '{model_id}'."}
        
        # Check if generator already exists
        if generator_id in network.generators.index:
            return {"error": f"Generator '{generator_id}' already exists in model '{model_id}'."}
        
        # Add the generator
        network.add("Generator",
                    generator_id,
                    bus=bus,
                    p_nom=p_nom,
                    p_nom_extendable=p_nom_extendable,
                    capital_cost=capital_cost,
                    marginal_cost=marginal_cost,
                    carrier=carrier,
                    efficiency=efficiency)
        
        return {
            "message": f"Generator '{generator_id}' added successfully to model '{model_id}'.",
            "generator_data": convert_to_serializable(network.generators.loc[generator_id]),
            "total_generators": len(network.generators)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def add_load(
    model_id: str,
    load_id: str,
    bus: str,
    p_set: float | list[float] = 0.0,
    q_set: float | list[float] = 0.0,
    snapshots: list[str] | None = None
) -> dict:
    """
    Add a load to a PyPSA model.
    
    Args:
        model_id: The ID of the model to add the load to
        load_id: The ID for the new load
        bus: The bus ID to connect the load to
        p_set: Active power demand in MW (scalar or time series)
        q_set: Reactive power demand in MVAr (scalar or time series)
        snapshots: List of time periods if providing time series data
        
    Returns:
        Information about the added load
    """
    try:
        network = get_model(model_id)
        
        # Check if bus exists
        if bus not in network.buses.index:
            return {"error": f"Bus '{bus}' does not exist in model '{model_id}'."}
        
        # Check if load already exists
        if load_id in network.loads.index:
            return {"error": f"Load '{load_id}' already exists in model '{model_id}'."}
        
        # Add the load
        network.add("Load",
                   load_id,
                   bus=bus)
        
        # Handle time series data
        if isinstance(p_set, list) or isinstance(q_set, list):
            if snapshots is None:
                return {"error": "Snapshots must be provided when adding time series load data."}
                
            # Set up snapshots if not already defined
            if network.snapshots is None or len(network.snapshots) == 0:
                network.set_snapshots(pd.to_datetime(snapshots))
            
            # Convert to pandas Series for time series
            if isinstance(p_set, list):
                p_series = pd.Series(p_set, index=network.snapshots[:len(p_set)])
                network.loads_t.p_set[load_id] = p_series
            
            if isinstance(q_set, list):
                q_series = pd.Series(q_set, index=network.snapshots[:len(q_set)])
                network.loads_t.q_set[load_id] = q_series
        else:
            # Set static values
            network.loads.loc[load_id, "p_set"] = p_set
            network.loads.loc[load_id, "q_set"] = q_set
        
        return {
            "message": f"Load '{load_id}' added successfully to model '{model_id}'.",
            "load_data": convert_to_serializable(network.loads.loc[load_id]),
            "total_loads": len(network.loads)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def add_line(
    model_id: str,
    line_id: str,
    bus0: str,
    bus1: str,
    x: float,
    r: float = 0.0,
    g: float = 0.0,
    b: float = 0.0,
    s_nom: float = 0.0,
    s_nom_extendable: bool = False,
    capital_cost: float | None = None,
    length: float | None = None
) -> dict:
    """
    Add a transmission line to a PyPSA model.
    
    Args:
        model_id: The ID of the model to add the line to
        line_id: The ID for the new line
        bus0: The ID of the first bus
        bus1: The ID of the second bus
        x: Reactance in ohm
        r: Resistance in ohm
        g: Shunt conductance in S
        b: Shunt susceptance in S
        s_nom: Nominal apparent power capacity in MVA
        s_nom_extendable: Whether the capacity can be expanded in optimization
        capital_cost: Investment cost in currency/MW
        length: Line length in km
        
    Returns:
        Information about the added line
    """
    try:
        network = get_model(model_id)
        
        # Check if buses exist
        if bus0 not in network.buses.index:
            return {"error": f"Bus '{bus0}' does not exist in model '{model_id}'."}
        if bus1 not in network.buses.index:
            return {"error": f"Bus '{bus1}' does not exist in model '{model_id}'."}
        
        # Check if line already exists
        if line_id in network.lines.index:
            return {"error": f"Line '{line_id}' already exists in model '{model_id}'."}
        
        # Add the line
        network.add("Line",
                   line_id,
                   bus0=bus0,
                   bus1=bus1,
                   x=x,
                   r=r,
                   g=g,
                   b=b,
                   s_nom=s_nom,
                   s_nom_extendable=s_nom_extendable,
                   capital_cost=capital_cost,
                   length=length)
        
        return {
            "message": f"Line '{line_id}' added successfully to model '{model_id}'.",
            "line_data": convert_to_serializable(network.lines.loc[line_id]),
            "total_lines": len(network.lines)
        }
    except Exception as e:
        return {"error": str(e)}

# --- Time handling tools ---

@mcp.tool()
async def set_snapshots(
    model_id: str,
    snapshots: list[str]
) -> dict:
    """
    Set time snapshots for a PyPSA model.
    
    Args:
        model_id: The ID of the model to set snapshots for
        snapshots: List of datetime strings for the snapshots
        
    Returns:
        Information about the set snapshots
    """
    try:
        network = get_model(model_id)
        
        # Convert to datetime and set snapshots
        snapshots_dt = pd.to_datetime(snapshots)
        network.set_snapshots(snapshots_dt)
        
        return {
            "message": f"Snapshots set successfully for model '{model_id}'.",
            "snapshots": snapshots,
            "count": len(snapshots)
        }
    except Exception as e:
        return {"error": str(e)}

# --- Analysis tools ---

@mcp.tool()
async def run_powerflow(
    model_id: str,
    snapshot: str | None = None
) -> dict:
    """
    Run a power flow calculation on a PyPSA model.
    
    Args:
        model_id: The ID of the model to run power flow on
        snapshot: Specific snapshot to run power flow for (optional)
        
    Returns:
        Power flow results
    """
    try:
        network = get_model(model_id)
        
        # Validate snapshot if provided
        if snapshot is not None:
            snapshot_dt = pd.to_datetime(snapshot)
            if snapshot_dt not in network.snapshots:
                return {"error": f"Snapshot '{snapshot}' not found in model '{model_id}'."}
            
            # Run power flow for a specific snapshot
            network.pf(snapshot_dt)
            
            # Collect results
            results = {
                "bus_v_mag_pu": convert_to_serializable(network.buses_t.v_mag_pu.loc[snapshot_dt]),
                "bus_v_ang": convert_to_serializable(network.buses_t.v_ang.loc[snapshot_dt]),
                "line_p0": convert_to_serializable(network.lines_t.p0.loc[snapshot_dt]),
                "line_p1": convert_to_serializable(network.lines_t.p1.loc[snapshot_dt]),
                "snapshot": snapshot
            }
        else:
            # Run power flow for all snapshots
            network.pf()
            
            # Collect results for all snapshots (could be large)
            results = {
                "bus_v_mag_pu": convert_to_serializable(network.buses_t.v_mag_pu),
                "bus_v_ang": convert_to_serializable(network.buses_t.v_ang),
                "line_p0": convert_to_serializable(network.lines_t.p0),
                "line_p1": convert_to_serializable(network.lines_t.p1),
                "snapshots_count": len(network.snapshots)
            }
        
        return {
            "message": f"Power flow calculation successful for model '{model_id}'.",
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def run_optimization(
    model_id: str,
    solver_name: str = "highs",
    formulation: str = "kirchhoff",
    extra_functionality: str | None = None
) -> dict:
    """
    Run an optimization on a PyPSA model.
    
    Args:
        model_id: The ID of the model to run optimization on
        solver_name: The solver to use (e.g., "highs", "glpk", "cplex", "gurobi")
        formulation: Network equations to use ("kirchhoff" or "ptdf")
        extra_functionality: Python code string with extra constraints (optional)
        
    Returns:
        Optimization results
    """
    try:
        network = get_model(model_id)
        
        # Validate formulation
        if formulation not in ["kirchhoff", "ptdf"]:
            return {"error": f"Invalid formulation '{formulation}'. Must be 'kirchhoff' or 'ptdf'."}
        
        # Prepare extra_functionality if provided
        extra_func = None
        if extra_functionality:
            try:
                # Compile the code string into a function
                namespace = {}
                exec(f"def extra_func(n, snapshots):\n{textwrap.indent(extra_functionality, '    ')}", namespace)
                extra_func = namespace["extra_func"]
            except Exception as e:
                return {"error": f"Error in extra_functionality code: {str(e)}"}
        
        # Initialize results dictionary for optimization results
        if not hasattr(network, 'results'):
            network.results = {}
            
        # Run the optimization using the modern optimize method instead of lopf
        network.optimize(
            solver_name=solver_name,
            formulation=formulation,
            extra_functionality=extra_func
        )
        
        # Collect optimization results
        results = {
            "objective": network.objective if hasattr(network, 'objective') else None,
            "termination_condition": str(network.results["termination_condition"]) if 'termination_condition' in network.results else None,
            "status": network.results["status"] if 'status' in network.results else None,
            "generator_p": convert_to_serializable(network.generators_t.p if hasattr(network.generators_t, "p") else {}),
            "storage_p": convert_to_serializable(network.storage_units_t.p if hasattr(network.storage_units_t, "p") else {}),
            "line_p0": convert_to_serializable(network.lines_t.p0 if hasattr(network.lines_t, "p0") else {}),
            "shadow_prices": convert_to_serializable(network.buses_t.marginal_price if hasattr(network.buses_t, "marginal_price") else {})
        }
        
        # Check for expansion results
        if hasattr(network, "lines") and "s_nom_opt" in network.lines.columns:
            results["line_expansion"] = convert_to_serializable(network.lines["s_nom_opt"])
        
        if hasattr(network, "generators") and "p_nom_opt" in network.generators.columns:
            results["generator_expansion"] = convert_to_serializable(network.generators["p_nom_opt"])
        
        if hasattr(network, "storage_units") and "p_nom_opt" in network.storage_units.columns:
            results["storage_expansion"] = convert_to_serializable(network.storage_units["p_nom_opt"])
        
        return {
            "message": f"Optimization successful for model '{model_id}'.",
            "objective_value": float(network.objective) if hasattr(network, 'objective') else None,
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def export_model_summary(
    model_id: str
) -> dict:
    """
    Export a comprehensive summary of the model.
    
    Args:
        model_id: The ID of the model to export
        
    Returns:
        A detailed summary of the model
    """
    try:
        network = get_model(model_id)
        
        # Generate summary
        summary = {
            "model_id": model_id,
            "name": network.name,
            "components": {}
        }
        
        # Add component counts
        for comp_name, comp_attrs in network.components.items():
            list_name = comp_attrs["list_name"]
            if hasattr(network, list_name):
                df = getattr(network, list_name)
                
                # Skip empty components
                if df.empty:
                    continue
                
                # Add component info
                summary["components"][comp_name] = {
                    "count": len(df),
                    "attributes": list(df.columns),
                    "ids": df.index.tolist()
                }
        
        # Add snapshot info if available
        if network.snapshots is not None and len(network.snapshots) > 0:
            summary["snapshots"] = {
                "count": len(network.snapshots),
                "start": str(network.snapshots[0]),
                "end": str(network.snapshots[-1]),
                "frequency": str(pd.infer_freq(network.snapshots)) if pd.infer_freq(network.snapshots) else "irregular"
            }
        
        return {
            "message": f"Model summary generated for '{model_id}'.",
            "summary": summary
        }
    except Exception as e:
        return {"error": str(e)}