"""
PyPSA MCP Core functionality

This module provides the core functionality for the PyPSA MCP server.
It includes the MCP object and global model storage.
"""

import numpy as np
import pandas as pd
import pypsa
from fastmcp import FastMCP


mcp = FastMCP(
    "PyPSA Energy Modeler",
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
