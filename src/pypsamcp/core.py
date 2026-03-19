"""
PyPSA MCP Core functionality

This module provides the core MCP object, global model storage, and shared helpers.
Tool modules in pypsamcp.tools register themselves via @mcp.tool() on import.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
from contextlib import contextmanager
from fastmcp import FastMCP

# Redirect Python loggers to stderr
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
for _name in ("pypsa", "linopy", "highs"):
    logging.getLogger(_name).handlers = [logging.StreamHandler(sys.stderr)]
    logging.getLogger(_name).propagate = False


@contextmanager
def stdout_to_stderr():
    """Redirect fd 1 (stdout) to fd 2 (stderr) at the OS level.

    HiGHS writes directly via C stdio, bypassing Python's sys.stdout,
    so we must redirect at the file descriptor level to keep the MCP
    stdio stream clean.
    """
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    try:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


mcp = FastMCP(
    "pypsa-mcp",
    on_duplicate="error",
)

MODELS = {}

# CamelCase type name -> lowercase list_name. Excludes internal types (SubNetwork, Shape).
VALID_COMPONENT_TYPES = {
    "Bus": "buses",
    "Generator": "generators",
    "Load": "loads",
    "Line": "lines",
    "Link": "links",
    "StorageUnit": "storage_units",
    "Store": "stores",
    "Transformer": "transformers",
    "ShuntImpedance": "shunt_impedances",
    "Carrier": "carriers",
    "GlobalConstraint": "global_constraints",
    "LineType": "line_types",
    "TransformerType": "transformer_types",
}

# Reverse lookup: list_name -> CamelCase type name
_LIST_NAME_TO_TYPE = {v: k for k, v in VALID_COMPONENT_TYPES.items()}


def get_energy_model(model_id: str) -> pypsa.Network:
    """Get a model by ID from the global models dictionary."""
    if model_id not in MODELS:
        raise ValueError(
            f"Model with ID '{model_id}' not found. "
            f"Available models: {list(MODELS.keys())}"
        )
    return MODELS[model_id]


def validate_component_type(component_type: str) -> str:
    """Validate and normalize component_type to CamelCase. Returns the canonical name.

    Accepts both CamelCase ('Bus') and list_name ('buses') forms.
    Raises ValueError if invalid.
    """
    if component_type in VALID_COMPONENT_TYPES:
        return component_type
    if component_type in _LIST_NAME_TO_TYPE:
        return _LIST_NAME_TO_TYPE[component_type]
    raise ValueError(
        f"Invalid component type '{component_type}'. "
        f"Valid types: {list(VALID_COMPONENT_TYPES.keys())}"
    )


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
        "snapshots": len(network.snapshots),
    }


def convert_to_serializable(data):
    """Convert PyPSA data to JSON-serializable format."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    elif isinstance(data, pd.Series):
        return data.to_dict()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data


# Import tool modules to trigger @mcp.tool() registration.
# This MUST be at the bottom, after mcp and all helpers are defined.
import pypsamcp.tools  # noqa: E402, F401
