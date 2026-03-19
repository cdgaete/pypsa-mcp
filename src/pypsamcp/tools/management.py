"""Model management tools: create, list, delete, export summary."""

import pandas as pd
import pypsa

from pypsamcp.core import (
    MODELS,
    convert_to_serializable,
    generate_network_summary,
    get_energy_model,
    mcp,
)


@mcp.tool()
async def create_energy_model(
    model_id: str,
    name: str | None = None,
    override: bool = False,
) -> dict:
    """Create a new PyPSA energy model with the given ID.

    Args:
        model_id: A unique identifier for the model
        name: A descriptive name for the model (defaults to model_id if not provided)
        override: Whether to override an existing model with the same ID
    """
    if model_id in MODELS and not override:
        return {
            "error": f"Model with ID '{model_id}' already exists. Use override=True to replace it.",
            "available_models": list(MODELS.keys()),
        }

    network = pypsa.Network()
    network.name = name if name else model_id
    MODELS[model_id] = network

    return {
        "model_id": model_id,
        "name": network.name,
        "message": f"PyPSA energy model '{model_id}' created successfully.",
        "model_summary": generate_network_summary(network),
    }


@mcp.tool()
async def list_models() -> dict:
    """List all currently available PyPSA models."""
    model_list = []
    for model_id, network in MODELS.items():
        model_list.append({
            "model_id": model_id,
            "name": network.name,
            "summary": generate_network_summary(network),
        })

    return {"count": len(model_list), "models": model_list}


@mcp.tool()
async def delete_model(model_id: str) -> dict:
    """Delete a PyPSA model by ID.

    Args:
        model_id: The ID of the model to delete
    """
    if model_id not in MODELS:
        return {
            "error": f"Model with ID '{model_id}' not found. Available models: {list(MODELS.keys())}"
        }

    del MODELS[model_id]
    return {
        "message": f"Model '{model_id}' deleted successfully.",
        "remaining_models": list(MODELS.keys()),
    }


@mcp.tool()
async def export_model_summary(model_id: str) -> dict:
    """Export a comprehensive summary of the model.

    Args:
        model_id: The ID of the model to export
    """
    try:
        network = get_energy_model(model_id)

        summary = {
            "model_id": model_id,
            "name": network.name,
            "components": {},
            "has_investment_periods": bool(getattr(network, "has_investment_periods", False)),
            "has_scenarios": bool(getattr(network, "has_scenarios", False)),
            "has_risk_preference": bool(getattr(network, "has_risk_preference", False)),
            "investment_periods": (
                list(network.investment_periods)
                if getattr(network, "has_investment_periods", False)
                else []
            ),
            "scenarios": (
                list(network.scenarios)
                if getattr(network, "has_scenarios", False)
                else []
            ),
        }

        # Add non-empty component info
        for type_name, list_name in [
            ("Bus", "buses"),
            ("Generator", "generators"),
            ("Load", "loads"),
            ("Line", "lines"),
            ("Link", "links"),
            ("StorageUnit", "storage_units"),
            ("Store", "stores"),
            ("Transformer", "transformers"),
            ("ShuntImpedance", "shunt_impedances"),
            ("Carrier", "carriers"),
            ("GlobalConstraint", "global_constraints"),
        ]:
            df = getattr(network, list_name)
            if not df.empty:
                summary["components"][type_name] = {
                    "count": len(df),
                    "attributes": list(df.columns),
                    "ids": df.index.tolist(),
                }

        # Snapshot info
        if len(network.snapshots) > 0:
            freq = None
            if len(network.snapshots) >= 3:
                freq = pd.infer_freq(network.snapshots)
            summary["snapshots"] = {
                "count": len(network.snapshots),
                "start": str(network.snapshots[0]),
                "end": str(network.snapshots[-1]),
                "frequency": str(freq) if freq else "irregular",
            }

        return {"message": f"Model summary generated for '{model_id}'.", "summary": summary}
    except Exception as e:
        return {"error": str(e)}
