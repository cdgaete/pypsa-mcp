"""I/O and network operations tool."""

import pandas as pd
import pypsa

from pypsamcp.core import MODELS, generate_network_summary, get_energy_model, mcp

VALID_OPERATIONS = {
    "export_netcdf", "export_csv", "import_netcdf", "import_csv",
    "merge", "copy", "consistency_check",
}


@mcp.tool()
async def network_io(
    model_id: str,
    operation: str,
    # export
    path: str | None = None,
    float32: bool = False,
    compression: dict | None = None,
    export_standard_types: bool = False,
    # import
    skip_time: bool = False,
    # merge
    other_model_id: str | None = None,
    output_model_id: str | None = None,
    components_to_skip: list[str] | None = None,
    with_time: bool = True,
    # copy/subset
    snapshots: list[str] | None = None,
    investment_periods: list | None = None,
    buses: list[str] | None = None,
) -> dict:
    """Perform I/O and network manipulation operations.

    Operations:
    - export_netcdf: Export network to NetCDF file (requires path)
    - export_csv: Export network to CSV folder (requires path)
    - import_netcdf: Import network from NetCDF file (requires path), stores as model_id
    - import_csv: Import network from CSV folder (requires path), stores as model_id
    - merge: Merge another network into this one (requires other_model_id, output_model_id)
    - copy: Copy this network (optionally subset by snapshots/buses), stores as output_model_id
    - consistency_check: Run consistency check on the network
    """
    if operation not in VALID_OPERATIONS:
        return {"error": f"Invalid operation '{operation}'. Valid operations: {sorted(VALID_OPERATIONS)}"}

    try:
        if operation == "export_netcdf":
            network = get_energy_model(model_id)
            if not path:
                return {"error": "path is required for export_netcdf"}
            kwargs = {"float32": float32}
            if compression is not None:
                kwargs["compression"] = compression
            network.export_to_netcdf(path, **kwargs)
            return {"message": f"Network '{model_id}' exported to NetCDF at '{path}'"}

        elif operation == "export_csv":
            network = get_energy_model(model_id)
            if not path:
                return {"error": "path is required for export_csv"}
            network.export_to_csv_folder(path, export_standard_types=export_standard_types)
            return {"message": f"Network '{model_id}' exported to CSV folder at '{path}'"}

        elif operation == "import_netcdf":
            if not path:
                return {"error": "path is required for import_netcdf"}
            network = pypsa.Network()
            network.import_from_netcdf(path, skip_time=skip_time)
            MODELS[model_id] = network
            summary = generate_network_summary(network)
            return {"message": f"Network imported from NetCDF at '{path}' as '{model_id}'", "summary": summary}

        elif operation == "import_csv":
            if not path:
                return {"error": "path is required for import_csv"}
            network = pypsa.Network()
            network.import_from_csv_folder(path, skip_time=skip_time)
            MODELS[model_id] = network
            summary = generate_network_summary(network)
            return {"message": f"Network imported from CSV folder at '{path}' as '{model_id}'", "summary": summary}

        elif operation == "merge":
            network = get_energy_model(model_id)
            if not other_model_id:
                return {"error": "other_model_id is required for merge"}
            if not output_model_id:
                return {"error": "output_model_id is required for merge"}
            other = get_energy_model(other_model_id)
            merged = network.merge(
                other,
                inplace=False,
                with_time=with_time,
                components_to_skip=components_to_skip,
            )
            MODELS[output_model_id] = merged
            summary = generate_network_summary(merged)
            return {"message": f"Networks '{model_id}' and '{other_model_id}' merged as '{output_model_id}'", "summary": summary}

        elif operation == "copy":
            network = get_energy_model(model_id)
            if not output_model_id:
                return {"error": "output_model_id is required for copy"}
            kwargs = {}
            if snapshots is not None:
                kwargs["snapshots"] = pd.DatetimeIndex(snapshots)
            if investment_periods is not None:
                kwargs["investment_periods"] = investment_periods
            copied = network.copy(**kwargs)
            if buses is not None:
                copied = copied.copy()
                # Filter to only the specified buses and their connected components
                buses_to_remove = [b for b in copied.buses.index if b not in buses]
                for bus_name in buses_to_remove:
                    copied.remove("Bus", bus_name)
            MODELS[output_model_id] = copied
            summary = generate_network_summary(copied)
            return {"message": f"Network '{model_id}' copied as '{output_model_id}'", "summary": summary}

        elif operation == "consistency_check":
            network = get_energy_model(model_id)
            try:
                network.consistency_check()
                return {"message": f"Consistency check passed for network '{model_id}'"}
            except Exception as e:
                return {"message": f"Consistency check failed for network '{model_id}': {e}"}

    except Exception as e:
        return {"error": str(e)}
