"""Spatial and temporal clustering tool."""

from pypsamcp.core import MODELS, get_energy_model, mcp

VALID_SPATIAL_METHODS = {"kmeans", "hac", "greedy_modularity"}
VALID_TEMPORAL_METHODS = {"resample", "downsample", "segment", "snapshot_map"}


@mcp.tool()
async def cluster_network(
    model_id: str,
    domain: str,
    method: str,
    output_model_id: str,
    # spatial
    n_clusters: int | None = None,
    bus_weightings: dict | None = None,
    line_length_factor: float = 1.0,
    affinity: str = "euclidean",
    linkage: str = "ward",
    # spatial cluster_options
    with_time: bool = True,
    aggregate_generators_weighted: bool = False,
    scale_link_capital_costs: bool = True,
    # temporal
    offset: str | None = None,
    stride: int | None = None,
    num_segments: int | None = None,
    solver: str = "highs",
    snapshot_map: dict | None = None,
) -> dict:
    """Cluster a network spatially or temporally, storing result as a new model.

    Args:
        model_id: Source model ID.
        domain: 'spatial' or 'temporal'.
        method: Clustering method. Spatial: 'kmeans', 'hac', 'greedy_modularity'.
                Temporal: 'resample', 'downsample', 'segment', 'snapshot_map'.
        output_model_id: ID to store the clustered network under.
        n_clusters: Number of clusters (spatial methods).
        bus_weightings: Optional bus weighting dict (spatial).
        line_length_factor: Line length factor (spatial kmeans).
        affinity: Affinity metric (spatial hac).
        linkage: Linkage method (spatial hac).
        with_time: Include time in spatial clustering.
        aggregate_generators_weighted: Weight generator aggregation.
        scale_link_capital_costs: Scale link capital costs.
        offset: Resample offset string, e.g. '3h' (temporal resample).
        stride: Downsample stride (temporal downsample).
        num_segments: Number of segments (temporal segment).
        solver: Solver for segmentation (temporal segment).
        snapshot_map: Snapshot mapping dict (temporal snapshot_map).

    Returns:
        Dict with clustering summary including input/output bus and snapshot counts.
    """
    # Validate model
    try:
        n = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    # Validate domain
    if domain not in ("spatial", "temporal"):
        return {"error": f"Invalid domain '{domain}'. Must be 'spatial' or 'temporal'."}

    # Validate method
    if domain == "spatial" and method not in VALID_SPATIAL_METHODS:
        return {
            "error": f"Invalid spatial method '{method}'. Valid: {sorted(VALID_SPATIAL_METHODS)}"
        }
    if domain == "temporal" and method not in VALID_TEMPORAL_METHODS:
        return {
            "error": f"Invalid temporal method '{method}'. Valid: {sorted(VALID_TEMPORAL_METHODS)}"
        }

    input_buses = len(n.buses)
    input_snapshots = len(n.snapshots)

    try:
        if domain == "spatial":
            clustered = _cluster_spatial(n, method, n_clusters, line_length_factor, affinity, linkage)
        else:
            clustered = _cluster_temporal(n, method, offset, stride, num_segments, solver, snapshot_map)
    except Exception as e:
        return {"error": f"Clustering failed: {e}"}

    MODELS[output_model_id] = clustered

    output_buses = len(clustered.buses)
    output_snapshots = len(clustered.snapshots)

    return {
        "domain": domain,
        "method": method,
        "input_model_id": model_id,
        "output_model_id": output_model_id,
        "input_buses": input_buses,
        "output_buses": output_buses,
        "input_snapshots": input_snapshots,
        "output_snapshots": output_snapshots,
        "message": (
            f"Clustered '{model_id}' ({domain}/{method}) -> '{output_model_id}'. "
            f"Buses: {input_buses}->{output_buses}, Snapshots: {input_snapshots}->{output_snapshots}."
        ),
    }


def _cluster_spatial(n, method, n_clusters, line_length_factor, affinity, linkage):
    """Run a spatial clustering method."""
    if n_clusters is None:
        raise ValueError("n_clusters is required for spatial clustering.")

    spatial = n.cluster.spatial
    if method == "kmeans":
        return spatial.cluster_by_kmeans(n_clusters=n_clusters, line_length_factor=line_length_factor)
    elif method == "hac":
        return spatial.cluster_by_hac(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    elif method == "greedy_modularity":
        return spatial.cluster_by_greedy_modularity(n_clusters=n_clusters)


def _cluster_temporal(n, method, offset, stride, num_segments, solver, snapshot_map):
    """Run a temporal clustering method."""
    temporal = n.cluster.temporal
    if method == "resample":
        if offset is None:
            raise ValueError("offset is required for resample.")
        return temporal.resample(offset=offset)
    elif method == "downsample":
        if stride is None:
            raise ValueError("stride is required for downsample.")
        return temporal.downsample(stride=stride)
    elif method == "segment":
        if num_segments is None:
            raise ValueError("num_segments is required for segment.")
        return temporal.segment(num_segments=num_segments, solver=solver)
    elif method == "snapshot_map":
        if snapshot_map is None:
            raise ValueError("snapshot_map is required for snapshot_map.")
        return temporal.from_snapshot_map(snapshot_map=snapshot_map)
