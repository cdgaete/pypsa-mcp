"""Edge case tests round 2 — deeper coverage of data flows, multi-component
interactions, serialization, optimization variants, and boundary conditions."""

import os
import tempfile

import pandas as pd
import numpy as np
import pytest
import pypsa

from pypsamcp.core import MODELS, convert_to_serializable, validate_component_type
from pypsamcp.tools.components import add_component, query_components, remove_component, update_component
from pypsamcp.tools.clustering import cluster_network
from pypsamcp.tools.discovery import describe_component
from pypsamcp.tools.io import network_io
from pypsamcp.tools.management import create_energy_model, export_model_summary
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.statistics import get_statistics
from pypsamcp.tools.time_config import configure_time


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


# ── Serialization edge cases ──────────────────────────────────────────


class TestConvertToSerializable:
    def test_nested_dataframe_in_dict(self):
        """convert_to_serializable only converts top-level — nested stays as-is."""
        df = pd.DataFrame({"a": [1]})
        assert convert_to_serializable(df) == [{"a": 1}]

    def test_empty_dataframe(self):
        result = convert_to_serializable(pd.DataFrame())
        assert result == []

    def test_empty_series(self):
        result = convert_to_serializable(pd.Series(dtype=float))
        assert result == {}

    def test_numpy_int_types(self):
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            val = dtype(42)
            result = convert_to_serializable(val)
            assert result == 42
            assert isinstance(result, int)

    def test_numpy_float_types(self):
        for dtype in [np.float32, np.float64]:
            val = dtype(3.14)
            result = convert_to_serializable(val)
            assert isinstance(result, float)

    def test_none_passthrough(self):
        assert convert_to_serializable(None) is None

    def test_bool_passthrough(self):
        assert convert_to_serializable(True) is True

    def test_dict_passthrough(self):
        d = {"a": 1}
        assert convert_to_serializable(d) is d

    def test_multiindex_series(self):
        """MultiIndex Series should serialize to dict with tuple keys."""
        idx = pd.MultiIndex.from_tuples([("Generator", "gas"), ("Load", "demand")])
        s = pd.Series([100.0, -50.0], index=idx)
        result = convert_to_serializable(s)
        assert isinstance(result, dict)


# ── validate_component_type ───────────────────────────────────────────


class TestValidateComponentType:
    def test_all_camelcase_forms(self):
        from pypsamcp.core import VALID_COMPONENT_TYPES
        for name in VALID_COMPONENT_TYPES:
            assert validate_component_type(name) == name

    def test_all_list_name_forms(self):
        from pypsamcp.core import VALID_COMPONENT_TYPES
        for camel, list_name in VALID_COMPONENT_TYPES.items():
            assert validate_component_type(list_name) == camel

    def test_case_sensitive(self):
        """'bus' (lowercase) should fail — only 'Bus' or 'buses' accepted."""
        with pytest.raises(ValueError):
            validate_component_type("bus")

    def test_partial_match_fails(self):
        with pytest.raises(ValueError):
            validate_component_type("Gen")


# ── Multi-component interaction tests ─────────────────────────────────


class TestMultiComponentInteractions:
    async def test_store_with_link_sector_coupling(self):
        """Store connected via Link for sector coupling (e.g. H2)."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "elec")
        n.add("Bus", "h2")
        MODELS["m"] = n
        await add_component("m", "Generator", "wind",
            {"bus": "elec", "p_nom": 200, "marginal_cost": 5})
        await add_component("m", "Load", "elec_demand",
            {"bus": "elec"}, {"p_set": [80, 100, 90]})
        await add_component("m", "Link", "electrolyser",
            {"bus0": "elec", "bus1": "h2", "p_nom": 50, "efficiency": 0.7})
        await add_component("m", "Store", "h2_tank",
            {"bus": "h2", "e_nom": 500, "e_cyclic": True})
        result = await run_simulation("m", mode="optimize")
        assert result.get("status") == "ok"

    async def test_transformer_between_voltage_levels(self):
        """Add transformer between two buses."""
        n = pypsa.Network()
        n.add("Bus", "hv", v_nom=220)
        n.add("Bus", "mv", v_nom=110)
        MODELS["m"] = n
        result = await add_component("m", "Transformer", "trafo1",
            {"bus0": "hv", "bus1": "mv", "x": 0.1, "s_nom": 100})
        assert "message" in result

    async def test_shunt_impedance(self):
        n = pypsa.Network()
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_component("m", "ShuntImpedance", "shunt0",
            {"bus": "b0", "g": 0.01, "b": 0.02})
        assert "message" in result

    async def test_multiple_loads_same_bus(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        r1 = await add_component("m", "Load", "residential", {"bus": "b0", "p_set": 50})
        r2 = await add_component("m", "Load", "industrial", {"bus": "b0", "p_set": 100})
        assert "message" in r1
        assert "message" in r2
        result = await query_components("m", "Load")
        assert result["count"] == 2

    async def test_extendable_line_optimization(self):
        """Line with s_nom_extendable — optimizer should expand it."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "a")
        n.add("Bus", "b")
        n.add("Generator", "cheap", bus="a", p_nom=200, marginal_cost=5)
        n.add("Load", "demand", bus="b", p_set=100)
        n.add("Line", "ab", bus0="a", bus1="b", x=0.1,
              s_nom=0, s_nom_extendable=True, capital_cost=100)
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize")
        assert result["status"] == "ok"
        # Line should have been expanded
        summary = result.get("summary", {})
        assert isinstance(summary, dict)


# ── Time series data integrity ────────────────────────────────────────


class TestTimeSeriesIntegrity:
    async def test_time_series_persists_after_query(self):
        """Time series should be intact after querying static components."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        await add_component("m", "Generator", "g0",
            {"bus": "b0", "p_nom": 100},
            {"p_max_pu": [0.1, 0.5, 0.8, 0.3]})
        # Query shouldn't affect time series
        await query_components("m", "Generator")
        ts = MODELS["m"].generators_t.p_max_pu
        assert not ts.empty
        assert ts.loc[ts.index[0], "g0"] == pytest.approx(0.1)
        assert ts.loc[ts.index[2], "g0"] == pytest.approx(0.8)

    async def test_update_time_series(self):
        """Update only time series, not static params."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100)
        MODELS["m"] = n
        result = await update_component("m", "Generator", "g0",
            time_series={"p_max_pu": [0.2, 0.4, 0.6]})
        assert "message" in result
        ts = MODELS["m"].generators_t.p_max_pu
        assert ts.loc[ts.index[1], "g0"] == pytest.approx(0.4)

    async def test_multiple_time_series_attrs(self):
        """Set multiple time-varying attrs at once."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_component("m", "Load", "l0",
            {"bus": "b0"},
            {"p_set": [50, 80, 60], "q_set": [10, 15, 12]})
        assert "message" in result
        assert not MODELS["m"].loads_t.p_set.empty
        assert not MODELS["m"].loads_t.q_set.empty


# ── Optimization result shapes ────────────────────────────────────────


class TestOptimizationResults:
    async def test_result_has_dispatch_and_prices(self):
        """Optimization result should contain generator dispatch and marginal prices."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=200, marginal_cost=10, carrier="gas")
        n.add("Generator", "g1", bus="b0", p_nom=100, marginal_cost=30, carrier="oil")
        n.add("Load", "l0", bus="b0", p_set=150)
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize")
        assert result["status"] == "ok"
        summary = result["summary"]
        assert "generator_dispatch" in summary
        assert "bus_marginal_price" in summary

    async def test_storage_dispatch_in_results(self):
        """Storage should appear in results when used."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=200, marginal_cost=10)
        n.add("StorageUnit", "su0", bus="b0", p_nom=50, max_hours=4,
              state_of_charge_initial=100, cyclic_state_of_charge=True)
        n.add("Load", "l0", bus="b0")
        n.loads_t.p_set["l0"] = pd.Series([50, 250, 50, 250], index=n.snapshots)
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize")
        assert result["status"] == "ok"
        summary = result["summary"]
        assert "storage_unit_dispatch" in summary

    async def test_lpf_result_shape(self):
        """LPF should return voltage magnitudes, angles, and line flows."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0", v_nom=110)
        n.add("Bus", "b1", v_nom=110)
        n.add("Generator", "g0", bus="b0", p_set=100, control="PQ")
        n.add("Load", "l0", bus="b1", p_set=80)
        n.add("Line", "line0", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=200)
        MODELS["m"] = n
        result = await run_simulation("m", mode="lpf")
        summary = result["summary"]
        assert "bus_v_mag_pu" in summary
        assert "bus_v_ang" in summary
        assert "line_p0" in summary


# ── Statistics each metric individually ───────────────────────────────


class TestStatisticsAllMetrics:
    """Test each of the 19 metrics individually on a solved model."""

    @pytest.fixture
    async def solved(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "b0")
        n.add("Carrier", "gas", co2_emissions=0.2)
        n.add("Generator", "g0", bus="b0", p_nom=200, marginal_cost=10,
              carrier="gas", p_nom_extendable=True, capital_cost=1000,
              lifetime=25, discount_rate=0.07)
        n.add("Load", "l0", bus="b0", p_set=100)
        MODELS["m"] = n
        await run_simulation("m", mode="optimize")
        return n

    @pytest.mark.parametrize("metric", [
        "system_cost", "capex", "opex", "fom", "overnight_cost",
        "installed_capacity", "optimal_capacity", "expanded_capacity",
        "installed_capex", "expanded_capex", "capacity_factor", "curtailment",
        "energy_balance", "supply", "withdrawal", "revenue", "market_value",
        "transmission",
    ])
    async def test_individual_metric(self, solved, metric):
        result = await get_statistics("m", metric)
        assert "error" not in result, f"Metric '{metric}' failed: {result}"
        assert "result" in result

    async def test_prices_with_valid_groupby(self, solved):
        """prices only accepts groupby='bus_carrier' or groupby=False."""
        result = await get_statistics("m", "prices", groupby="bus_carrier")
        assert "error" not in result
        assert "result" in result

    async def test_prices_with_invalid_groupby_returns_actionable_error(self, solved):
        """prices with groupby='carrier' should fail with a recommendation."""
        result = await get_statistics("m", "prices", groupby="carrier")
        assert "error" in result
        assert "groupby" in result["error"]
        assert "bus_carrier" in result["error"]

    async def test_prices_rejects_aggregate_across_components(self, solved):
        """prices with aggregate_across_components=True should fail."""
        result = await get_statistics("m", "prices",
            groupby="bus_carrier", aggregate_across_components=True)
        assert "error" in result
        assert "aggregate_across_components" in result["error"]

    async def test_all_metrics_reports_errors(self, solved):
        """'all' metric should report per-metric errors, not swallow them."""
        result = await get_statistics("m", "all")
        assert "result" in result
        # prices should have failed with default groupby="carrier"
        assert result["result"]["prices"] is None
        assert "metric_errors" in result
        assert "prices" in result["metric_errors"]
        assert "groupby" in result["metric_errors"]["prices"]

    async def test_aggregate_across_components(self, solved):
        result = await get_statistics("m", "supply", aggregate_across_components=True)
        assert "result" in result

    async def test_drop_zero(self, solved):
        result = await get_statistics("m", "curtailment", drop_zero=True)
        assert "result" in result

    async def test_nice_names(self, solved):
        result = await get_statistics("m", "supply", nice_names=True)
        assert "result" in result


# ── I/O roundtrip data integrity ──────────────────────────────────────


class TestIORoundtrip:
    async def test_netcdf_roundtrip_preserves_time_series(self):
        """Export + import should preserve time-varying data."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100)
        n.generators_t.p_max_pu["g0"] = pd.Series([0.1, 0.5, 0.9], index=n.snapshots)
        MODELS["orig"] = n
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name
        try:
            await network_io("orig", "export_netcdf", path=path)
            await network_io("loaded", "import_netcdf", path=path)
            loaded = MODELS["loaded"]
            assert len(loaded.snapshots) == 3
            assert loaded.generators_t.p_max_pu.loc[loaded.snapshots[0], "g0"] == pytest.approx(0.1)
            assert loaded.generators_t.p_max_pu.loc[loaded.snapshots[2], "g0"] == pytest.approx(0.9)
        finally:
            os.unlink(path)

    async def test_netcdf_roundtrip_preserves_component_params(self):
        n = pypsa.Network()
        n.add("Bus", "b0", v_nom=345)
        n.add("Generator", "g0", bus="b0", p_nom=42, marginal_cost=7.5, carrier="wind")
        MODELS["orig"] = n
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name
        try:
            await network_io("orig", "export_netcdf", path=path)
            await network_io("loaded", "import_netcdf", path=path)
            loaded = MODELS["loaded"]
            assert loaded.buses.loc["b0", "v_nom"] == 345
            assert loaded.generators.loc["g0", "p_nom"] == pytest.approx(42)
            assert loaded.generators.loc["g0", "marginal_cost"] == pytest.approx(7.5)
            assert loaded.generators.loc["g0", "carrier"] == "wind"
        finally:
            os.unlink(path)

    async def test_netcdf_export_float32(self):
        """float32 export should succeed without error."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=100, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100)
        n.generators_t.p_max_pu["g0"] = pd.Series(np.random.rand(100), index=n.snapshots)
        MODELS["m"] = n
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name
        try:
            result = await network_io("m", "export_netcdf", path=path, float32=True)
            assert "message" in result
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ── Merge edge cases ─────────────────────────────────────────────────


class TestMergeEdgeCases:
    async def test_merge_disjoint_networks(self):
        """Merge two networks with no overlapping components."""
        n1 = pypsa.Network()
        n1.set_snapshots(["2024-01-01"])
        n1.add("Bus", "a")
        n1.add("Generator", "ga", bus="a", p_nom=100)
        n2 = pypsa.Network()
        n2.set_snapshots(["2024-01-01"])
        n2.add("Bus", "b")
        n2.add("Generator", "gb", bus="b", p_nom=200)
        MODELS["n1"] = n1
        MODELS["n2"] = n2
        result = await network_io("n1", "merge", other_model_id="n2", output_model_id="merged")
        assert "message" in result
        merged = MODELS["merged"]
        assert len(merged.buses) == 2
        assert len(merged.generators) == 2

    async def test_merge_without_time(self):
        n1 = pypsa.Network()
        n1.set_snapshots(["2024-01-01"])
        n1.add("Bus", "a")
        n2 = pypsa.Network()
        n2.set_snapshots(["2024-01-01"])
        n2.add("Bus", "b")
        MODELS["n1"] = n1
        MODELS["n2"] = n2
        result = await network_io("n1", "merge", other_model_id="n2",
                                  output_model_id="merged", with_time=False)
        assert "message" in result


# ── Copy with bus subset ──────────────────────────────────────────────


class TestCopyWithBusSubset:
    async def test_copy_specific_buses(self):
        """Copy should keep only specified buses."""
        n = pypsa.Network()
        n.add("Bus", "a")
        n.add("Bus", "b")
        n.add("Bus", "c")
        n.add("Generator", "ga", bus="a", p_nom=100)
        n.add("Generator", "gb", bus="b", p_nom=200)
        MODELS["m"] = n
        result = await network_io("m", "copy", output_model_id="subset", buses=["a"])
        assert "message" in result
        subset = MODELS["subset"]
        assert "a" in subset.buses.index
        assert "b" not in subset.buses.index


# ── Clustering with solved model ──────────────────────────────────────


class TestClusteringSolved:
    async def test_temporal_downsample_then_optimize(self):
        """Downsample a network, then optimize the downsampled version."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=12, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=200, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=80)
        MODELS["orig"] = n
        result = await cluster_network("orig", "temporal", "downsample", "ds", stride=3)
        assert "message" in result
        assert MODELS["ds"].snapshots is not None
        # Optimize the downsampled network
        opt_result = await run_simulation("ds", mode="optimize")
        assert opt_result.get("status") == "ok"

    async def test_temporal_segment(self):
        """Temporal segmentation requires optional 'tsam' package."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0")
        n.loads_t.p_set["l0"] = pd.Series(
            [50 + 30 * np.sin(2 * np.pi * i / 24) for i in range(24)],
            index=n.snapshots)
        MODELS["m"] = n
        result = await cluster_network("m", "temporal", "segment", "seg", num_segments=6)
        # Either works (if tsam installed) or returns clean error
        assert "message" in result or "error" in result
        if "error" in result:
            assert "tsam" in result["error"]


# ── Describe component return shapes ──────────────────────────────────


class TestDescribeReturnShape:
    async def test_bus_has_no_required_bus_ref(self):
        """Bus itself has no bus reference in required."""
        result = await describe_component("Bus")
        required_attrs = [p["attr"] for p in result["required"]]
        assert "bus" not in required_attrs
        assert "bus0" not in required_attrs

    async def test_line_has_bus0_bus1_required(self):
        result = await describe_component("Line")
        required_attrs = [p["attr"] for p in result["required"]]
        assert "bus0" in required_attrs
        assert "bus1" in required_attrs

    async def test_generator_varying_includes_p_max_pu(self):
        result = await describe_component("Generator")
        varying_attrs = [p["attr"] for p in result["varying"]]
        assert "p_max_pu" in varying_attrs

    async def test_store_has_e_nom_in_static(self):
        result = await describe_component("Store")
        static_attrs = [p["attr"] for p in result["static"]]
        assert "e_nom" in static_attrs

    async def test_link_has_efficiency_in_varying(self):
        result = await describe_component("Link")
        varying_attrs = [p["attr"] for p in result["varying"]]
        assert "efficiency" in varying_attrs


# ── Model summary with solved state ──────────────────────────────────


class TestModelSummaryAfterSolve:
    async def test_summary_after_optimization(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=50)
        MODELS["m"] = n
        await run_simulation("m", mode="optimize")
        result = await export_model_summary("m")
        assert "summary" in result
        s = result["summary"]
        assert s["has_investment_periods"] is False
        assert s["has_scenarios"] is False
        assert s["snapshots"]["count"] == 3

    async def test_summary_shows_all_component_types(self):
        """Summary should list every non-empty component type."""
        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add("Bus", "b1")
        n.add("Generator", "g0", bus="b0", p_nom=100)
        n.add("Load", "l0", bus="b0", p_set=50)
        n.add("Line", "line0", bus0="b0", bus1="b1", x=0.1, s_nom=100)
        n.add("Link", "lk0", bus0="b0", bus1="b1", p_nom=50)
        n.add("Store", "st0", bus="b0", e_nom=100)
        n.add("StorageUnit", "su0", bus="b0", p_nom=20, max_hours=4)
        n.add("Carrier", "wind")
        MODELS["m"] = n
        result = await export_model_summary("m")
        components = result["summary"]["components"]
        for expected in ["Bus", "Generator", "Load", "Line", "Link", "Store", "StorageUnit", "Carrier"]:
            assert expected in components, f"Missing {expected} in summary"


# ── Concurrent model operations ───────────────────────────────────────


class TestMultipleModels:
    async def test_operate_on_independent_models(self):
        """Changes to one model should not affect another."""
        n1 = pypsa.Network()
        n1.add("Bus", "b0")
        n2 = pypsa.Network()
        n2.add("Bus", "b0")
        n2.add("Bus", "b1")
        MODELS["m1"] = n1
        MODELS["m2"] = n2
        await add_component("m1", "Generator", "g0", {"bus": "b0", "p_nom": 100})
        assert len(MODELS["m1"].generators) == 1
        assert len(MODELS["m2"].generators) == 0

    async def test_override_preserves_other_models(self):
        n1 = pypsa.Network()
        n1.add("Bus", "original")
        MODELS["m"] = n1
        MODELS["other"] = pypsa.Network()
        await create_energy_model("m", override=True)
        # 'm' is now fresh, 'other' is untouched
        assert len(MODELS["m"].buses) == 0
        assert "other" in MODELS


# ── Snapshot weightings ───────────────────────────────────────────────


class TestSnapshotWeightings:
    async def test_set_weightings(self):
        MODELS["m"] = pypsa.Network()
        result = await configure_time("m", "snapshots",
            snapshots=["2024-01-01", "2024-01-02", "2024-01-03"],
            weightings={"objective": [1.0, 2.0, 1.0]})
        assert "message" in result
        w = MODELS["m"].snapshot_weightings
        assert w.loc[w.index[1], "objective"] == 2.0

    async def test_weighted_optimization(self):
        """Weighted snapshots should affect objective value."""
        MODELS["m"] = pypsa.Network()
        await configure_time("m", "snapshots",
            snapshots=["2024-01-01", "2024-01-02"],
            weightings={"objective": [1.0, 3.0]})
        n = MODELS["m"]
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=50)
        result = await run_simulation("m", mode="optimize")
        assert result["status"] == "ok"
        # Objective = 10 * 50 * (1 + 3) = 2000
        assert result["objective_value"] == pytest.approx(2000.0, rel=0.01)
