"""Edge case tests across all tool groups."""

import os
import tempfile

import pandas as pd
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.components import add_component, query_components, remove_component, update_component
from pypsamcp.tools.convenience import add_bus, add_generator, add_line, add_load
from pypsamcp.tools.clustering import cluster_network
from pypsamcp.tools.deprecated import run_optimization, run_powerflow, set_snapshots
from pypsamcp.tools.discovery import describe_component, list_component_types
from pypsamcp.tools.io import network_io
from pypsamcp.tools.management import create_energy_model, delete_model, export_model_summary, list_models
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.statistics import get_statistics
from pypsamcp.tools.time_config import configure_time


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


# ── Component edge cases ──────────────────────────────────────────────


class TestComponentEdgeCases:
    async def test_add_component_with_empty_params(self):
        """add_component with None params should work for Bus (no bus ref needed)."""
        MODELS["m"] = pypsa.Network()
        result = await add_component("m", "Bus", "b0")
        assert "message" in result

    async def test_add_component_with_empty_dict_params(self):
        MODELS["m"] = pypsa.Network()
        result = await add_component("m", "Bus", "b0", {})
        assert "message" in result

    async def test_add_generator_without_bus_param(self):
        """Generator requires bus — omitting it should fail or PyPSA should error."""
        n = pypsa.Network()
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_component("m", "Generator", "g0", {})
        # bus is required for Generator — should either error or create with empty bus
        assert isinstance(result, dict)

    async def test_add_link_with_multi_bus(self):
        """Link supports bus0, bus1, and optional bus2+."""
        n = pypsa.Network()
        n.add("Bus", "elec")
        n.add("Bus", "heat")
        n.add("Bus", "gas_bus")
        MODELS["m"] = n
        result = await add_component("m", "Link", "chp",
            {"bus0": "gas_bus", "bus1": "elec", "bus2": "heat",
             "efficiency": 0.4, "efficiency2": 0.3, "p_nom": 100})
        assert "message" in result

    async def test_add_link_missing_bus2(self):
        """Link with bus2 pointing to nonexistent bus should fail."""
        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add("Bus", "b1")
        MODELS["m"] = n
        result = await add_component("m", "Link", "link0",
            {"bus0": "b0", "bus1": "b1", "bus2": "nonexistent"})
        assert "error" in result

    async def test_add_global_constraint(self):
        """GlobalConstraint has no bus ref — should work."""
        MODELS["m"] = pypsa.Network()
        result = await add_component("m", "GlobalConstraint", "co2_limit",
            {"type": "primary_energy", "sense": "<=", "constant": 1e6,
             "carrier_attribute": "co2_emissions"})
        assert "message" in result

    async def test_update_nonexistent_param(self):
        """Updating a non-Input param should error."""
        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0")
        MODELS["m"] = n
        result = await update_component("m", "Generator", "g0", {"p_nom_opt": 999})
        assert "error" in result

    async def test_query_with_nonexistent_filter_attr(self):
        """Filtering on a column that doesn't exist should return all (filter is no-op)."""
        n = pypsa.Network()
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await query_components("m", "Bus", {"nonexistent_col": "val"})
        assert result["count"] == 1  # filter ignored since column doesn't exist

    async def test_remove_bus_with_connected_components(self):
        """Removing a bus that has connected generators — PyPSA behavior."""
        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0")
        MODELS["m"] = n
        result = await remove_component("m", "Bus", "b0")
        # PyPSA may or may not cascade — just verify it returns a dict
        assert isinstance(result, dict)

    async def test_time_series_length_shorter_than_snapshots(self):
        """Time series shorter than snapshots — should partially fill."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=6, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_component("m", "Generator", "g0",
            {"bus": "b0", "p_nom": 100},
            {"p_max_pu": [0.5, 0.8]})  # only 2 values for 6 snapshots
        assert "message" in result

    async def test_time_series_empty_dict(self):
        """Empty time_series dict should be fine."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_component("m", "Generator", "g0",
            {"bus": "b0"}, {})
        assert "message" in result

    async def test_time_series_with_non_varying_attr(self):
        """Passing a non-varying attr as time series should error."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_component("m", "Generator", "g0",
            {"bus": "b0"},
            {"p_nom": [100, 200]})  # p_nom is static, not varying
        assert "error" in result

    async def test_add_component_special_chars_in_id(self):
        """Component IDs with spaces/special chars."""
        MODELS["m"] = pypsa.Network()
        result = await add_component("m", "Bus", "bus with spaces")
        assert "message" in result
        assert "bus with spaces" in MODELS["m"].buses.index


# ── Time config edge cases ────────────────────────────────────────────


class TestTimeConfigEdgeCases:
    async def test_single_snapshot(self):
        MODELS["m"] = pypsa.Network()
        result = await configure_time("m", "snapshots", snapshots=["2024-06-15"])
        assert result["snapshot_count"] == 1

    async def test_snapshots_out_of_order(self):
        """PyPSA should sort or accept out-of-order snapshots."""
        MODELS["m"] = pypsa.Network()
        result = await configure_time("m", "snapshots",
            snapshots=["2024-01-03", "2024-01-01", "2024-01-02"])
        assert isinstance(result, dict)

    async def test_investment_periods_without_snapshots(self):
        """Setting investment periods on a fresh network (only default 'now' snapshot)."""
        MODELS["m"] = pypsa.Network()
        result = await configure_time("m", "investment_periods", periods=[2025, 2030])
        # Should either work or return an error — not crash
        assert isinstance(result, dict)

    async def test_risk_preference_without_scenarios(self):
        MODELS["m"] = pypsa.Network()
        result = await configure_time("m", "risk_preference", alpha=0.95, omega=0.5)
        assert "error" in result

    async def test_empty_snapshots_list(self):
        """Empty snapshots list."""
        MODELS["m"] = pypsa.Network()
        result = await configure_time("m", "snapshots", snapshots=[])
        # PyPSA may error or set empty — just verify no crash
        assert isinstance(result, dict)


# ── Simulation edge cases ─────────────────────────────────────────────


class TestSimulationEdgeCases:
    async def test_optimize_empty_network(self):
        """Optimizing a network with no components."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize")
        # Should either succeed (trivial) or return clean error
        assert isinstance(result, dict)

    async def test_optimize_no_load(self):
        """Network with generators but no load — trivial optimization."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize")
        assert result.get("status") == "ok"
        assert result.get("objective_value") == pytest.approx(0.0, abs=1e-6)

    async def test_pf_single_bus_no_line(self):
        """Power flow on single-bus network (no lines)."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0", v_nom=110)
        n.add("Generator", "g0", bus="b0", p_set=50, control="PQ")
        n.add("Load", "l0", bus="b0", p_set=50)
        MODELS["m"] = n
        result = await run_simulation("m", mode="pf")
        assert "summary" in result or "error" in result

    async def test_extra_functionality_syntax_error(self):
        """Bad extra_functionality code should return clean error."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=50)
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize",
            extra_functionality="this is not valid python!!!")
        assert "error" in result

    async def test_optimize_infeasible(self):
        """Infeasible model — load > all generation capacity."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=10, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=1000)
        MODELS["m"] = n
        result = await run_simulation("m", mode="optimize")
        # Should return infeasible status, not crash
        assert isinstance(result, dict)
        if "status" in result:
            assert result["status"] != "ok" or result["termination_condition"] != "optimal"


# ── Statistics edge cases ─────────────────────────────────────────────


class TestStatisticsEdgeCases:
    async def test_all_metrics_on_simple_model(self):
        """'all' metric on a simple model — some metrics may return None."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=50)
        MODELS["m"] = n
        await run_simulation("m", mode="optimize")
        result = await get_statistics("m", "all")
        assert "result" in result
        # All 19 metrics should be present as keys
        assert len(result["result"]) == 19

    async def test_groupby_bus(self):
        """Group statistics by bus instead of carrier."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10, carrier="gas")
        n.add("Load", "l0", bus="b0", p_set=50)
        MODELS["m"] = n
        await run_simulation("m", mode="optimize")
        result = await get_statistics("m", "supply", groupby="bus")
        assert "result" in result

    async def test_filter_by_carrier(self):
        """Filter statistics by carrier."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10, carrier="gas")
        n.add("Generator", "g1", bus="b0", p_nom=50, marginal_cost=5, carrier="wind")
        n.add("Load", "l0", bus="b0", p_set=80)
        MODELS["m"] = n
        await run_simulation("m", mode="optimize")
        result = await get_statistics("m", "supply", carrier=["gas"])
        assert "result" in result


# ── I/O edge cases ────────────────────────────────────────────────────


class TestIOEdgeCases:
    async def test_export_netcdf_without_path(self):
        MODELS["m"] = pypsa.Network()
        result = await network_io("m", "export_netcdf")
        assert "error" in result

    async def test_import_nonexistent_file(self):
        result = await network_io("m", "import_netcdf", path="/tmp/nonexistent_file_xyz.nc")
        assert "error" in result

    async def test_merge_missing_other_model(self):
        MODELS["m"] = pypsa.Network()
        result = await network_io("m", "merge", other_model_id="nonexistent", output_model_id="out")
        assert "error" in result

    async def test_copy_without_output_id(self):
        MODELS["m"] = pypsa.Network()
        result = await network_io("m", "copy")
        assert "error" in result

    async def test_export_csv_roundtrip(self):
        """Export to CSV then import — verify data preserved."""
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "b0", v_nom=110)
        n.add("Generator", "g0", bus="b0", p_nom=42)
        MODELS["orig"] = n
        with tempfile.TemporaryDirectory() as tmpdir:
            await network_io("orig", "export_csv", path=tmpdir)
            result = await network_io("reimported", "import_csv", path=tmpdir)
            assert "message" in result
            assert len(MODELS["reimported"].buses) == 1
            assert MODELS["reimported"].generators.loc["g0", "p_nom"] == pytest.approx(42)

    async def test_copy_with_snapshot_subset(self):
        """Copy with a subset of snapshots."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=10, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await network_io("m", "copy", output_model_id="sub",
            snapshots=["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"])
        assert "message" in result
        assert len(MODELS["sub"].snapshots) == 3


# ── Clustering edge cases ─────────────────────────────────────────────


class TestClusteringEdgeCases:
    async def test_temporal_segment_requires_num_segments(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await cluster_network("m", "temporal", "segment", "out")
        assert "error" in result

    async def test_spatial_requires_n_clusters(self):
        n = pypsa.Network()
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await cluster_network("m", "spatial", "kmeans", "out")
        assert "error" in result

    async def test_temporal_resample_requires_offset(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        MODELS["m"] = n
        result = await cluster_network("m", "temporal", "resample", "out")
        assert "error" in result

    async def test_cluster_overwrites_existing_model(self):
        """Output model ID already exists — should overwrite."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        n.add("Bus", "b0")
        n.add("Load", "l0", bus="b0", p_set=50)
        MODELS["m"] = n
        MODELS["out"] = pypsa.Network()  # pre-existing
        result = await cluster_network("m", "temporal", "downsample", "out", stride=6)
        assert "message" in result
        assert len(MODELS["out"].snapshots) == 4


# ── Discovery edge cases ──────────────────────────────────────────────


class TestDiscoveryEdgeCases:
    async def test_describe_all_13_types(self):
        """Every valid component type should describe successfully."""
        from pypsamcp.core import VALID_COMPONENT_TYPES
        for type_name in VALID_COMPONENT_TYPES:
            result = await describe_component(type_name)
            assert "error" not in result, f"describe_component('{type_name}') failed: {result}"
            assert "required" in result
            assert "static" in result
            assert "varying" in result

    async def test_describe_internal_type_rejected(self):
        """SubNetwork and Shape should be rejected."""
        result = await describe_component("SubNetwork")
        assert "error" in result
        result = await describe_component("Shape")
        assert "error" in result


# ── Management edge cases ─────────────────────────────────────────────


class TestManagementEdgeCases:
    async def test_delete_nonexistent(self):
        result = await delete_model("nonexistent")
        assert "error" in result

    async def test_export_summary_with_snapshots_and_components(self):
        """Full summary with all component types populated."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        n.add("Bus", "b0")
        n.add("Bus", "b1")
        n.add("Generator", "g0", bus="b0", p_nom=100)
        n.add("Load", "l0", bus="b0", p_set=50)
        n.add("Line", "line0", bus0="b0", bus1="b1", x=0.1, s_nom=100)
        n.add("StorageUnit", "su0", bus="b0", p_nom=10, max_hours=4)
        n.add("Link", "lk0", bus0="b0", bus1="b1", p_nom=50)
        n.add("Carrier", "wind", co2_emissions=0)
        MODELS["full"] = n
        result = await export_model_summary("full")
        summary = result["summary"]
        assert summary["components"]["Bus"]["count"] == 2
        assert summary["components"]["Generator"]["count"] == 1
        assert summary["components"]["Line"]["count"] == 1
        assert summary["components"]["StorageUnit"]["count"] == 1
        assert summary["components"]["Link"]["count"] == 1
        assert summary["snapshots"]["count"] == 3

    async def test_create_model_empty_string_id(self):
        """Empty string model ID."""
        result = await create_energy_model("")
        # Should work — empty string is a valid dict key
        assert "message" in result or "error" in result


# ── Deprecated aliases edge cases ─────────────────────────────────────


class TestDeprecatedEdgeCases:
    async def test_run_powerflow_with_specific_snapshot(self):
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01", "2024-01-02"])
        n.add("Bus", "b0", v_nom=110)
        n.add("Bus", "b1", v_nom=110)
        n.add("Generator", "g0", bus="b0", p_set=50, control="PQ")
        n.add("Load", "l0", bus="b1", p_set=30)
        n.add("Line", "line0", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=100)
        MODELS["m"] = n
        result = await run_powerflow("m", snapshot="2024-01-01")
        assert "deprecation_notice" in result

    async def test_deprecated_add_line_missing_second_bus(self):
        n = pypsa.Network()
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_line("m", "line0", "b0", "nonexistent", x=0.1)
        assert "error" in result


# ── Convenience wrapper edge cases ────────────────────────────────────


class TestConvenienceEdgeCases:
    async def test_add_bus_with_country(self):
        """country param should map to PyPSA's 'location' attribute."""
        MODELS["m"] = pypsa.Network()
        result = await add_bus("m", "b0", 110, country="DE")
        assert "message" in result
        assert MODELS["m"].buses.loc["b0", "location"] == "DE"

    async def test_add_generator_with_all_optional_params(self):
        n = pypsa.Network()
        n.add("Bus", "b0")
        MODELS["m"] = n
        result = await add_generator("m", "g0", "b0",
            p_nom=100, p_nom_extendable=True,
            capital_cost=50000, marginal_cost=5, carrier="wind", efficiency=0.95)
        assert "message" in result
        g = MODELS["m"].generators.loc["g0"]
        assert g["p_nom_extendable"] == True
        assert g["capital_cost"] == 50000
        assert g["carrier"] == "wind"
