"""Tests for bugs from user exploration report (session 2).

Bug 1: rolling_horizon crashes (return type mismatch)
Bug 2: compute_infeasibilities returns stale data
Bug 3: temporal resample fails with int64 dtype
Bug 4: scenarios + configure_time order causes bus duplication
Bug 5: deprecated add_bus fails with MultiIndex (same root cause as Bug 4)
"""

import pandas as pd
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.clustering import cluster_network
from pypsamcp.tools.components import add_component
from pypsamcp.tools.convenience import add_bus
from pypsamcp.tools.management import create_energy_model
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.time_config import configure_time


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


class TestBug1RollingHorizon:
    """Bug 1: rolling_horizon mode crashes because optimize_with_rolling_horizon
    returns Network, not (status, condition)."""

    @pytest.fixture
    def rolling_model(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=50)
        MODELS["rolling"] = n
        return n

    async def test_rolling_horizon_basic(self, rolling_model):
        """Rolling horizon should complete without crashing."""
        result = await run_simulation(
            "rolling", mode="rolling_horizon",
            solver_name="highs", horizon=6, overlap=0,
        )
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert result["mode"] == "rolling_horizon"
        assert result["status"] == "ok"
        assert result["objective_value"] is not None

    async def test_rolling_horizon_with_overlap(self, rolling_model):
        """Rolling horizon with overlap should also work."""
        result = await run_simulation(
            "rolling", mode="rolling_horizon",
            solver_name="highs", horizon=6, overlap=2,
        )
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert result["status"] == "ok"

    async def test_rolling_horizon_with_storage(self):
        """Rolling horizon with storage units that carry state across windows."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=12, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=50)
        n.add("StorageUnit", "batt0", bus="bus0", p_nom=20,
              max_hours=4, cyclic_state_of_charge=False)
        MODELS["rolling_storage"] = n

        result = await run_simulation(
            "rolling_storage", mode="rolling_horizon",
            solver_name="highs", horizon=6, overlap=0,
        )
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert result["status"] == "ok"

    async def test_rolling_horizon_dispatch_populated(self, rolling_model):
        """Rolling horizon should produce dispatch results."""
        result = await run_simulation(
            "rolling", mode="rolling_horizon",
            solver_name="highs", horizon=12, overlap=0,
        )
        assert "error" not in result
        summary = result.get("summary", {})
        assert "generator_dispatch" in summary


class TestBug2ComputeInfeasibilities:
    """Bug 2: compute_infeasibilities returns cached prior solve results."""

    @pytest.fixture
    def infeasible_model(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=10, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=100)  # way more than gen can provide
        MODELS["infeas"] = n
        return n

    async def test_infeasible_returns_null_objective(self, infeasible_model):
        """Infeasible model should return None objective, not stale value."""
        result = await run_simulation("infeas", mode="optimize", solver_name="highs")
        assert result["termination_condition"] == "infeasible"
        assert result["objective_value"] is None
        assert result["summary"] == {}

    async def test_infeasible_no_stale_dispatch(self):
        """After a feasible solve followed by infeasible, dispatch should not be stale."""
        # First: solve a feasible model
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=50)
        MODELS["stale_test"] = n

        r1 = await run_simulation("stale_test", mode="optimize", solver_name="highs")
        assert r1["status"] == "ok"
        assert r1["objective_value"] > 0

        # Now make it infeasible
        n.generators.loc["gen0", "p_nom"] = 1  # too small
        n.loads.loc["load0", "p_set"] = 200  # too large

        r2 = await run_simulation("stale_test", mode="optimize", solver_name="highs")
        assert r2["termination_condition"] == "infeasible"
        assert r2["objective_value"] is None
        assert r2["summary"] == {}

    async def test_compute_infeasibilities_gurobi_note(self, infeasible_model):
        """compute_infeasibilities=True should include a note about Gurobi requirement."""
        result = await run_simulation(
            "infeas", mode="optimize", solver_name="highs",
            compute_infeasibilities=True,
        )
        assert result["termination_condition"] == "infeasible"
        assert "infeasibility_note" in result
        assert "Gurobi" in result["infeasibility_note"]


class TestBug3ResampleDtype:
    """Bug 3: temporal resample fails when time series have int64 dtype."""

    @pytest.fixture
    def int_ts_model(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=500, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=200)
        # Set integer-valued time series
        n.loads_t.p_set["load0"] = [
            100, 150, 200, 250, 300, 200, 150, 100,
            200, 250, 300, 250, 100, 150, 200, 250,
            300, 200, 150, 100, 200, 250, 300, 250,
        ]
        MODELS["int_ts"] = n
        return n

    async def test_resample_with_int_timeseries(self, int_ts_model):
        """Resample should succeed even when time series have int64 dtype."""
        result = await cluster_network(
            "int_ts", "temporal", "resample", "resampled", offset="4h",
        )
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert result["output_snapshots"] < result["input_snapshots"]
        assert result["output_snapshots"] == 6  # 24h / 4h = 6

    async def test_resample_preserves_data(self, int_ts_model):
        """Resampled time series should have float dtype and reasonable values."""
        await cluster_network(
            "int_ts", "temporal", "resample", "resampled", offset="4h",
        )
        resampled_n = MODELS["resampled"]
        p_set = resampled_n.loads_t.p_set["load0"]
        assert p_set.dtype.kind == "f"  # float
        assert p_set.min() >= 100
        assert p_set.max() <= 300


class TestBug4ScenariosOrder:
    """Bug 4: scenarios + configure_time causes confusing component count.

    Fixed by returning unique component count (not per-scenario duplicates)
    and properly handling MultiIndex in add_component.
    """

    async def test_scenarios_before_components_works(self):
        """Configuring scenarios before adding components should work."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        result = await configure_time("m", "scenarios", scenarios=["low", "high"])
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert result["has_scenarios"] is True

    async def test_add_component_after_scenarios_correct_count(self):
        """After scenarios, add_component should report unique count, not per-scenario."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        await configure_time("m", "scenarios", scenarios=["low", "high"])

        result = await add_component("m", "Bus", "bus0")
        assert "error" not in result, f"Got error: {result.get('error')}"
        # Should report 1 bus, not 2 (one per scenario)
        assert result["total_count"] == 1

    async def test_scenarios_after_components_works(self):
        """Configuring scenarios after adding components should also work."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        await add_component("m", "Bus", "bus0")
        await add_component("m", "Generator", "gen0", {"bus": "bus0", "p_nom": 100})

        result = await configure_time("m", "scenarios", scenarios=["low", "high"])
        assert "error" not in result, f"Got error: {result.get('error')}"

    async def test_no_duplicate_add_with_scenarios(self):
        """Cannot add same component ID twice with scenarios active."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        await configure_time("m", "scenarios", scenarios=["low", "high"])
        await add_component("m", "Bus", "bus0")

        result = await add_component("m", "Bus", "bus0")
        assert "error" in result
        assert "already exists" in result["error"]


class TestBug5DeprecatedAddBusMultiIndex:
    """Bug 5: deprecated add_bus and add_component work with scenario MultiIndex."""

    async def test_add_component_after_scenarios_works(self):
        """Adding components via add_component after scenarios should work."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        await configure_time("m", "scenarios", scenarios=["low", "high"])

        result = await add_component("m", "Bus", "bus0")
        assert "error" not in result, f"Got error: {result.get('error')}"

    async def test_deprecated_add_bus_after_scenarios_works(self):
        """Deprecated add_bus should work when scenarios are configured first."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        await configure_time("m", "scenarios", scenarios=["low", "high"])

        result = await add_bus("m", "bus0", v_nom=110.0)
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert "deprecation_notice" in result

    async def test_add_generator_with_bus_ref_after_scenarios(self):
        """Adding generator referencing a bus should work with scenarios."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        await configure_time("m", "scenarios", scenarios=["low", "high"])
        await add_component("m", "Bus", "bus0")

        result = await add_component("m", "Generator", "gen0",
                                     {"bus": "bus0", "p_nom": 100})
        assert "error" not in result, f"Got error: {result.get('error')}"
