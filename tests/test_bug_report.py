"""Tests for bugs from user exploration report."""

import pandas as pd
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.components import add_component
from pypsamcp.tools.management import create_energy_model
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.time_config import configure_time


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


class TestBug1InvestmentPeriodsTimeSeries:
    """Investment periods create a MultiIndex snapshot, not DatetimeIndex.
    add_component must accept both for time series."""

    async def test_investment_periods_create_multiindex(self):
        """After set_investment_periods, snapshots is a MultiIndex."""
        await create_energy_model("m")
        r = await configure_time("m", "investment_periods", periods=[2025, 2030, 2035])
        assert r["snapshot_count"] == 3
        assert isinstance(MODELS["m"].snapshots, pd.MultiIndex)

    async def test_add_component_with_ts_after_investment_periods(self):
        """Time series should work on MultiIndex snapshots from investment periods."""
        await create_energy_model("m")
        await configure_time("m", "investment_periods", periods=[2025, 2030, 2035])
        await add_component("m", "Bus", "b0")
        r = await add_component("m", "Generator", "solar",
            {"bus": "b0", "p_nom": 100},
            {"p_max_pu": [0.0, 0.9, 0.5]})
        assert "message" in r, f"Expected success, got: {r}"

    async def test_snapshots_then_investment_periods_with_ts(self):
        """Full workflow: snapshots first, then periods, then time series."""
        await create_energy_model("m")
        await configure_time("m", "snapshots",
            snapshots=["2025-01-01", "2025-07-01"])
        await configure_time("m", "investment_periods", periods=[2025, 2030, 2035])
        n = MODELS["m"]
        assert isinstance(n.snapshots, pd.MultiIndex)
        assert len(n.snapshots) == 6  # 3 periods * 2 snapshots

        await add_component("m", "Bus", "b0")
        r = await add_component("m", "Generator", "solar",
            {"bus": "b0", "p_nom": 100},
            {"p_max_pu": [0.0, 0.9, 0.5, 0.8, 0.3, 0.7]})
        assert "message" in r, f"Expected success, got: {r}"

    async def test_configure_snapshots_after_periods_no_crash(self):
        """Setting snapshots after investment periods should not crash with dtype 'M' error."""
        await create_energy_model("m")
        await configure_time("m", "investment_periods", periods=[2025, 2030, 2035])
        r = await configure_time("m", "snapshots",
            snapshots=["2025-01-01", "2025-07-01",
                       "2030-01-01", "2030-07-01",
                       "2035-01-01", "2035-07-01"])
        # Should either succeed or return a clear error, not crash with dtype 'M'
        assert isinstance(r, dict)
        if "error" in r:
            assert "dtype" not in r["error"], f"Got serialization crash: {r['error']}"


class TestBug2MGAPostProcessing:
    """MGA crashes in PyPSA post-processing when loads have pnl time series.
    This is an upstream PyPSA bug — our code should catch it gracefully."""

    async def test_mga_with_time_series_load(self):
        """MGA with time-varying load should not crash with 'setting an array element'."""
        await create_energy_model("m")
        await configure_time("m", "snapshots",
            snapshots=["2024-01-01 00:00", "2024-01-01 06:00",
                       "2024-01-01 12:00", "2024-01-01 18:00"])
        await add_component("m", "Bus", "b0")
        await add_component("m", "Generator", "solar",
            {"bus": "b0", "p_nom_extendable": True, "capital_cost": 50000, "carrier": "solar"},
            {"p_max_pu": [0.0, 0.8, 0.6, 0.1]})
        await add_component("m", "Generator", "gas",
            {"bus": "b0", "p_nom": 500, "marginal_cost": 40, "carrier": "gas"})
        await add_component("m", "Load", "demand",
            {"bus": "b0"}, {"p_set": [100, 150, 120, 80]})

        r = await run_simulation("m", mode="mga", slack=0.05)
        # Should either succeed or return a clear error about the upstream issue
        assert isinstance(r, dict)
        if "error" in r:
            assert "setting an array element" not in r["error"], (
                f"Raw numpy error leaked to user: {r['error']}")

    async def test_mga_with_static_load_works(self):
        """MGA without time-varying load should work fine."""
        await create_energy_model("m")
        await configure_time("m", "snapshots",
            snapshots=["2024-01-01 00:00", "2024-01-01 06:00",
                       "2024-01-01 12:00", "2024-01-01 18:00"])
        await add_component("m", "Bus", "b0")
        await add_component("m", "Generator", "solar",
            {"bus": "b0", "p_nom_extendable": True, "capital_cost": 50000, "carrier": "solar"})
        await add_component("m", "Generator", "gas",
            {"bus": "b0", "p_nom": 500, "marginal_cost": 40, "carrier": "gas"})
        await add_component("m", "Load", "demand", {"bus": "b0", "p_set": 100})

        r = await run_simulation("m", mode="mga", slack=0.05)
        assert r.get("status") == "ok"


class TestBug3ExtraFunctionalityNaming:
    """extra_functionality uses 'network' but PyPSA convention is 'n'.
    Both should work."""

    async def test_network_variable_works(self):
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        n = MODELS["m"]
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=50)
        r = await run_simulation("m", mode="optimize",
            extra_functionality="x = network.buses")
        assert r.get("status") == "ok"

    async def test_n_alias_works(self):
        """'n' should also be available as an alias for the network object."""
        await create_energy_model("m")
        await configure_time("m", "snapshots", snapshots=["2024-01-01"])
        n = MODELS["m"]
        n.add("Bus", "b0")
        n.add("Generator", "g0", bus="b0", p_nom=100, marginal_cost=10)
        n.add("Load", "l0", bus="b0", p_set=50)
        r = await run_simulation("m", mode="optimize",
            extra_functionality="x = n.buses")
        assert r.get("status") == "ok", f"Expected success, got: {r}"
