import pandas as pd
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.components import (
    add_component,
    query_components,
    remove_component,
    update_component,
)


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def model_with_bus():
    n = pypsa.Network()
    n.name = "test"
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="D"))
    n.add("Bus", "bus0")
    MODELS["test"] = n
    return n


class TestAddComponent:
    async def test_add_bus(self):
        n = pypsa.Network()
        MODELS["test"] = n
        result = await add_component("test", "Bus", "bus0", {"v_nom": 110.0})
        assert "message" in result
        assert "bus0" in n.buses.index

    async def test_add_generator_with_bus_check(self, model_with_bus):
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "bus0", "p_nom": 100.0, "marginal_cost": 10.0},
        )
        assert "message" in result
        assert "gen0" in model_with_bus.generators.index

    async def test_rejects_missing_model(self):
        result = await add_component("nonexistent", "Bus", "bus0")
        assert "error" in result

    async def test_rejects_invalid_component_type(self):
        MODELS["test"] = pypsa.Network()
        result = await add_component("test", "FakeType", "x")
        assert "error" in result

    async def test_rejects_output_params(self, model_with_bus):
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "bus0", "p_nom_opt": 999},
        )
        assert "error" in result
        assert "p_nom_opt" in result["error"]

    async def test_rejects_missing_bus(self):
        n = pypsa.Network()
        MODELS["test"] = n
        n.add("Bus", "bus0")
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "nonexistent"},
        )
        assert "error" in result

    async def test_rejects_duplicate_id(self, model_with_bus):
        result = await add_component("test", "Bus", "bus0")
        assert "error" in result
        assert "already exists" in result["error"]

    async def test_time_series(self, model_with_bus):
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "bus0", "p_nom": 100.0},
            {"p_max_pu": [0.5, 0.8]},
        )
        assert "message" in result
        assert model_with_bus.generators_t.p_max_pu.loc[pd.Timestamp("2024-01-01"), "gen0"] == 0.5

    async def test_time_series_requires_snapshots(self):
        """A fresh network has snapshots=['now'] (not DatetimeIndex), should reject time series."""
        n = pypsa.Network()
        n.add("Bus", "bus0")
        MODELS["fresh"] = n
        result = await add_component(
            "fresh", "Generator", "gen0",
            {"bus": "bus0"},
            {"p_max_pu": [0.5, 0.8]},
        )
        assert "error" in result
        assert "snapshot" in result["error"].lower()

    async def test_accepts_list_name_form(self, model_with_bus):
        result = await add_component("test", "generators", "gen0", {"bus": "bus0"})
        assert "message" in result

    async def test_no_bus_check_for_carrier(self):
        MODELS["test"] = pypsa.Network()
        result = await add_component("test", "Carrier", "wind", {"co2_emissions": 0.0})
        assert "message" in result


class TestUpdateComponent:
    async def test_update_existing(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0", "p_nom": 100.0})
        result = await update_component("test", "Generator", "gen0", {"p_nom": 200.0})
        assert "message" in result
        assert model_with_bus.generators.loc["gen0", "p_nom"] == 200.0

    async def test_rejects_nonexistent(self, model_with_bus):
        result = await update_component("test", "Generator", "nonexistent", {"p_nom": 100.0})
        assert "error" in result


class TestRemoveComponent:
    async def test_remove_existing(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0"})
        result = await remove_component("test", "Generator", "gen0")
        assert "message" in result
        assert "gen0" not in model_with_bus.generators.index

    async def test_rejects_nonexistent(self, model_with_bus):
        result = await remove_component("test", "Generator", "nonexistent")
        assert "error" in result


class TestQueryComponents:
    async def test_query_all(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0", "p_nom": 100.0})
        await add_component("test", "Generator", "gen1", {"bus": "bus0", "p_nom": 200.0})
        result = await query_components("test", "Generator")
        assert result["count"] == 2

    async def test_query_with_filter(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0", "p_nom": 100.0, "carrier": "solar"})
        await add_component("test", "Generator", "gen1", {"bus": "bus0", "p_nom": 200.0, "carrier": "wind"})
        result = await query_components("test", "Generator", {"carrier": "solar"})
        assert result["count"] == 1

    async def test_query_empty(self):
        MODELS["test"] = pypsa.Network()
        result = await query_components("test", "Generator")
        assert result["count"] == 0
