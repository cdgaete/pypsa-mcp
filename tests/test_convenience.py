import pytest
import pypsa
import pandas as pd

from pypsamcp.core import MODELS
from pypsamcp.tools.convenience import add_bus, add_generator, add_line, add_load


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def model_with_bus():
    n = pypsa.Network()
    n.name = "test"
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Bus", "bus0", v_nom=110.0)
    n.add("Bus", "bus1", v_nom=110.0)
    MODELS["test"] = n
    return n


class TestAddBus:
    async def test_adds_bus(self):
        MODELS["test"] = pypsa.Network()
        result = await add_bus("test", "bus0", 110.0)
        assert "message" in result
        assert "deprecation_notice" in result

    async def test_with_optional_params(self):
        MODELS["test"] = pypsa.Network()
        result = await add_bus("test", "bus0", 110.0, x=1.0, y=2.0, carrier="DC")
        assert "message" in result


class TestAddGenerator:
    async def test_adds_generator(self, model_with_bus):
        result = await add_generator("test", "gen0", "bus0", p_nom=100.0)
        assert "message" in result
        assert "deprecation_notice" in result

    async def test_rejects_missing_bus(self, model_with_bus):
        result = await add_generator("test", "gen0", "nonexistent")
        assert "error" in result


class TestAddLoad:
    async def test_adds_load(self, model_with_bus):
        result = await add_load("test", "load0", "bus0", p_set=50.0)
        assert "message" in result
        assert "deprecation_notice" in result


class TestAddLine:
    async def test_adds_line(self, model_with_bus):
        result = await add_line("test", "line0", "bus0", "bus1", x=0.1)
        assert "message" in result
        assert "deprecation_notice" in result

    async def test_rejects_missing_bus(self, model_with_bus):
        result = await add_line("test", "line0", "bus0", "nonexistent", x=0.1)
        assert "error" in result
