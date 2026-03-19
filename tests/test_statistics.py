import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.statistics import get_statistics


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
async def solved_model():
    n = pypsa.Network()
    n.set_snapshots(["2024-01-01"])
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10, carrier="gas")
    n.add("Load", "load0", bus="bus0", p_set=50)
    MODELS["test"] = n
    await run_simulation("test", mode="optimize")
    return n


class TestGetStatistics:
    async def test_system_cost(self, solved_model):
        result = await get_statistics("test", "system_cost")
        assert "result" in result
        assert result["metric"] == "system_cost"

    async def test_capex(self, solved_model):
        result = await get_statistics("test", "capex")
        assert "result" in result

    async def test_all_metrics(self, solved_model):
        result = await get_statistics("test", "all")
        assert "result" in result
        assert isinstance(result["result"], dict)
        assert "system_cost" in result["result"]

    async def test_invalid_metric(self, solved_model):
        result = await get_statistics("test", "fake_metric")
        assert "error" in result

    async def test_unsolved_model(self):
        n = pypsa.Network()
        MODELS["unsolved"] = n
        result = await get_statistics("unsolved", "system_cost")
        assert "error" in result

    async def test_missing_model(self):
        result = await get_statistics("nonexistent", "system_cost")
        assert "error" in result
