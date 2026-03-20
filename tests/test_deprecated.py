import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.deprecated import run_optimization, run_powerflow, set_snapshots


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


class TestSetSnapshots:
    async def test_delegates_to_configure_time(self):
        MODELS["test"] = pypsa.Network()
        result = await set_snapshots("test", ["2024-01-01", "2024-01-02"])
        assert "deprecation_notice" in result
        assert len(MODELS["test"].snapshots) == 2


class TestRunPowerflow:
    async def test_delegates_to_run_simulation(self):
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "bus0", v_nom=110)
        n.add("Bus", "bus1", v_nom=110)
        n.add("Generator", "gen0", bus="bus0", p_set=100, control="PQ")
        n.add("Load", "load0", bus="bus1", p_set=50)
        n.add("Line", "line0", bus0="bus0", bus1="bus1", x=0.1, r=0.01, s_nom=200)
        MODELS["test"] = n
        result = await run_powerflow("test")
        assert "deprecation_notice" in result


class TestRunOptimization:
    async def test_delegates_to_run_simulation(self):
        n = pypsa.Network()
        n.set_snapshots(["2024-01-01"])
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=50)
        MODELS["test"] = n
        result = await run_optimization("test")
        assert "deprecation_notice" in result
        assert result["status"] == "ok"
