import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.simulation import run_simulation


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def simple_model():
    """A minimal solvable model."""
    n = pypsa.Network()
    n.set_snapshots(["2024-01-01"])
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10)
    n.add("Load", "load0", bus="bus0", p_set=50)
    MODELS["test"] = n
    return n


@pytest.fixture
def pf_model():
    """A model suitable for power flow."""
    n = pypsa.Network()
    n.set_snapshots(["2024-01-01"])
    n.add("Bus", "bus0", v_nom=110)
    n.add("Bus", "bus1", v_nom=110)
    n.add("Generator", "gen0", bus="bus0", p_set=100, control="PQ")
    n.add("Load", "load0", bus="bus1", p_set=50)
    n.add("Line", "line0", bus0="bus0", bus1="bus1", x=0.1, r=0.01, s_nom=200)
    MODELS["test"] = n
    return n


class TestRunSimulationOptimize:
    async def test_basic_optimize(self, simple_model):
        result = await run_simulation("test", mode="optimize")
        assert result["status"] == "ok"
        assert result["termination_condition"] == "optimal"
        assert result["objective_value"] == pytest.approx(500.0)

    async def test_missing_model(self):
        result = await run_simulation("nonexistent")
        assert "error" in result

    async def test_invalid_mode(self, simple_model):
        result = await run_simulation("test", mode="invalid")
        assert "error" in result


class TestRunSimulationPF:
    async def test_basic_pf(self, pf_model):
        result = await run_simulation("test", mode="pf")
        assert "summary" in result
        assert result["mode"] == "pf"

    async def test_lpf(self, pf_model):
        result = await run_simulation("test", mode="lpf")
        assert result["mode"] == "lpf"


class TestFormulationNotLeaked:
    """formulation param must only be passed to network.optimize(), not to
    rolling_horizon/security_constrained/etc which forward it as a solver
    option causing HiGHS errors."""

    async def test_rolling_horizon_kwargs_exclude_formulation(self):
        """_run_rolling_horizon should not include formulation in kwargs."""
        from pypsamcp.tools.simulation import _run_rolling_horizon
        import inspect
        sig = inspect.signature(_run_rolling_horizon)
        assert "formulation" not in sig.parameters

    async def test_security_constrained_kwargs_exclude_formulation(self):
        from pypsamcp.tools.simulation import _run_security_constrained
        import inspect
        sig = inspect.signature(_run_security_constrained)
        assert "formulation" not in sig.parameters

    async def test_transmission_expansion_kwargs_exclude_formulation(self):
        from pypsamcp.tools.simulation import _run_transmission_expansion
        import inspect
        sig = inspect.signature(_run_transmission_expansion)
        assert "formulation" not in sig.parameters

    async def test_optimize_and_pf_kwargs_exclude_formulation(self):
        from pypsamcp.tools.simulation import _run_optimize_and_pf
        import inspect
        sig = inspect.signature(_run_optimize_and_pf)
        assert "formulation" not in sig.parameters
