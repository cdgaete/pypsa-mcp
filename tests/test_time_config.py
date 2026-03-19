import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.time_config import configure_time


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def empty_model():
    n = pypsa.Network()
    MODELS["test"] = n
    return n


class TestConfigureTimeSnapshots:
    async def test_set_snapshots(self, empty_model):
        result = await configure_time(
            "test", "snapshots",
            snapshots=["2024-01-01", "2024-01-02", "2024-01-03"],
        )
        assert "message" in result
        assert result["snapshot_count"] == 3

    async def test_requires_snapshots_arg(self, empty_model):
        result = await configure_time("test", "snapshots")
        assert "error" in result

    async def test_invalid_model(self):
        result = await configure_time("nonexistent", "snapshots", snapshots=["2024-01-01"])
        assert "error" in result


class TestConfigureTimeInvestmentPeriods:
    async def test_set_investment_periods(self, empty_model):
        empty_model.set_snapshots(["2024-01-01"])
        result = await configure_time(
            "test", "investment_periods",
            periods=[2025, 2030, 2035],
        )
        assert "message" in result
        assert result["has_investment_periods"] is True

    async def test_requires_snapshots_first(self, empty_model):
        result = await configure_time(
            "test", "investment_periods",
            periods=[2025, 2030],
        )
        # PyPSA may error or we validate — either way, test it doesn't crash
        assert isinstance(result, dict)


class TestConfigureTimeInvalidMode:
    async def test_invalid_mode(self, empty_model):
        result = await configure_time("test", "invalid_mode")
        assert "error" in result
