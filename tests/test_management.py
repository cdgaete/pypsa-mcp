import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.management import (
    create_energy_model,
    delete_model,
    export_model_summary,
    list_models,
)


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


class TestCreateEnergyModel:
    async def test_creates_model(self):
        result = await create_energy_model("test")
        assert result["model_id"] == "test"
        assert "test" in MODELS
        assert isinstance(MODELS["test"], pypsa.Network)

    async def test_rejects_duplicate(self):
        await create_energy_model("test")
        result = await create_energy_model("test")
        assert "error" in result

    async def test_override(self):
        await create_energy_model("test")
        result = await create_energy_model("test", override=True)
        assert "error" not in result

    async def test_custom_name(self):
        result = await create_energy_model("test", name="My Model")
        assert result["name"] == "My Model"


class TestListModels:
    async def test_empty(self):
        result = await list_models()
        assert result["count"] == 0

    async def test_with_models(self):
        await create_energy_model("a")
        await create_energy_model("b")
        result = await list_models()
        assert result["count"] == 2


class TestDeleteModel:
    async def test_deletes_existing(self):
        await create_energy_model("test")
        result = await delete_model("test")
        assert "test" not in MODELS
        assert "message" in result

    async def test_error_on_missing(self):
        result = await delete_model("nonexistent")
        assert "error" in result


class TestExportModelSummary:
    async def test_basic_summary(self):
        await create_energy_model("test")
        result = await export_model_summary("test")
        assert "summary" in result
        assert result["summary"]["model_id"] == "test"

    async def test_includes_investment_fields(self):
        await create_energy_model("test")
        result = await export_model_summary("test")
        summary = result["summary"]
        assert "has_investment_periods" in summary
        assert "has_scenarios" in summary

    async def test_error_on_missing(self):
        result = await export_model_summary("nonexistent")
        assert "error" in result
