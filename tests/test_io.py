import os
import tempfile

import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.io import network_io


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def simple_model():
    n = pypsa.Network()
    n.name = "test_model"
    n.set_snapshots(["2024-01-01"])
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100)
    MODELS["test"] = n
    return n


class TestExportNetcdf:
    async def test_export(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name
        try:
            result = await network_io("test", "export_netcdf", path=path)
            assert "message" in result
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestExportCsv:
    async def test_export(self, simple_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await network_io("test", "export_csv", path=tmpdir)
            assert "message" in result


class TestImportNetcdf:
    async def test_import(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name
        try:
            simple_model.export_to_netcdf(path)
            result = await network_io("imported", "import_netcdf", path=path)
            assert "message" in result
            assert "imported" in MODELS
            assert len(MODELS["imported"].buses) == 1
        finally:
            os.unlink(path)


class TestCopy:
    async def test_copy(self, simple_model):
        result = await network_io("test", "copy", output_model_id="copy1")
        assert "message" in result
        assert "copy1" in MODELS
        assert len(MODELS["copy1"].buses) == 1

    async def test_copy_solved_network(self):
        """Copying a solved network should work — clear solver model first."""
        import pandas as pd
        from pypsamcp.tools.simulation import run_simulation
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=6, freq="h"))
        n.add("Bus", "north", v_nom=110)
        n.add("Bus", "south", v_nom=110)
        n.add("Generator", "solar", bus="north", p_nom=200, marginal_cost=0,
              p_nom_extendable=True, capital_cost=50000)
        n.add("Generator", "gas", bus="south", p_nom=500, marginal_cost=40)
        n.add("StorageUnit", "battery", bus="south", p_nom=50, max_hours=4)
        n.add("Load", "demand", bus="south", p_set=100)
        n.add("Line", "link", bus0="north", bus1="south", x=0.1, s_nom=100)
        MODELS["complex"] = n
        await run_simulation("complex", mode="optimize")
        result = await network_io("complex", "copy", output_model_id="complex_copy")
        assert "message" in result
        assert "complex_copy" in MODELS


class TestMerge:
    async def test_merge(self, simple_model):
        n2 = pypsa.Network()
        n2.set_snapshots(["2024-01-01"])
        n2.add("Bus", "bus1")
        MODELS["other"] = n2
        result = await network_io(
            "test", "merge",
            other_model_id="other",
            output_model_id="merged",
        )
        assert "message" in result
        assert "merged" in MODELS


class TestConsistencyCheck:
    async def test_consistency_check(self, simple_model):
        result = await network_io("test", "consistency_check")
        assert "message" in result


class TestInvalidOperation:
    async def test_invalid(self, simple_model):
        result = await network_io("test", "invalid_op")
        assert "error" in result
