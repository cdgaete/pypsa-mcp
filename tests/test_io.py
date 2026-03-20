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
