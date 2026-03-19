import pytest
import pypsa
import pandas as pd

from pypsamcp.core import MODELS
from pypsamcp.tools.clustering import cluster_network


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def multi_bus_model():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
    for i in range(5):
        n.add("Bus", f"bus{i}", x=float(i), y=float(i))
        n.add("Generator", f"gen{i}", bus=f"bus{i}", p_nom=100, marginal_cost=10 + i)
        n.add("Load", f"load{i}", bus=f"bus{i}", p_set=50)
    for i in range(4):
        n.add("Line", f"line{i}", bus0=f"bus{i}", bus1=f"bus{i+1}", x=0.1, s_nom=200)
    MODELS["test"] = n
    return n


class TestClusterSpatial:
    async def test_invalid_model(self):
        result = await cluster_network("nonexistent", "spatial", "kmeans", "out", n_clusters=2)
        assert "error" in result

    async def test_invalid_domain(self, multi_bus_model):
        result = await cluster_network("test", "invalid", "kmeans", "out", n_clusters=2)
        assert "error" in result

    async def test_invalid_method(self, multi_bus_model):
        result = await cluster_network("test", "spatial", "invalid", "out", n_clusters=2)
        assert "error" in result


class TestClusterTemporal:
    async def test_resample(self, multi_bus_model):
        result = await cluster_network("test", "temporal", "resample", "resampled", offset="3h")
        assert "message" in result
        assert "resampled" in MODELS
        assert result["output_snapshots"] < result["input_snapshots"]

    async def test_downsample(self, multi_bus_model):
        result = await cluster_network("test", "temporal", "downsample", "downsampled", stride=4)
        assert "message" in result
        assert "downsampled" in MODELS
