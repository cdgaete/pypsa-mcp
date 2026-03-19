import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsamcp.core import (
    MODELS,
    VALID_COMPONENT_TYPES,
    convert_to_serializable,
    get_energy_model,
    mcp,
)


class TestGetEnergyModel:
    def setup_method(self):
        MODELS.clear()

    def teardown_method(self):
        MODELS.clear()

    def test_returns_network_when_exists(self):
        n = pypsa.Network()
        MODELS["test"] = n
        assert get_energy_model("test") is n

    def test_raises_on_missing_model(self):
        with pytest.raises(ValueError, match="not found"):
            get_energy_model("nonexistent")

    def test_error_lists_available_models(self):
        MODELS["a"] = pypsa.Network()
        MODELS["b"] = pypsa.Network()
        with pytest.raises(ValueError, match="'a'"):
            get_energy_model("missing")


class TestConvertToSerializable:
    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = convert_to_serializable(df)
        assert result == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    def test_series(self):
        s = pd.Series({"x": 1.0, "y": 2.0})
        result = convert_to_serializable(s)
        assert result == {"x": 1.0, "y": 2.0}

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = convert_to_serializable(arr)
        assert result == [1, 2, 3]

    def test_numpy_scalar(self):
        val = np.float64(3.14)
        result = convert_to_serializable(val)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_passthrough(self):
        assert convert_to_serializable("hello") == "hello"
        assert convert_to_serializable(42) == 42


class TestValidComponentTypes:
    def test_has_13_types(self):
        assert len(VALID_COMPONENT_TYPES) == 13

    def test_excludes_internal_types(self):
        assert "SubNetwork" not in VALID_COMPONENT_TYPES
        assert "Shape" not in VALID_COMPONENT_TYPES
        assert "sub_networks" not in VALID_COMPONENT_TYPES
        assert "shapes" not in VALID_COMPONENT_TYPES

    def test_maps_camelcase_to_list_name(self):
        assert VALID_COMPONENT_TYPES["Bus"] == "buses"
        assert VALID_COMPONENT_TYPES["Generator"] == "generators"
        assert VALID_COMPONENT_TYPES["StorageUnit"] == "storage_units"

    def test_bus_in_types(self):
        expected = [
            "Bus", "Generator", "Load", "Line", "Link",
            "StorageUnit", "Store", "Transformer", "ShuntImpedance",
            "Carrier", "GlobalConstraint", "LineType", "TransformerType",
        ]
        for t in expected:
            assert t in VALID_COMPONENT_TYPES


class TestMcpObject:
    def test_mcp_name(self):
        assert mcp.name == "pypsa-mcp"
