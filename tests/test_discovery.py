import pytest

from pypsamcp.tools.discovery import describe_component, list_component_types


class TestListComponentTypes:
    async def test_returns_13_types(self):
        result = await list_component_types()
        assert len(result["component_types"]) == 13

    async def test_excludes_internal(self):
        result = await list_component_types()
        names = [c["type"] for c in result["component_types"]]
        assert "SubNetwork" not in names
        assert "Shape" not in names

    async def test_has_required_fields(self):
        result = await list_component_types()
        for comp in result["component_types"]:
            assert "type" in comp
            assert "list_name" in comp
            assert "description" in comp

    async def test_bus_present(self):
        result = await list_component_types()
        bus = next(c for c in result["component_types"] if c["type"] == "Bus")
        assert bus["list_name"] == "buses"


class TestDescribeComponent:
    async def test_valid_type(self):
        result = await describe_component("Generator")
        assert "required" in result
        assert "static" in result
        assert "varying" in result
        assert "note" in result

    async def test_invalid_type(self):
        result = await describe_component("FakeType")
        assert "error" in result

    async def test_generator_has_bus_required(self):
        result = await describe_component("Generator")
        required_attrs = [p["attr"] for p in result["required"]]
        assert "bus" in required_attrs

    async def test_accepts_list_name_form(self):
        result = await describe_component("generators")
        assert "error" not in result
        assert result["component_type"] == "Generator"

    async def test_include_defaults_false(self):
        result = await describe_component("Bus", include_defaults=False)
        for p in result["static"]:
            assert "default" not in p

    async def test_varying_fields_exist_for_generator(self):
        result = await describe_component("Generator")
        varying_attrs = [p["attr"] for p in result["varying"]]
        assert len(varying_attrs) > 0
