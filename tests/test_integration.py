"""End-to-end integration test: build -> configure -> optimize -> query statistics."""

import pytest

from pypsamcp.core import MODELS
from pypsamcp.tools.management import create_energy_model, export_model_summary, list_models
from pypsamcp.tools.components import add_component, query_components
from pypsamcp.tools.time_config import configure_time
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.statistics import get_statistics
from pypsamcp.tools.discovery import list_component_types, describe_component


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


class TestFullWorkflow:
    async def test_build_optimize_analyze(self):
        # 1. Create model
        result = await create_energy_model("demo")
        assert "error" not in result

        # 2. Configure time
        result = await configure_time(
            "demo", "snapshots",
            snapshots=["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
        )
        assert "error" not in result
        assert result["snapshot_count"] == 3

        # 3. Add components via generic tool
        result = await add_component("demo", "Bus", "north")
        assert "error" not in result

        result = await add_component("demo", "Bus", "south")
        assert "error" not in result

        result = await add_component(
            "demo", "Generator", "solar_north",
            {"bus": "north", "p_nom": 200, "p_nom_extendable": True,
             "capital_cost": 50000, "marginal_cost": 0, "carrier": "solar"},
            {"p_max_pu": [0.0, 0.8, 0.6]},
        )
        assert "error" not in result

        result = await add_component(
            "demo", "Generator", "gas_south",
            {"bus": "south", "p_nom": 500, "marginal_cost": 40, "carrier": "gas"},
        )
        assert "error" not in result

        result = await add_component(
            "demo", "Load", "demand_south",
            {"bus": "south"},
            {"p_set": [100, 150, 120]},
        )
        assert "error" not in result

        result = await add_component(
            "demo", "Line", "north_south",
            {"bus0": "north", "bus1": "south", "x": 0.1, "s_nom": 100},
        )
        assert "error" not in result

        # 4. Query components
        result = await query_components("demo", "Generator")
        assert result["count"] == 2

        # 5. Optimize
        result = await run_simulation("demo", mode="optimize")
        assert result["status"] == "ok"
        assert result["termination_condition"] == "optimal"
        assert isinstance(result["objective_value"], (int, float))

        # 6. Get statistics
        result = await get_statistics("demo", "system_cost")
        assert "result" in result

        # 7. Export summary
        result = await export_model_summary("demo")
        assert "error" not in result
        assert result["summary"]["components"]["Generator"]["count"] == 2

    async def test_discovery_flow(self):
        result = await list_component_types()
        assert len(result["component_types"]) == 13

        result = await describe_component("Generator")
        assert len(result["required"]) > 0
        assert len(result["varying"]) > 0

    async def test_tool_count(self):
        """Verify we have exactly 22 registered tools."""
        from pypsamcp.core import mcp

        tools = await mcp.list_tools()
        tool_names = sorted(t.name for t in tools)
        assert len(tools) == 22, f"Expected 22 tools, got {len(tools)}: {tool_names}"
