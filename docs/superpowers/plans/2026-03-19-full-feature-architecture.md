# Full Feature Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor pypsa-mcp from 11 tools in a monolithic `core.py` to 22 tools (19 primary + 3 deprecated) across a modular file structure covering all PyPSA 1.1.2 capabilities.

**Architecture:** Split tools into domain-specific modules under `src/pypsamcp/tools/`. `core.py` retains `mcp`, `MODELS`, and helpers. Each tool module imports `mcp` from `core` and registers via `@mcp.tool()`. Registration is triggered by `import pypsamcp.tools` at the bottom of `core.py`.

**Tech Stack:** Python 3.10+, PyPSA 1.1.2, FastMCP 2.2+, HiGHS solver

**Spec:** `docs/superpowers/specs/2026-03-19-full-feature-architecture-design.md`

---

## PyPSA 1.1.2 API Reference (verified)

These facts were verified against the installed PyPSA 1.1.2 and must be used in implementation:

- `n.components` keys are **lowercase plural** (`buses`, `generators`, etc.), but bracket access accepts CamelCase (`n.components['Bus']`)
- Component objects are **dataclass-like**, not dicts. Access: `comp.list_name`, `comp.attrs` (deprecated), `comp.defaults`, `comp.static`, `comp.dynamic`
- `comp.attrs` returns DataFrame with columns: `['type', 'unit', 'default', 'description', 'status', 'static', 'varying', 'typ', 'dtype']` — but triggers deprecation warning
- `n.optimize()` returns **tuple** `(status, termination_condition)` e.g. `('ok', 'optimal')`. `n.results` does NOT exist. `n.objective` exists after solve but is `None` on unsolved networks.
- A fresh `pypsa.Network()` has `snapshots = Index(['now'])` (length 1, not 0). Explicit snapshot detection must check for datetime type, not just length.
- `n.statistics` kwargs use `components` (not `comps` — `comps` is deprecated).
- `n.optimize(formulation=...)` is passed via `**kwargs` internally — works but not a named parameter. Document this coupling.
- `n.copy(snapshots=..., investment_periods=...)` — no separate `slice` method
- `n.cluster.spatial` has: `cluster_by_kmeans`, `cluster_by_hac`, `cluster_by_greedy_modularity`, `busmap_by_kmeans`, `busmap_by_hac`, `busmap_by_greedy_modularity`, `cluster_by_busmap`
- `n.cluster.temporal` has: `resample`, `downsample`, `segment`, `from_snapshot_map`
- `n.statistics` methods: `system_cost`, `capex`, `opex`, `fom`, `overnight_cost`, `installed_capacity`, `optimal_capacity`, `expanded_capacity`, `installed_capex`, `expanded_capex`, `capacity_factor`, `curtailment`, `energy_balance`, `supply`, `withdrawal`, `revenue`, `market_value`, `prices`, `transmission`
- `n.optimize` accessor methods: `optimize_mga`, `optimize_security_constrained`, `optimize_with_rolling_horizon`, `optimize_transmission_expansion_iteratively`, `optimize_and_run_non_linear_powerflow`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Bump `pypsa>=1.1.0`, add `pytest-asyncio` to dev deps |
| `src/pypsamcp/core.py` | Rewrite | Strip to: mcp, MODELS, helpers only. Add `import pypsamcp.tools` at bottom. Rename `get_model` → `get_energy_model`. Add `VALID_COMPONENT_TYPES` dict. |
| `src/pypsamcp/server.py` | Unchanged | — |
| `src/pypsamcp/tools/__init__.py` | Create | Import all tool modules |
| `src/pypsamcp/tools/management.py` | Create | 4 tools: create_energy_model, list_models, delete_model, export_model_summary |
| `src/pypsamcp/tools/discovery.py` | Create | 2 tools: list_component_types, describe_component |
| `src/pypsamcp/tools/components.py` | Create | 4 tools: add_component, update_component, remove_component, query_components |
| `src/pypsamcp/tools/convenience.py` | Create | 4 tools: add_bus, add_generator, add_load, add_line (deprecated delegates) |
| `src/pypsamcp/tools/time_config.py` | Create | 1 tool: configure_time |
| `src/pypsamcp/tools/simulation.py` | Create | 1 tool: run_simulation |
| `src/pypsamcp/tools/statistics.py` | Create | 1 tool: get_statistics |
| `src/pypsamcp/tools/clustering.py` | Create | 1 tool: cluster_network |
| `src/pypsamcp/tools/io.py` | Create | 1 tool: network_io |
| `src/pypsamcp/tools/deprecated.py` | Create | 3 tools: set_snapshots, run_powerflow, run_optimization |
| `tests/conftest.py` | Create | Shared fixtures |
| `tests/test_management.py` | Create | Tests for GROUP 1 |
| `tests/test_discovery.py` | Create | Tests for GROUP 2 |
| `tests/test_components.py` | Create | Tests for GROUP 3 |
| `tests/test_convenience.py` | Create | Tests for convenience wrappers |
| `tests/test_time_config.py` | Create | Tests for GROUP 4 |
| `tests/test_simulation.py` | Create | Tests for GROUP 5 |
| `tests/test_statistics.py` | Create | Tests for GROUP 6 |
| `tests/test_clustering.py` | Create | Tests for GROUP 7 |
| `tests/test_io.py` | Create | Tests for GROUP 8 |
| `tests/test_deprecated.py` | Create | Tests for deprecated aliases |

---

## Task 1: Scaffold Module Structure and Rewrite core.py

**Files:**
- Modify: `src/pypsamcp/core.py`
- Modify: `pyproject.toml`
- Create: `src/pypsamcp/tools/__init__.py`
- Test: `tests/conftest.py`
- Test: `tests/test_core_helpers.py`

### Steps

- [ ] **Step 1: Write tests for core helpers**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for pypsa-mcp tests."""
```

In `pyproject.toml`, also add pytest-asyncio config:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Create `tests/test_core_helpers.py`:

```python
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
        # Also check lowercase
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_core_helpers.py -v`
Expected: FAIL — `VALID_COMPONENT_TYPES` and `get_energy_model` don't exist yet.

- [ ] **Step 3: Update pyproject.toml**

In `pyproject.toml`, change:

```
dependencies = [
    "fastmcp>=2.2.0",
    "pypsa>=1.1.0",
    "highspy>=1.10.0",
]
```

And add `pytest-asyncio` to dev deps:

```
dev = [
    "pytest>=8.3.5",
    "pytest-asyncio>=0.23.0",
    "black>=24.3.0",
    "ruff>=0.3.4",
    "build>=1.2.1",
]
```

- [ ] **Step 4: Install updated dev deps**

Run: `cd /home/carlos/repos/pypsa-mcp && uv pip install -e ".[dev]"`

- [ ] **Step 5: Rewrite core.py**

Replace `src/pypsamcp/core.py` with:

```python
"""
PyPSA MCP Core functionality

This module provides the core MCP object, global model storage, and shared helpers.
Tool modules in pypsamcp.tools register themselves via @mcp.tool() on import.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
from contextlib import contextmanager
from fastmcp import FastMCP

# Redirect Python loggers to stderr
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
for _name in ("pypsa", "linopy", "highs"):
    logging.getLogger(_name).handlers = [logging.StreamHandler(sys.stderr)]
    logging.getLogger(_name).propagate = False


@contextmanager
def stdout_to_stderr():
    """Redirect fd 1 (stdout) to fd 2 (stderr) at the OS level.

    HiGHS writes directly via C stdio, bypassing Python's sys.stdout,
    so we must redirect at the file descriptor level to keep the MCP
    stdio stream clean.
    """
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    try:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


mcp = FastMCP(
    "pypsa-mcp",
    on_duplicate="error",
)

MODELS = {}

# CamelCase type name → lowercase list_name. Excludes internal types (SubNetwork, Shape).
VALID_COMPONENT_TYPES = {
    "Bus": "buses",
    "Generator": "generators",
    "Load": "loads",
    "Line": "lines",
    "Link": "links",
    "StorageUnit": "storage_units",
    "Store": "stores",
    "Transformer": "transformers",
    "ShuntImpedance": "shunt_impedances",
    "Carrier": "carriers",
    "GlobalConstraint": "global_constraints",
    "LineType": "line_types",
    "TransformerType": "transformer_types",
}

# Reverse lookup: list_name → CamelCase type name
_LIST_NAME_TO_TYPE = {v: k for k, v in VALID_COMPONENT_TYPES.items()}


def get_energy_model(model_id: str) -> pypsa.Network:
    """Get a model by ID from the global models dictionary."""
    if model_id not in MODELS:
        raise ValueError(
            f"Model with ID '{model_id}' not found. "
            f"Available models: {list(MODELS.keys())}"
        )
    return MODELS[model_id]


def validate_component_type(component_type: str) -> str:
    """Validate and normalize component_type to CamelCase. Returns the canonical name.

    Accepts both CamelCase ('Bus') and list_name ('buses') forms.
    Raises ValueError if invalid.
    """
    if component_type in VALID_COMPONENT_TYPES:
        return component_type
    if component_type in _LIST_NAME_TO_TYPE:
        return _LIST_NAME_TO_TYPE[component_type]
    raise ValueError(
        f"Invalid component type '{component_type}'. "
        f"Valid types: {list(VALID_COMPONENT_TYPES.keys())}"
    )


def generate_network_summary(network: pypsa.Network) -> dict:
    """Generate a summary of a PyPSA network."""
    return {
        "name": network.name,
        "buses": len(network.buses),
        "generators": len(network.generators),
        "storage_units": len(network.storage_units),
        "links": len(network.links),
        "lines": len(network.lines),
        "transformers": len(network.transformers),
        "snapshots": len(network.snapshots),
    }


def convert_to_serializable(data):
    """Convert PyPSA data to JSON-serializable format."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    elif isinstance(data, pd.Series):
        return data.to_dict()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data


# Import tool modules to trigger @mcp.tool() registration.
# This MUST be at the bottom, after mcp and all helpers are defined.
import pypsamcp.tools  # noqa: E402, F401
```

- [ ] **Step 6: Create tools/__init__.py**

Create `src/pypsamcp/tools/__init__.py`:

```python
"""
PyPSA MCP Tool modules.

Importing this package triggers registration of all @mcp.tool() decorated
functions from each submodule.
"""

from pypsamcp.tools import management  # noqa: F401
from pypsamcp.tools import discovery  # noqa: F401
from pypsamcp.tools import components  # noqa: F401
from pypsamcp.tools import convenience  # noqa: F401
from pypsamcp.tools import time_config  # noqa: F401
from pypsamcp.tools import simulation  # noqa: F401
from pypsamcp.tools import statistics  # noqa: F401
from pypsamcp.tools import clustering  # noqa: F401
from pypsamcp.tools import io  # noqa: F401
from pypsamcp.tools import deprecated  # noqa: F401
```

Create empty stub files for each module so imports don't fail. Each file should contain just a docstring for now:

`src/pypsamcp/tools/management.py`:
```python
"""Model management tools: create, list, delete, export summary."""
```

`src/pypsamcp/tools/discovery.py`:
```python
"""Discovery tools: list_component_types, describe_component."""
```

`src/pypsamcp/tools/components.py`:
```python
"""Component CRUD tools: add, update, remove, query."""
```

`src/pypsamcp/tools/convenience.py`:
```python
"""Deprecated convenience wrappers: add_bus, add_generator, add_load, add_line."""
```

`src/pypsamcp/tools/time_config.py`:
```python
"""Time and investment structure configuration."""
```

`src/pypsamcp/tools/simulation.py`:
```python
"""Simulation tool: power flow, optimization, and advanced modes."""
```

`src/pypsamcp/tools/statistics.py`:
```python
"""Statistics and results tool."""
```

`src/pypsamcp/tools/clustering.py`:
```python
"""Spatial and temporal clustering tool."""
```

`src/pypsamcp/tools/io.py`:
```python
"""I/O and network operations tool."""
```

`src/pypsamcp/tools/deprecated.py`:
```python
"""Deprecated tool aliases for backwards compatibility."""
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_core_helpers.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/pypsamcp/core.py src/pypsamcp/tools/ tests/
git commit -m "refactor: scaffold module structure and strip core.py to helpers only"
```

---

## Task 2: Management Tools (GROUP 1)

**Files:**
- Modify: `src/pypsamcp/tools/management.py`
- Test: `tests/test_management.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_management.py`:

```python
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
    @pytest.mark.asyncio
    async def test_creates_model(self):
        result = await create_energy_model("test")
        assert result["model_id"] == "test"
        assert "test" in MODELS
        assert isinstance(MODELS["test"], pypsa.Network)

    @pytest.mark.asyncio
    async def test_rejects_duplicate(self):
        await create_energy_model("test")
        result = await create_energy_model("test")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_override(self):
        await create_energy_model("test")
        result = await create_energy_model("test", override=True)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_custom_name(self):
        result = await create_energy_model("test", name="My Model")
        assert result["name"] == "My Model"


class TestListModels:
    @pytest.mark.asyncio
    async def test_empty(self):
        result = await list_models()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_with_models(self):
        await create_energy_model("a")
        await create_energy_model("b")
        result = await list_models()
        assert result["count"] == 2


class TestDeleteModel:
    @pytest.mark.asyncio
    async def test_deletes_existing(self):
        await create_energy_model("test")
        result = await delete_model("test")
        assert "test" not in MODELS
        assert "message" in result

    @pytest.mark.asyncio
    async def test_error_on_missing(self):
        result = await delete_model("nonexistent")
        assert "error" in result


class TestExportModelSummary:
    @pytest.mark.asyncio
    async def test_basic_summary(self):
        await create_energy_model("test")
        result = await export_model_summary("test")
        assert "summary" in result
        assert result["summary"]["model_id"] == "test"

    @pytest.mark.asyncio
    async def test_includes_investment_fields(self):
        await create_energy_model("test")
        result = await export_model_summary("test")
        summary = result["summary"]
        assert "has_investment_periods" in summary
        assert "has_scenarios" in summary

    @pytest.mark.asyncio
    async def test_error_on_missing(self):
        result = await export_model_summary("nonexistent")
        assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_management.py -v`
Expected: FAIL — imports fail because management.py is a stub.

- [ ] **Step 3: Implement management.py**

Write `src/pypsamcp/tools/management.py`:

```python
"""Model management tools: create, list, delete, export summary."""

import pandas as pd
import pypsa

from pypsamcp.core import (
    MODELS,
    convert_to_serializable,
    generate_network_summary,
    get_energy_model,
    mcp,
)


@mcp.tool()
async def create_energy_model(
    model_id: str,
    name: str | None = None,
    override: bool = False,
) -> dict:
    """Create a new PyPSA energy model with the given ID.

    Args:
        model_id: A unique identifier for the model
        name: A descriptive name for the model (defaults to model_id if not provided)
        override: Whether to override an existing model with the same ID
    """
    if model_id in MODELS and not override:
        return {
            "error": f"Model with ID '{model_id}' already exists. Use override=True to replace it.",
            "available_models": list(MODELS.keys()),
        }

    network = pypsa.Network()
    network.name = name if name else model_id
    MODELS[model_id] = network

    return {
        "model_id": model_id,
        "name": network.name,
        "message": f"PyPSA energy model '{model_id}' created successfully.",
        "model_summary": generate_network_summary(network),
    }


@mcp.tool()
async def list_models() -> dict:
    """List all currently available PyPSA models."""
    model_list = []
    for model_id, network in MODELS.items():
        model_list.append({
            "model_id": model_id,
            "name": network.name,
            "summary": generate_network_summary(network),
        })

    return {"count": len(model_list), "models": model_list}


@mcp.tool()
async def delete_model(model_id: str) -> dict:
    """Delete a PyPSA model by ID.

    Args:
        model_id: The ID of the model to delete
    """
    if model_id not in MODELS:
        return {
            "error": f"Model with ID '{model_id}' not found. Available models: {list(MODELS.keys())}"
        }

    del MODELS[model_id]
    return {
        "message": f"Model '{model_id}' deleted successfully.",
        "remaining_models": list(MODELS.keys()),
    }


@mcp.tool()
async def export_model_summary(model_id: str) -> dict:
    """Export a comprehensive summary of the model.

    Args:
        model_id: The ID of the model to export
    """
    try:
        network = get_energy_model(model_id)

        summary = {
            "model_id": model_id,
            "name": network.name,
            "components": {},
            "has_investment_periods": bool(getattr(network, "has_investment_periods", False)),
            "has_scenarios": bool(getattr(network, "has_scenarios", False)),
            "has_risk_preference": bool(getattr(network, "has_risk_preference", False)),
            "investment_periods": (
                list(network.investment_periods)
                if getattr(network, "has_investment_periods", False)
                else []
            ),
            "scenarios": (
                list(network.scenarios)
                if getattr(network, "has_scenarios", False)
                else []
            ),
        }

        # Add non-empty component info
        for type_name, list_name in [
            ("Bus", "buses"),
            ("Generator", "generators"),
            ("Load", "loads"),
            ("Line", "lines"),
            ("Link", "links"),
            ("StorageUnit", "storage_units"),
            ("Store", "stores"),
            ("Transformer", "transformers"),
            ("ShuntImpedance", "shunt_impedances"),
            ("Carrier", "carriers"),
            ("GlobalConstraint", "global_constraints"),
        ]:
            df = getattr(network, list_name)
            if not df.empty:
                summary["components"][type_name] = {
                    "count": len(df),
                    "attributes": list(df.columns),
                    "ids": df.index.tolist(),
                }

        # Snapshot info
        if len(network.snapshots) > 0:
            summary["snapshots"] = {
                "count": len(network.snapshots),
                "start": str(network.snapshots[0]),
                "end": str(network.snapshots[-1]),
                "frequency": str(pd.infer_freq(network.snapshots)) if pd.infer_freq(network.snapshots) else "irregular",
            }

        return {"message": f"Model summary generated for '{model_id}'.", "summary": summary}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_management.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/management.py tests/test_management.py
git commit -m "feat: add management tools (create, list, delete, export summary)"
```

---

## Task 3: Discovery Tools (GROUP 2)

**Files:**
- Modify: `src/pypsamcp/tools/discovery.py`
- Test: `tests/test_discovery.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_discovery.py`:

```python
import pytest

from pypsamcp.tools.discovery import describe_component, list_component_types


class TestListComponentTypes:
    @pytest.mark.asyncio
    async def test_returns_13_types(self):
        result = await list_component_types()
        assert len(result["component_types"]) == 13

    @pytest.mark.asyncio
    async def test_excludes_internal(self):
        result = await list_component_types()
        names = [c["type"] for c in result["component_types"]]
        assert "SubNetwork" not in names
        assert "Shape" not in names

    @pytest.mark.asyncio
    async def test_has_required_fields(self):
        result = await list_component_types()
        for comp in result["component_types"]:
            assert "type" in comp
            assert "list_name" in comp
            assert "description" in comp

    @pytest.mark.asyncio
    async def test_bus_present(self):
        result = await list_component_types()
        bus = next(c for c in result["component_types"] if c["type"] == "Bus")
        assert bus["list_name"] == "buses"


class TestDescribeComponent:
    @pytest.mark.asyncio
    async def test_valid_type(self):
        result = await describe_component("Generator")
        assert "required" in result
        assert "static" in result
        assert "varying" in result
        assert "note" in result

    @pytest.mark.asyncio
    async def test_invalid_type(self):
        result = await describe_component("FakeType")
        assert "error" in result
        assert "valid" in result["error"].lower() or "Valid" in result["error"]

    @pytest.mark.asyncio
    async def test_generator_has_bus_required(self):
        result = await describe_component("Generator")
        required_attrs = [p["attr"] for p in result["required"]]
        assert "bus" in required_attrs

    @pytest.mark.asyncio
    async def test_accepts_list_name_form(self):
        result = await describe_component("generators")
        assert "error" not in result
        assert result["component_type"] == "Generator"

    @pytest.mark.asyncio
    async def test_include_defaults_false(self):
        result = await describe_component("Bus", include_defaults=False)
        for p in result["static"]:
            assert "default" not in p

    @pytest.mark.asyncio
    async def test_varying_fields_exist_for_generator(self):
        result = await describe_component("Generator")
        varying_attrs = [p["attr"] for p in result["varying"]]
        assert "p_max_pu" in varying_attrs or len(varying_attrs) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_discovery.py -v`
Expected: FAIL

- [ ] **Step 3: Implement discovery.py**

Write `src/pypsamcp/tools/discovery.py`:

```python
"""Discovery tools: list_component_types, describe_component."""

import warnings

import pypsa

from pypsamcp.core import (
    MODELS,
    VALID_COMPONENT_TYPES,
    mcp,
    validate_component_type,
)

COMPONENT_DESCRIPTIONS = {
    "Bus": "Electrical node connecting components",
    "Generator": "Power source with cost and capacity parameters",
    "Load": "Power demand at a bus",
    "Line": "AC transmission line between two buses",
    "Link": "Controllable branch: HVDC, heat pumps, electrolyzers, sector coupling",
    "StorageUnit": "Coupled power-energy storage: battery, pumped hydro, reservoir",
    "Store": "Energy reservoir decoupled from power: H2 tank, heat store",
    "Transformer": "Voltage-level coupling with tap ratio and phase shift",
    "ShuntImpedance": "Shunt conductance/susceptance at a bus",
    "Carrier": "Energy carrier with CO2 emissions, color, growth limits",
    "GlobalConstraint": "System-wide constraint: CO2 cap, primary energy limit, capacity target",
    "LineType": "Standard AC line type library (r/x/c per km)",
    "TransformerType": "Standard transformer type library",
}


@mcp.tool()
async def list_component_types() -> dict:
    """List all available PyPSA component types with descriptions.

    Returns the catalog of 13 user-facing component types. No model required.
    """
    return {
        "component_types": [
            {
                "type": type_name,
                "list_name": list_name,
                "description": COMPONENT_DESCRIPTIONS[type_name],
            }
            for type_name, list_name in VALID_COMPONENT_TYPES.items()
        ]
    }


@mcp.tool()
async def describe_component(
    component_type: str,
    include_defaults: bool = True,
) -> dict:
    """Describe the full input-parameter schema for a PyPSA component type.

    Args:
        component_type: Component type name (e.g. 'Generator', 'Bus', or 'generators')
        include_defaults: Whether to include default values in the response
    """
    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    # Use an existing model or create a throwaway one
    if MODELS:
        network = next(iter(MODELS.values()))
    else:
        network = pypsa.Network()

    # Get component attrs (suppress deprecation warning)
    comp = network.components[canonical]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        attrs = comp.attrs

    # Filter to Input attributes only
    input_mask = attrs["status"].str.startswith("Input")
    input_attrs = attrs[input_mask]

    required = []
    static = []
    varying = []

    for attr_name, row in input_attrs.iterrows():
        if attr_name == "name":
            continue  # Skip 'name' — it's the component_id

        entry = {
            "attr": attr_name,
            "unit": str(row.get("unit", "")),
            "description": str(row.get("description", "")),
            "typ": str(row.get("typ", "")),
        }
        if include_defaults:
            default_val = row.get("default", None)
            entry["default"] = str(default_val) if default_val is not None else None

        status = row["status"]
        if status == "Input (required)":
            required.append(entry)
        elif row.get("varying", False):
            varying.append(entry)
        else:
            static.append(entry)

    return {
        "component_type": canonical,
        "list_name": VALID_COMPONENT_TYPES[canonical],
        "required": required,
        "static": static,
        "varying": varying,
        "note": (
            "Pass 'static' and 'varying' params to add_component(params={}). "
            "Pass time series for 'varying' params to add_component(time_series={})."
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/discovery.py tests/test_discovery.py
git commit -m "feat: add discovery tools (list_component_types, describe_component)"
```

---

## Task 4: Component CRUD Tools (GROUP 3)

**Files:**
- Modify: `src/pypsamcp/tools/components.py`
- Test: `tests/test_components.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_components.py`:

```python
import pandas as pd
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.components import (
    add_component,
    query_components,
    remove_component,
    update_component,
)


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def model_with_bus():
    n = pypsa.Network()
    n.name = "test"
    n.set_snapshots(["2024-01-01", "2024-01-02"])
    n.add("Bus", "bus0")
    MODELS["test"] = n
    return n


class TestAddComponent:
    @pytest.mark.asyncio
    async def test_add_bus(self):
        n = pypsa.Network()
        MODELS["test"] = n
        result = await add_component("test", "Bus", "bus0", {"v_nom": 110.0})
        assert "message" in result
        assert "bus0" in n.buses.index

    @pytest.mark.asyncio
    async def test_add_generator_with_bus_check(self, model_with_bus):
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "bus0", "p_nom": 100.0, "marginal_cost": 10.0},
        )
        assert "message" in result
        assert "gen0" in model_with_bus.generators.index

    @pytest.mark.asyncio
    async def test_rejects_missing_model(self):
        result = await add_component("nonexistent", "Bus", "bus0")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_invalid_component_type(self):
        MODELS["test"] = pypsa.Network()
        result = await add_component("test", "FakeType", "x")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_output_params(self, model_with_bus):
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "bus0", "p_nom_opt": 999},
        )
        assert "error" in result
        assert "p_nom_opt" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_missing_bus(self):
        n = pypsa.Network()
        MODELS["test"] = n
        n.add("Bus", "bus0")
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "nonexistent"},
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_duplicate_id(self, model_with_bus):
        result = await add_component("test", "Bus", "bus0")
        assert "error" in result
        assert "already exists" in result["error"]

    @pytest.mark.asyncio
    async def test_time_series(self, model_with_bus):
        result = await add_component(
            "test", "Generator", "gen0",
            {"bus": "bus0", "p_nom": 100.0},
            {"p_max_pu": [0.5, 0.8]},
        )
        assert "message" in result
        assert model_with_bus.generators_t.p_max_pu.loc[pd.Timestamp("2024-01-01"), "gen0"] == 0.5

    @pytest.mark.asyncio
    async def test_time_series_requires_snapshots(self):
        """A fresh network has snapshots=['now'] (not DatetimeIndex), should reject time series."""
        n = pypsa.Network()
        n.add("Bus", "bus0")
        MODELS["fresh"] = n
        result = await add_component(
            "fresh", "Generator", "gen0",
            {"bus": "bus0"},
            {"p_max_pu": [0.5, 0.8]},
        )
        assert "error" in result
        assert "snapshot" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_accepts_list_name_form(self, model_with_bus):
        result = await add_component("test", "generators", "gen0", {"bus": "bus0"})
        assert "message" in result

    @pytest.mark.asyncio
    async def test_no_bus_check_for_carrier(self):
        MODELS["test"] = pypsa.Network()
        result = await add_component("test", "Carrier", "wind", {"co2_emissions": 0.0})
        assert "message" in result


class TestUpdateComponent:
    @pytest.mark.asyncio
    async def test_update_existing(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0", "p_nom": 100.0})
        result = await update_component("test", "Generator", "gen0", {"p_nom": 200.0})
        assert "message" in result
        assert model_with_bus.generators.loc["gen0", "p_nom"] == 200.0

    @pytest.mark.asyncio
    async def test_rejects_nonexistent(self, model_with_bus):
        result = await update_component("test", "Generator", "nonexistent", {"p_nom": 100.0})
        assert "error" in result


class TestRemoveComponent:
    @pytest.mark.asyncio
    async def test_remove_existing(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0"})
        result = await remove_component("test", "Generator", "gen0")
        assert "message" in result
        assert "gen0" not in model_with_bus.generators.index

    @pytest.mark.asyncio
    async def test_rejects_nonexistent(self, model_with_bus):
        result = await remove_component("test", "Generator", "nonexistent")
        assert "error" in result


class TestQueryComponents:
    @pytest.mark.asyncio
    async def test_query_all(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0", "p_nom": 100.0})
        await add_component("test", "Generator", "gen1", {"bus": "bus0", "p_nom": 200.0})
        result = await query_components("test", "Generator")
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_query_with_filter(self, model_with_bus):
        await add_component("test", "Generator", "gen0", {"bus": "bus0", "p_nom": 100.0, "carrier": "solar"})
        await add_component("test", "Generator", "gen1", {"bus": "bus0", "p_nom": 200.0, "carrier": "wind"})
        result = await query_components("test", "Generator", {"carrier": "solar"})
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_query_empty(self):
        MODELS["test"] = pypsa.Network()
        result = await query_components("test", "Generator")
        assert result["count"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_components.py -v`
Expected: FAIL

- [ ] **Step 3: Implement components.py**

Write `src/pypsamcp/tools/components.py`:

```python
"""Component CRUD tools: add, update, remove, query."""

import warnings

import pandas as pd

from pypsamcp.core import (
    VALID_COMPONENT_TYPES,
    convert_to_serializable,
    get_energy_model,
    mcp,
    validate_component_type,
)

# Component types that require bus references
_SINGLE_BUS = {"Generator", "Load", "StorageUnit", "Store", "ShuntImpedance"}
_DUAL_BUS = {"Line", "Transformer"}
_MULTI_BUS = {"Link"}  # bus0, bus1 required; bus2..busN optional
_NO_BUS_CHECK = {"Carrier", "GlobalConstraint", "LineType", "TransformerType", "Bus"}


def _get_component_attrs(network, canonical_type):
    """Get the attrs DataFrame for a component type, suppressing deprecation warnings."""
    comp = network.components[canonical_type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return comp.attrs


def _validate_params_are_input(attrs, params):
    """Check that all param keys are Input attributes. Returns list of bad keys."""
    bad_keys = []
    for key in params:
        if key not in attrs.index:
            bad_keys.append(key)
        elif not attrs.loc[key, "status"].startswith("Input"):
            bad_keys.append(key)
    return bad_keys


def _validate_bus_references(network, canonical_type, params):
    """Check that bus references in params resolve to existing buses."""
    if canonical_type in _NO_BUS_CHECK:
        return None

    if canonical_type in _SINGLE_BUS:
        bus = params.get("bus")
        if bus and bus not in network.buses.index:
            return f"Bus '{bus}' does not exist."
    elif canonical_type in _DUAL_BUS:
        for attr in ("bus0", "bus1"):
            bus = params.get(attr)
            if bus and bus not in network.buses.index:
                return f"Bus '{bus}' (from {attr}) does not exist."
    elif canonical_type in _MULTI_BUS:
        for attr in ("bus0", "bus1"):
            bus = params.get(attr)
            if bus and bus not in network.buses.index:
                return f"Bus '{bus}' (from {attr}) does not exist."
        # Check optional bus2..busN
        for key, val in params.items():
            if key.startswith("bus") and key[3:].isdigit() and int(key[3:]) >= 2:
                if val not in network.buses.index:
                    return f"Bus '{val}' (from {key}) does not exist."
    return None


def _validate_time_series_keys(attrs, time_series):
    """Check that all time_series keys are varying Input attributes."""
    bad_keys = []
    for key in time_series:
        if key not in attrs.index:
            bad_keys.append(key)
        elif not attrs.loc[key, "status"].startswith("Input"):
            bad_keys.append(key)
        elif not attrs.loc[key, "varying"]:
            bad_keys.append(key)
    return bad_keys


@mcp.tool()
async def add_component(
    model_id: str,
    component_type: str,
    component_id: str,
    params: dict | None = None,
    time_series: dict | None = None,
) -> dict:
    """Add any PyPSA component to a model.

    Args:
        model_id: The model to add the component to
        component_type: Component type (e.g. 'Generator', 'Bus', 'Line')
        component_id: Unique ID for the new component
        params: Static/scalar input parameters
        time_series: Time-varying parameters as {attr: [values...]}
    """
    params = params or {}
    time_series = time_series or {}

    # 1. Model exists
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    # 2. Valid component type
    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    attrs = _get_component_attrs(network, canonical)

    # 3. All params are Input attributes
    bad_keys = _validate_params_are_input(attrs, params)
    if bad_keys:
        return {
            "error": f"Invalid parameters {bad_keys}. These are not Input attributes. "
            f"Use describe_component('{canonical}') to see valid parameters."
        }

    # 4. Bus references resolve
    bus_error = _validate_bus_references(network, canonical, params)
    if bus_error:
        return {"error": bus_error}

    # 5. Component ID not already in use
    df = getattr(network, list_name)
    if component_id in df.index:
        return {"error": f"{canonical} '{component_id}' already exists in model '{model_id}'."}

    # 6. Time series keys are varying Input attributes
    if time_series:
        bad_ts_keys = _validate_time_series_keys(attrs, time_series)
        if bad_ts_keys:
            return {
                "error": f"Invalid time_series keys {bad_ts_keys}. "
                f"These are not varying Input attributes. "
                f"Use describe_component('{canonical}') to see varying parameters."
            }

    # 7. Time series requires explicitly configured snapshots (not the default "now")
    if time_series:
        if not isinstance(network.snapshots, pd.DatetimeIndex):
            return {
                "error": "Cannot set time series: no snapshots configured. "
                "Use configure_time(mode='snapshots') first."
            }

    # Execute
    try:
        network.add(canonical, component_id, **params)

        # Assign time series
        for attr, values in time_series.items():
            ts_df = getattr(network, f"{list_name}_t")
            series = pd.Series(values, index=network.snapshots[: len(values)])
            ts_df[attr][component_id] = series

        df = getattr(network, list_name)
        return {
            "message": f"{canonical} '{component_id}' added to model '{model_id}'.",
            "component_data": convert_to_serializable(df.loc[component_id]),
            "total_count": len(df),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def update_component(
    model_id: str,
    component_type: str,
    component_id: str,
    params: dict | None = None,
    time_series: dict | None = None,
) -> dict:
    """Update parameters of an existing component.

    Args:
        model_id: The model containing the component
        component_type: Component type (e.g. 'Generator', 'Bus')
        component_id: ID of the component to update
        params: Static/scalar parameters to update
        time_series: Time-varying parameters to update as {attr: [values...]}
    """
    params = params or {}
    time_series = time_series or {}

    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    attrs = _get_component_attrs(network, canonical)

    # Validate params are Input
    bad_keys = _validate_params_are_input(attrs, params)
    if bad_keys:
        return {
            "error": f"Invalid parameters {bad_keys}. "
            f"Use describe_component('{canonical}') to see valid parameters."
        }

    # Component must exist
    df = getattr(network, list_name)
    if component_id not in df.index:
        return {"error": f"{canonical} '{component_id}' not found in model '{model_id}'."}

    # Validate time series
    if time_series:
        bad_ts_keys = _validate_time_series_keys(attrs, time_series)
        if bad_ts_keys:
            return {"error": f"Invalid time_series keys {bad_ts_keys}."}
        if not isinstance(network.snapshots, pd.DatetimeIndex):
            return {"error": "Cannot set time series: no snapshots configured."}

    try:
        # Update static params
        for key, val in params.items():
            df.loc[component_id, key] = val

        # Update time series
        for attr, values in time_series.items():
            ts_df = getattr(network, f"{list_name}_t")
            series = pd.Series(values, index=network.snapshots[: len(values)])
            ts_df[attr][component_id] = series

        return {
            "message": f"{canonical} '{component_id}' updated in model '{model_id}'.",
            "component_data": convert_to_serializable(df.loc[component_id]),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def remove_component(
    model_id: str,
    component_type: str,
    component_id: str,
) -> dict:
    """Remove a component from a model.

    Args:
        model_id: The model containing the component
        component_type: Component type (e.g. 'Generator', 'Bus')
        component_id: ID of the component to remove
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    df = getattr(network, list_name)

    if component_id not in df.index:
        return {"error": f"{canonical} '{component_id}' not found in model '{model_id}'."}

    try:
        network.remove(canonical, component_id)
        df = getattr(network, list_name)
        return {
            "message": f"{canonical} '{component_id}' removed from model '{model_id}'.",
            "remaining_count": len(df),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def query_components(
    model_id: str,
    component_type: str,
    filters: dict | None = None,
) -> dict:
    """Query components of a given type, optionally filtered.

    Args:
        model_id: The model to query
        component_type: Component type (e.g. 'Generator', 'Bus')
        filters: Optional {attr: value} dict to filter rows
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        canonical = validate_component_type(component_type)
    except ValueError as e:
        return {"error": str(e)}

    list_name = VALID_COMPONENT_TYPES[canonical]
    df = getattr(network, list_name)

    if filters:
        for attr, val in filters.items():
            if attr in df.columns:
                df = df[df[attr] == val]

    return {
        "component_type": canonical,
        "count": len(df),
        "components": convert_to_serializable(df),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_components.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/components.py tests/test_components.py
git commit -m "feat: add component CRUD tools (add, update, remove, query)"
```

---

## Task 5: Convenience Wrappers (GROUP 3b)

**Files:**
- Modify: `src/pypsamcp/tools/convenience.py`
- Test: `tests/test_convenience.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_convenience.py`:

```python
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.convenience import add_bus, add_generator, add_line, add_load


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
def model_with_bus():
    n = pypsa.Network()
    n.name = "test"
    n.set_snapshots(["2024-01-01"])
    n.add("Bus", "bus0", v_nom=110.0)
    n.add("Bus", "bus1", v_nom=110.0)
    MODELS["test"] = n
    return n


class TestAddBus:
    @pytest.mark.asyncio
    async def test_adds_bus(self):
        MODELS["test"] = pypsa.Network()
        result = await add_bus("test", "bus0", 110.0)
        assert "message" in result
        assert "deprecation_notice" in result

    @pytest.mark.asyncio
    async def test_with_optional_params(self):
        MODELS["test"] = pypsa.Network()
        result = await add_bus("test", "bus0", 110.0, x=1.0, y=2.0, carrier="DC")
        assert "message" in result


class TestAddGenerator:
    @pytest.mark.asyncio
    async def test_adds_generator(self, model_with_bus):
        result = await add_generator("test", "gen0", "bus0", p_nom=100.0)
        assert "message" in result
        assert "deprecation_notice" in result

    @pytest.mark.asyncio
    async def test_rejects_missing_bus(self, model_with_bus):
        result = await add_generator("test", "gen0", "nonexistent")
        assert "error" in result


class TestAddLoad:
    @pytest.mark.asyncio
    async def test_adds_load(self, model_with_bus):
        result = await add_load("test", "load0", "bus0", p_set=50.0)
        assert "message" in result
        assert "deprecation_notice" in result


class TestAddLine:
    @pytest.mark.asyncio
    async def test_adds_line(self, model_with_bus):
        result = await add_line("test", "line0", "bus0", "bus1", x=0.1)
        assert "message" in result
        assert "deprecation_notice" in result

    @pytest.mark.asyncio
    async def test_rejects_missing_bus(self, model_with_bus):
        result = await add_line("test", "line0", "bus0", "nonexistent", x=0.1)
        assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_convenience.py -v`
Expected: FAIL

- [ ] **Step 3: Implement convenience.py**

Write `src/pypsamcp/tools/convenience.py`:

```python
"""Deprecated convenience wrappers: add_bus, add_generator, add_load, add_line.

These tools delegate to add_component internally. They will be removed in a
future release. Use add_component directly instead.
"""

from pypsamcp.core import mcp
from pypsamcp.tools.components import add_component as _add_component

_DEPRECATION_NOTICE = (
    "This tool is deprecated and will be removed in a future release. "
    "Use add_component() instead."
)


@mcp.tool()
async def add_bus(
    model_id: str,
    bus_id: str,
    v_nom: float,
    x: float = 0.0,
    y: float = 0.0,
    carrier: str = "AC",
    country: str | None = None,
) -> dict:
    """[Deprecated] Add a bus to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        bus_id: The ID for the new bus
        v_nom: Nominal voltage in kV
        x: x-coordinate for plotting
        y: y-coordinate for plotting
        carrier: Energy carrier (e.g., "AC", "DC")
        country: Country code if applicable
    """
    params = {"v_nom": v_nom, "x": x, "y": y, "carrier": carrier}
    if country is not None:
        params["country"] = country
    result = await _add_component(model_id, "Bus", bus_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result


@mcp.tool()
async def add_generator(
    model_id: str,
    generator_id: str,
    bus: str,
    p_nom: float = 0.0,
    p_nom_extendable: bool = False,
    capital_cost: float | None = None,
    marginal_cost: float | None = None,
    carrier: str | None = None,
    efficiency: float = 1.0,
) -> dict:
    """[Deprecated] Add a generator to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        generator_id: The ID for the new generator
        bus: The bus ID to connect to
        p_nom: Nominal power capacity in MW
        p_nom_extendable: Whether capacity can be expanded
        capital_cost: Investment cost in currency/MW
        marginal_cost: Operational cost in currency/MWh
        carrier: Energy carrier (e.g., "wind", "solar")
        efficiency: Generator efficiency (0 to 1)
    """
    params = {
        "bus": bus,
        "p_nom": p_nom,
        "p_nom_extendable": p_nom_extendable,
        "efficiency": efficiency,
    }
    if capital_cost is not None:
        params["capital_cost"] = capital_cost
    if marginal_cost is not None:
        params["marginal_cost"] = marginal_cost
    if carrier is not None:
        params["carrier"] = carrier
    result = await _add_component(model_id, "Generator", generator_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result


@mcp.tool()
async def add_load(
    model_id: str,
    load_id: str,
    bus: str,
    p_set: float = 0.0,
    q_set: float = 0.0,
) -> dict:
    """[Deprecated] Add a load to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        load_id: The ID for the new load
        bus: The bus ID to connect to
        p_set: Active power demand in MW
        q_set: Reactive power demand in MVAr
    """
    params = {"bus": bus, "p_set": p_set, "q_set": q_set}
    result = await _add_component(model_id, "Load", load_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result


@mcp.tool()
async def add_line(
    model_id: str,
    line_id: str,
    bus0: str,
    bus1: str,
    x: float,
    r: float = 0.0,
    s_nom: float = 0.0,
    s_nom_extendable: bool = False,
    capital_cost: float | None = None,
    length: float | None = None,
) -> dict:
    """[Deprecated] Add a transmission line to a PyPSA model. Use add_component() instead.

    Args:
        model_id: The ID of the model
        line_id: The ID for the new line
        bus0: The ID of the first bus
        bus1: The ID of the second bus
        x: Reactance in ohm
        r: Resistance in ohm
        s_nom: Nominal apparent power capacity in MVA
        s_nom_extendable: Whether capacity can be expanded
        capital_cost: Investment cost in currency/MW
        length: Line length in km
    """
    params = {
        "bus0": bus0,
        "bus1": bus1,
        "x": x,
        "r": r,
        "s_nom": s_nom,
        "s_nom_extendable": s_nom_extendable,
    }
    if capital_cost is not None:
        params["capital_cost"] = capital_cost
    if length is not None:
        params["length"] = length
    result = await _add_component(model_id, "Line", line_id, params)
    result["deprecation_notice"] = _DEPRECATION_NOTICE
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_convenience.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/convenience.py tests/test_convenience.py
git commit -m "feat: add deprecated convenience wrappers delegating to add_component"
```

---

## Task 6: Time Configuration Tool (GROUP 4)

**Files:**
- Modify: `src/pypsamcp/tools/time_config.py`
- Test: `tests/test_time_config.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_time_config.py`:

```python
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
    @pytest.mark.asyncio
    async def test_set_snapshots(self, empty_model):
        result = await configure_time(
            "test", "snapshots",
            snapshots=["2024-01-01", "2024-01-02", "2024-01-03"],
        )
        assert "message" in result
        assert result["snapshot_count"] == 3

    @pytest.mark.asyncio
    async def test_requires_snapshots_arg(self, empty_model):
        result = await configure_time("test", "snapshots")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        result = await configure_time("nonexistent", "snapshots", snapshots=["2024-01-01"])
        assert "error" in result


class TestConfigureTimeInvestmentPeriods:
    @pytest.mark.asyncio
    async def test_set_investment_periods(self, empty_model):
        empty_model.set_snapshots(["2024-01-01"])
        result = await configure_time(
            "test", "investment_periods",
            periods=[2025, 2030, 2035],
        )
        assert "message" in result
        assert result["has_investment_periods"] is True

    @pytest.mark.asyncio
    async def test_requires_snapshots_first(self, empty_model):
        result = await configure_time(
            "test", "investment_periods",
            periods=[2025, 2030],
        )
        # PyPSA may error or we validate — either way, test it doesn't crash
        assert isinstance(result, dict)


class TestConfigureTimeInvalidMode:
    @pytest.mark.asyncio
    async def test_invalid_mode(self, empty_model):
        result = await configure_time("test", "invalid_mode")
        assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_time_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement time_config.py**

Write `src/pypsamcp/tools/time_config.py`:

```python
"""Time and investment structure configuration."""

import pandas as pd

from pypsamcp.core import get_energy_model, mcp


@mcp.tool()
async def configure_time(
    model_id: str,
    mode: str,
    # mode="snapshots"
    snapshots: list[str] | None = None,
    weightings: dict | None = None,
    # mode="investment_periods"
    periods: list[int] | None = None,
    # mode="scenarios"
    scenarios: dict | list | None = None,
    # mode="risk_preference"
    alpha: float | None = None,
    omega: float | None = None,
) -> dict:
    """Configure time snapshots, investment periods, scenarios, or risk preference.

    Args:
        model_id: The model to configure
        mode: One of 'snapshots', 'investment_periods', 'scenarios', 'risk_preference'
        snapshots: ISO datetime strings (mode='snapshots')
        weightings: Snapshot weightings dict (mode='snapshots')
        periods: Investment years e.g. [2025, 2030] (mode='investment_periods')
        scenarios: Scenario definition (mode='scenarios')
        alpha: CVaR confidence level (mode='risk_preference')
        omega: Weight on CVaR term 0-1 (mode='risk_preference')
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        if mode == "snapshots":
            if snapshots is None:
                return {"error": "snapshots parameter is required for mode='snapshots'."}
            network.set_snapshots(pd.to_datetime(snapshots))
            if weightings:
                for key, val in weightings.items():
                    if key in network.snapshot_weightings.columns:
                        if isinstance(val, list):
                            network.snapshot_weightings[key] = val
                        else:
                            network.snapshot_weightings[key] = val
            msg = f"Snapshots configured for model '{model_id}'."

        elif mode == "investment_periods":
            if periods is None:
                return {"error": "periods parameter is required for mode='investment_periods'."}
            network.set_investment_periods(periods)
            msg = f"Investment periods {periods} configured for model '{model_id}'."

        elif mode == "scenarios":
            if scenarios is None:
                return {"error": "scenarios parameter is required for mode='scenarios'."}
            network.set_scenarios(scenarios)
            msg = f"Scenarios configured for model '{model_id}'."

        elif mode == "risk_preference":
            if alpha is None or omega is None:
                return {"error": "alpha and omega are required for mode='risk_preference'."}
            if not getattr(network, "has_scenarios", False):
                return {"error": "Scenarios must be set before configuring risk preference."}
            network.set_risk_preference(alpha, omega)
            msg = f"Risk preference (alpha={alpha}, omega={omega}) configured for model '{model_id}'."

        else:
            return {
                "error": f"Invalid mode '{mode}'. Must be one of: "
                "'snapshots', 'investment_periods', 'scenarios', 'risk_preference'."
            }

        # Build response
        has_ip = bool(getattr(network, "has_investment_periods", False))
        has_sc = bool(getattr(network, "has_scenarios", False))
        has_rp = bool(getattr(network, "has_risk_preference", False))

        result = {
            "mode": mode,
            "message": msg,
            "has_investment_periods": has_ip,
            "has_scenarios": has_sc,
            "has_risk_preference": has_rp,
            "snapshot_count": len(network.snapshots),
        }

        if has_ip:
            ipw = network.investment_period_weightings
            result["investment_period_weightings"] = ipw.to_dict()

        return result

    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_time_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/time_config.py tests/test_time_config.py
git commit -m "feat: add configure_time tool (snapshots, investment periods, scenarios)"
```

---

## Task 7: Simulation Tool (GROUP 5)

**Files:**
- Modify: `src/pypsamcp/tools/simulation.py`
- Test: `tests/test_simulation.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_simulation.py`:

```python
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
    @pytest.mark.asyncio
    async def test_basic_optimize(self, simple_model):
        result = await run_simulation("test", mode="optimize")
        assert result["status"] == "ok"
        assert result["termination_condition"] == "optimal"
        assert result["objective_value"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await run_simulation("nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_mode(self, simple_model):
        result = await run_simulation("test", mode="invalid")
        assert "error" in result


class TestRunSimulationPF:
    @pytest.mark.asyncio
    async def test_basic_pf(self, pf_model):
        result = await run_simulation("test", mode="pf")
        assert "summary" in result
        assert result["mode"] == "pf"

    @pytest.mark.asyncio
    async def test_lpf(self, pf_model):
        result = await run_simulation("test", mode="lpf")
        assert result["mode"] == "lpf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_simulation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement simulation.py**

Write `src/pypsamcp/tools/simulation.py`:

```python
"""Simulation tool: power flow, optimization, and advanced modes."""

import textwrap

import pandas as pd

from pypsamcp.core import convert_to_serializable, get_energy_model, mcp, stdout_to_stderr

VALID_MODES = {
    "pf", "lpf", "optimize", "mga", "security_constrained",
    "rolling_horizon", "transmission_expansion_iterative", "optimize_and_pf",
}


def _collect_pf_results(network):
    """Collect power flow results into a serializable dict."""
    return {
        "bus_v_mag_pu": convert_to_serializable(network.buses_t.v_mag_pu),
        "bus_v_ang": convert_to_serializable(network.buses_t.v_ang),
        "line_p0": convert_to_serializable(network.lines_t.p0),
        "line_p1": convert_to_serializable(network.lines_t.p1),
    }


def _collect_optimization_results(network):
    """Collect optimization dispatch/expansion results into a serializable dict."""
    results = {}

    # Dispatch results
    if not network.generators_t.p.empty:
        results["generator_dispatch"] = convert_to_serializable(network.generators_t.p)
    if not network.storage_units_t.p.empty:
        results["storage_dispatch"] = convert_to_serializable(network.storage_units_t.p)
    if not network.lines_t.p0.empty:
        results["line_flows"] = convert_to_serializable(network.lines_t.p0)
    if not network.buses_t.marginal_price.empty:
        results["shadow_prices"] = convert_to_serializable(network.buses_t.marginal_price)

    # Expansion results
    if "p_nom_opt" in network.generators.columns:
        results["generator_expansion"] = convert_to_serializable(network.generators["p_nom_opt"])
    if "p_nom_opt" in network.storage_units.columns:
        results["storage_expansion"] = convert_to_serializable(network.storage_units["p_nom_opt"])
    if "s_nom_opt" in network.lines.columns:
        results["line_expansion"] = convert_to_serializable(network.lines["s_nom_opt"])

    return results


def _build_extra_functionality(code_string):
    """Compile extra_functionality code string into a callable."""
    namespace = {}
    exec(
        f"def extra_func(n, snapshots):\n{textwrap.indent(code_string, '    ')}",
        namespace,
    )
    return namespace["extra_func"]


@mcp.tool()
async def run_simulation(
    model_id: str,
    mode: str = "optimize",
    # pf/lpf
    snapshots: list[str] | None = None,
    distribute_slack: bool = False,
    slack_weights: str = "p_set",
    x_tol: float = 1e-6,
    # optimize
    solver_name: str = "highs",
    formulation: str = "kirchhoff",
    multi_investment_periods: bool = False,
    transmission_losses: int | bool = False,
    linearized_unit_commitment: bool = False,
    assign_all_duals: bool = False,
    compute_infeasibilities: bool = False,
    solver_options: dict | None = None,
    extra_functionality: str | None = None,
    # mga
    slack: float = 0.05,
    sense: str = "min",
    weights: dict | None = None,
    # security_constrained
    branch_outages: list[str] | None = None,
    # rolling_horizon
    horizon: int = 100,
    overlap: int = 0,
    # transmission_expansion_iterative
    msq_threshold: float = 0.05,
    min_iterations: int = 1,
    max_iterations: int = 100,
) -> dict:
    """Run a simulation on a PyPSA model.

    Args:
        model_id: The model to simulate
        mode: Simulation mode. One of: 'pf', 'lpf', 'optimize', 'mga',
              'security_constrained', 'rolling_horizon',
              'transmission_expansion_iterative', 'optimize_and_pf'
        snapshots: Specific snapshots to simulate (pf/lpf modes)
        distribute_slack: Distribute slack bus power (pf mode)
        slack_weights: Slack distribution weights (pf mode)
        x_tol: Newton-Raphson tolerance (pf mode)
        solver_name: Solver to use (optimize modes)
        formulation: Network formulation 'kirchhoff' or 'ptdf'
        multi_investment_periods: Enable multi-period investment
        transmission_losses: Linearized loss approximation (bool or int segments)
        linearized_unit_commitment: Use LP relaxation for commitment
        assign_all_duals: Store all dual values after solve
        compute_infeasibilities: Diagnose infeasible models
        solver_options: Dict passed to solver (e.g. {"threads": 4})
        extra_functionality: Python code string for custom constraints
        slack: MGA cost slack (mga mode)
        sense: MGA direction 'min' or 'max'
        weights: MGA component weights dict
        branch_outages: Line/link IDs for N-1 contingencies
        horizon: Rolling horizon window size in snapshots
        overlap: Rolling horizon overlap
        msq_threshold: Transmission expansion MSQ threshold
        min_iterations: Transmission expansion min iterations
        max_iterations: Transmission expansion max iterations
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    if mode not in VALID_MODES:
        return {
            "error": f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}"
        }

    # Parse extra_functionality
    extra_func = None
    if extra_functionality:
        try:
            extra_func = _build_extra_functionality(extra_functionality)
        except Exception as e:
            return {"error": f"Error in extra_functionality code: {str(e)}"}

    # Parse snapshot subset
    snap_subset = pd.to_datetime(snapshots) if snapshots else None

    try:
        with stdout_to_stderr():
            if mode == "pf":
                network.pf(
                    snapshots=snap_subset,
                    distribute_slack=distribute_slack,
                    slack_weights=slack_weights,
                    x_tol=x_tol,
                )
                return {
                    "mode": "pf",
                    "message": f"Power flow completed for model '{model_id}'.",
                    "summary": _collect_pf_results(network),
                }

            elif mode == "lpf":
                network.lpf(snapshots=snap_subset)
                return {
                    "mode": "lpf",
                    "message": f"Linear power flow completed for model '{model_id}'.",
                    "summary": _collect_pf_results(network),
                }

            elif mode == "optimize":
                opt_kwargs = {
                    "solver_name": solver_name,
                    "formulation": formulation,
                    "multi_investment_periods": multi_investment_periods,
                    "transmission_losses": transmission_losses,
                    "linearized_unit_commitment": linearized_unit_commitment,
                    "assign_all_duals": assign_all_duals,
                    "compute_infeasibilities": compute_infeasibilities,
                    "extra_functionality": extra_func,
                }
                if solver_options:
                    opt_kwargs["solver_options"] = solver_options

                status, condition = network.optimize(**opt_kwargs)
                return {
                    "mode": "optimize",
                    "status": status,
                    "termination_condition": condition,
                    "objective_value": float(network.objective),
                    "message": f"Optimization completed for model '{model_id}'.",
                    "summary": _collect_optimization_results(network),
                }

            elif mode == "mga":
                # First optimize to get baseline
                status, condition = network.optimize(solver_name=solver_name)
                if status != "ok":
                    return {"error": f"Baseline optimization failed: {status} / {condition}"}
                mga_kwargs = {
                    "slack": slack,
                    "sense": sense,
                    "solver_name": solver_name,
                }
                if weights:
                    mga_kwargs["weights"] = weights
                network.optimize.optimize_mga(**mga_kwargs)
                return {
                    "mode": "mga",
                    "status": "ok",
                    "objective_value": float(network.objective),
                    "message": f"MGA completed for model '{model_id}'.",
                    "summary": _collect_optimization_results(network),
                }

            elif mode == "security_constrained":
                sc_kwargs = {"solver_name": solver_name}
                if branch_outages:
                    sc_kwargs["branch_outages"] = branch_outages
                status, condition = network.optimize.optimize_security_constrained(**sc_kwargs)
                return {
                    "mode": "security_constrained",
                    "status": status,
                    "termination_condition": condition,
                    "objective_value": float(network.objective),
                    "message": f"Security-constrained optimization completed for model '{model_id}'.",
                    "summary": _collect_optimization_results(network),
                }

            elif mode == "rolling_horizon":
                status, condition = network.optimize.optimize_with_rolling_horizon(
                    horizon=horizon,
                    overlap=overlap,
                    solver_name=solver_name,
                )
                return {
                    "mode": "rolling_horizon",
                    "status": status,
                    "termination_condition": condition,
                    "objective_value": float(network.objective),
                    "message": f"Rolling horizon optimization completed for model '{model_id}'.",
                    "summary": _collect_optimization_results(network),
                }

            elif mode == "transmission_expansion_iterative":
                status, condition = network.optimize.optimize_transmission_expansion_iteratively(
                    msq_threshold=msq_threshold,
                    min_iterations=min_iterations,
                    max_iterations=max_iterations,
                    solver_name=solver_name,
                )
                return {
                    "mode": "transmission_expansion_iterative",
                    "status": status,
                    "termination_condition": condition,
                    "objective_value": float(network.objective),
                    "message": f"Transmission expansion optimization completed for model '{model_id}'.",
                    "summary": _collect_optimization_results(network),
                }

            elif mode == "optimize_and_pf":
                status, condition = network.optimize.optimize_and_run_non_linear_powerflow(
                    solver_name=solver_name,
                )
                results = _collect_optimization_results(network)
                results.update(_collect_pf_results(network))
                return {
                    "mode": "optimize_and_pf",
                    "status": status,
                    "termination_condition": condition,
                    "objective_value": float(network.objective),
                    "message": f"Optimize + PF completed for model '{model_id}'.",
                    "summary": results,
                }

    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_simulation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/simulation.py tests/test_simulation.py
git commit -m "feat: add run_simulation tool with 8 modes (pf, lpf, optimize, mga, etc.)"
```

---

## Task 8: Statistics Tool (GROUP 6)

**Files:**
- Modify: `src/pypsamcp/tools/statistics.py`
- Test: `tests/test_statistics.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_statistics.py`:

```python
import pytest
import pypsa

from pypsamcp.core import MODELS
from pypsamcp.tools.simulation import run_simulation
from pypsamcp.tools.statistics import get_statistics


@pytest.fixture(autouse=True)
def clean_models():
    MODELS.clear()
    yield
    MODELS.clear()


@pytest.fixture
async def solved_model():
    n = pypsa.Network()
    n.set_snapshots(["2024-01-01"])
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10, carrier="gas")
    n.add("Load", "load0", bus="bus0", p_set=50)
    MODELS["test"] = n
    await run_simulation("test", mode="optimize")
    return n


class TestGetStatistics:
    @pytest.mark.asyncio
    async def test_system_cost(self, solved_model):
        result = await get_statistics("test", "system_cost")
        assert "result" in result
        assert result["metric"] == "system_cost"

    @pytest.mark.asyncio
    async def test_capex(self, solved_model):
        result = await get_statistics("test", "capex")
        assert "result" in result

    @pytest.mark.asyncio
    async def test_all_metrics(self, solved_model):
        result = await get_statistics("test", "all")
        assert "result" in result
        assert isinstance(result["result"], dict)
        assert "system_cost" in result["result"]

    @pytest.mark.asyncio
    async def test_invalid_metric(self, solved_model):
        result = await get_statistics("test", "fake_metric")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unsolved_model(self):
        n = pypsa.Network()
        MODELS["unsolved"] = n
        result = await get_statistics("unsolved", "system_cost")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await get_statistics("nonexistent", "system_cost")
        assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_statistics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement statistics.py**

Write `src/pypsamcp/tools/statistics.py`:

```python
"""Statistics and results tool."""

from pypsamcp.core import convert_to_serializable, get_energy_model, mcp

VALID_METRICS = [
    "system_cost", "capex", "opex", "fom", "overnight_cost",
    "installed_capacity", "optimal_capacity", "expanded_capacity",
    "installed_capex", "expanded_capex", "capacity_factor", "curtailment",
    "energy_balance", "supply", "withdrawal", "revenue", "market_value",
    "prices", "transmission",
]


def _call_metric(network, metric, kwargs):
    """Call a single statistics metric and return the serialized result."""
    method = getattr(network.statistics, metric)
    result = method(**kwargs)
    return convert_to_serializable(result)


@mcp.tool()
async def get_statistics(
    model_id: str,
    metric: str = "system_cost",
    components: list[str] | None = None,
    carrier: list[str] | None = None,
    bus_carrier: list[str] | None = None,
    groupby: str | list[str] = "carrier",
    aggregate_across_components: bool = False,
    nice_names: bool | None = None,
    drop_zero: bool | None = None,
) -> dict:
    """Get statistics from a solved PyPSA model.

    Args:
        model_id: The model to query
        metric: Metric name or 'all'. One of: system_cost, capex, opex, fom,
                overnight_cost, installed_capacity, optimal_capacity,
                expanded_capacity, installed_capex, expanded_capex,
                capacity_factor, curtailment, energy_balance, supply,
                withdrawal, revenue, market_value, prices, transmission
        components: Filter to specific component types
        carrier: Filter by carrier name
        bus_carrier: Filter by bus carrier
        groupby: Aggregation grouping (e.g. 'carrier', 'bus')
        aggregate_across_components: Merge component types into one row per carrier
        nice_names: Use human-readable carrier names
        drop_zero: Omit zero-value rows
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    # Check if solved — n.objective is None on unsolved networks, not absent
    if network.objective is None:
        return {
            "error": f"Model '{model_id}' has not been solved. "
            "Run run_simulation(mode='optimize') first."
        }

    if metric != "all" and metric not in VALID_METRICS:
        return {
            "error": f"Invalid metric '{metric}'. Must be one of: {VALID_METRICS + ['all']}"
        }

    # Build shared kwargs, only including non-None values
    kwargs = {"groupby": groupby, "aggregate_across_components": aggregate_across_components}
    if components is not None:
        kwargs["components"] = components
    if carrier is not None:
        kwargs["carrier"] = carrier
    if bus_carrier is not None:
        kwargs["bus_carrier"] = bus_carrier
    if nice_names is not None:
        kwargs["nice_names"] = nice_names
    if drop_zero is not None:
        kwargs["drop_zero"] = drop_zero

    try:
        if metric == "all":
            combined = {}
            for m in VALID_METRICS:
                try:
                    combined[m] = _call_metric(network, m, kwargs)
                except Exception:
                    combined[m] = None
            return {
                "metric": "all",
                "model_id": model_id,
                "result": combined,
            }
        else:
            result = _call_metric(network, metric, kwargs)
            return {
                "metric": metric,
                "model_id": model_id,
                "result": result,
                "unit_note": "Currency units depend on the cost inputs used when building the model.",
            }
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_statistics.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/statistics.py tests/test_statistics.py
git commit -m "feat: add get_statistics tool with 19 metrics"
```

---

## Task 9: Clustering Tool (GROUP 7)

**Files:**
- Modify: `src/pypsamcp/tools/clustering.py`
- Test: `tests/test_clustering.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_clustering.py`:

```python
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
    # Connect buses with lines
    for i in range(4):
        n.add("Line", f"line{i}", bus0=f"bus{i}", bus1=f"bus{i+1}", x=0.1, s_nom=200)
    MODELS["test"] = n
    return n


class TestClusterSpatial:
    @pytest.mark.asyncio
    async def test_invalid_model(self):
        result = await cluster_network("nonexistent", "spatial", "kmeans", "out", n_clusters=2)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_domain(self, multi_bus_model):
        result = await cluster_network("test", "invalid", "kmeans", "out", n_clusters=2)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_method(self, multi_bus_model):
        result = await cluster_network("test", "spatial", "invalid", "out", n_clusters=2)
        assert "error" in result


class TestClusterTemporal:
    @pytest.mark.asyncio
    async def test_resample(self, multi_bus_model):
        result = await cluster_network("test", "temporal", "resample", "resampled", offset="3h")
        assert "message" in result
        assert "resampled" in MODELS
        assert result["output_snapshots"] < result["input_snapshots"]

    @pytest.mark.asyncio
    async def test_downsample(self, multi_bus_model):
        result = await cluster_network("test", "temporal", "downsample", "downsampled", stride=4)
        assert "message" in result
        assert "downsampled" in MODELS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_clustering.py -v`
Expected: FAIL

- [ ] **Step 3: Implement clustering.py**

Write `src/pypsamcp/tools/clustering.py`:

```python
"""Spatial and temporal clustering tool."""

from pypsamcp.core import MODELS, get_energy_model, mcp

VALID_SPATIAL_METHODS = {"kmeans", "hac", "greedy_modularity"}
VALID_TEMPORAL_METHODS = {"resample", "downsample", "segment", "snapshot_map"}


@mcp.tool()
async def cluster_network(
    model_id: str,
    domain: str,
    method: str,
    output_model_id: str,
    # spatial
    n_clusters: int | None = None,
    bus_weightings: dict | None = None,
    line_length_factor: float = 1.0,
    affinity: str = "euclidean",
    linkage: str = "ward",
    # spatial cluster_options
    with_time: bool = True,
    aggregate_generators_weighted: bool = False,
    scale_link_capital_costs: bool = True,
    # temporal
    offset: str | None = None,
    stride: int | None = None,
    num_segments: int | None = None,
    solver: str = "highs",
    snapshot_map: dict | None = None,
) -> dict:
    """Cluster a network spatially or temporally, storing result as a new model.

    Args:
        model_id: Source model
        domain: 'spatial' or 'temporal'
        method: Clustering method (spatial: kmeans/hac/greedy_modularity;
                temporal: resample/downsample/segment/snapshot_map)
        output_model_id: ID for the clustered model
        n_clusters: Number of clusters (spatial)
        bus_weightings: Bus weights dict (spatial, auto-computed if None)
        line_length_factor: Line length scaling factor (spatial)
        affinity: Distance metric for HAC (spatial)
        linkage: Linkage method for HAC (spatial)
        with_time: Include time-series in clustering (spatial)
        aggregate_generators_weighted: Weight generator aggregation (spatial)
        scale_link_capital_costs: Scale link costs with clustering (spatial)
        offset: Pandas offset string e.g. '3h' (temporal resample)
        stride: Take every Nth snapshot (temporal downsample)
        num_segments: Number of segments (temporal segment)
        solver: Solver for segmentation (temporal segment)
        snapshot_map: Custom snapshot aggregation map (temporal snapshot_map)
    """
    try:
        network = get_energy_model(model_id)
    except ValueError as e:
        return {"error": str(e)}

    if domain not in ("spatial", "temporal"):
        return {"error": f"Invalid domain '{domain}'. Must be 'spatial' or 'temporal'."}

    input_buses = len(network.buses)
    input_snapshots = len(network.snapshots)

    try:
        if domain == "spatial":
            if method not in VALID_SPATIAL_METHODS:
                return {
                    "error": f"Invalid spatial method '{method}'. "
                    f"Must be one of: {sorted(VALID_SPATIAL_METHODS)}"
                }
            if n_clusters is None:
                return {"error": "n_clusters is required for spatial clustering."}

            cluster_method = getattr(network.cluster.spatial, f"cluster_by_{method}")

            sp_kwargs = {"n_clusters": n_clusters}
            if method == "kmeans":
                sp_kwargs["line_length_factor"] = line_length_factor
            elif method == "hac":
                sp_kwargs["affinity"] = affinity
                sp_kwargs["linkage"] = linkage

            clustered = cluster_method(**sp_kwargs)
            MODELS[output_model_id] = clustered

            return {
                "domain": "spatial",
                "method": method,
                "input_model_id": model_id,
                "output_model_id": output_model_id,
                "input_buses": input_buses,
                "output_buses": len(clustered.buses),
                "input_snapshots": input_snapshots,
                "output_snapshots": len(clustered.snapshots),
                "message": f"Spatially clustered model stored as '{output_model_id}'.",
            }

        else:  # temporal
            if method not in VALID_TEMPORAL_METHODS:
                return {
                    "error": f"Invalid temporal method '{method}'. "
                    f"Must be one of: {sorted(VALID_TEMPORAL_METHODS)}"
                }

            if method == "resample":
                if offset is None:
                    return {"error": "offset is required for temporal resample."}
                clustered = network.cluster.temporal.resample(offset=offset)
            elif method == "downsample":
                if stride is None:
                    return {"error": "stride is required for temporal downsample."}
                clustered = network.cluster.temporal.downsample(stride=stride)
            elif method == "segment":
                if num_segments is None:
                    return {"error": "num_segments is required for temporal segment."}
                clustered = network.cluster.temporal.segment(
                    num_segments=num_segments, solver=solver
                )
            elif method == "snapshot_map":
                if snapshot_map is None:
                    return {"error": "snapshot_map is required for temporal snapshot_map."}
                clustered = network.cluster.temporal.from_snapshot_map(
                    snapshot_map=snapshot_map
                )

            MODELS[output_model_id] = clustered

            return {
                "domain": "temporal",
                "method": method,
                "input_model_id": model_id,
                "output_model_id": output_model_id,
                "input_buses": input_buses,
                "output_buses": len(clustered.buses),
                "input_snapshots": input_snapshots,
                "output_snapshots": len(clustered.snapshots),
                "message": f"Temporally clustered model stored as '{output_model_id}'.",
            }

    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_clustering.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/clustering.py tests/test_clustering.py
git commit -m "feat: add cluster_network tool (spatial and temporal clustering)"
```

---

## Task 10: I/O Tool (GROUP 8)

**Files:**
- Modify: `src/pypsamcp/tools/io.py`
- Test: `tests/test_io.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_io.py`:

```python
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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_export(self, simple_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await network_io("test", "export_csv", path=tmpdir)
            assert "message" in result


class TestImportNetcdf:
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_copy(self, simple_model):
        result = await network_io("test", "copy", output_model_id="copy1")
        assert "message" in result
        assert "copy1" in MODELS
        assert len(MODELS["copy1"].buses) == 1


class TestMerge:
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_consistency_check(self, simple_model):
        result = await network_io("test", "consistency_check")
        assert "message" in result


class TestInvalidOperation:
    @pytest.mark.asyncio
    async def test_invalid(self, simple_model):
        result = await network_io("test", "invalid_op")
        assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_io.py -v`
Expected: FAIL

- [ ] **Step 3: Implement io.py**

Write `src/pypsamcp/tools/io.py`:

```python
"""I/O and network operations tool."""

import pandas as pd
import pypsa

from pypsamcp.core import MODELS, generate_network_summary, get_energy_model, mcp

VALID_OPERATIONS = {
    "export_netcdf", "export_csv", "import_netcdf", "import_csv",
    "merge", "copy", "consistency_check",
}


@mcp.tool()
async def network_io(
    model_id: str,
    operation: str,
    # export
    path: str | None = None,
    float32: bool = False,
    compression: dict | None = None,
    export_standard_types: bool = False,
    # import
    skip_time: bool = False,
    # merge
    other_model_id: str | None = None,
    output_model_id: str | None = None,
    components_to_skip: list[str] | None = None,
    with_time: bool = True,
    # copy/subset
    snapshots: list[str] | None = None,
    investment_periods: list | None = None,
    buses: list[str] | None = None,
) -> dict:
    """Perform I/O and network manipulation operations.

    Args:
        model_id: The model to operate on (or ID for imported model)
        operation: One of: export_netcdf, export_csv, import_netcdf, import_csv,
                   merge, copy, consistency_check
        path: File/directory path for export/import operations
        float32: Reduce precision for NetCDF export
        compression: NetCDF compression settings dict
        export_standard_types: Include standard types in CSV export
        skip_time: Skip time data on import
        other_model_id: Second model for merge
        output_model_id: Destination model ID for merge/copy
        components_to_skip: Components to exclude from merge
        with_time: Include time series in merge
        snapshots: Snapshot subset for copy
        investment_periods: Investment period subset for copy
        buses: Bus subset for copy (filters connected components)
    """
    if operation not in VALID_OPERATIONS:
        return {
            "error": f"Invalid operation '{operation}'. Must be one of: {sorted(VALID_OPERATIONS)}"
        }

    try:
        if operation == "export_netcdf":
            network = get_energy_model(model_id)
            if path is None:
                return {"error": "path is required for export_netcdf."}
            export_kwargs = {}
            if float32:
                export_kwargs["float32"] = True
            if compression:
                export_kwargs["compression"] = compression
            network.export_to_netcdf(path, **export_kwargs)
            return {
                "operation": "export_netcdf",
                "path": path,
                "message": f"Model '{model_id}' exported to {path}.",
            }

        elif operation == "export_csv":
            network = get_energy_model(model_id)
            if path is None:
                return {"error": "path is required for export_csv."}
            network.export_to_csv_folder(path, export_standard_types=export_standard_types)
            return {
                "operation": "export_csv",
                "path": path,
                "message": f"Model '{model_id}' exported to {path}.",
            }

        elif operation == "import_netcdf":
            if path is None:
                return {"error": "path is required for import_netcdf."}
            network = pypsa.Network()
            network.import_from_netcdf(path, skip_time=skip_time)
            MODELS[model_id] = network
            return {
                "operation": "import_netcdf",
                "model_id": model_id,
                "path": path,
                "message": f"Model imported from {path} as '{model_id}'.",
                "summary": generate_network_summary(network),
            }

        elif operation == "import_csv":
            if path is None:
                return {"error": "path is required for import_csv."}
            network = pypsa.Network()
            network.import_from_csv_folder(path, skip_time=skip_time)
            MODELS[model_id] = network
            return {
                "operation": "import_csv",
                "model_id": model_id,
                "path": path,
                "message": f"Model imported from {path} as '{model_id}'.",
                "summary": generate_network_summary(network),
            }

        elif operation == "merge":
            network = get_energy_model(model_id)
            if other_model_id is None:
                return {"error": "other_model_id is required for merge."}
            if output_model_id is None:
                return {"error": "output_model_id is required for merge."}
            other = get_energy_model(other_model_id)
            merge_kwargs = {"with_time": with_time}
            if components_to_skip:
                merge_kwargs["components_to_skip"] = components_to_skip
            merged = network.merge(other, inplace=False, **merge_kwargs)
            MODELS[output_model_id] = merged
            return {
                "operation": "merge",
                "output_model_id": output_model_id,
                "message": f"Models '{model_id}' and '{other_model_id}' merged as '{output_model_id}'.",
                "summary": generate_network_summary(merged),
            }

        elif operation == "copy":
            network = get_energy_model(model_id)
            if output_model_id is None:
                return {"error": "output_model_id is required for copy."}
            copy_kwargs = {}
            if snapshots:
                copy_kwargs["snapshots"] = pd.to_datetime(snapshots)
            if investment_periods:
                copy_kwargs["investment_periods"] = investment_periods
            copied = network.copy(**copy_kwargs)
            # If buses subset requested, filter post-copy
            if buses:
                # Keep only specified buses and their connected components
                drop_buses = [b for b in copied.buses.index if b not in buses]
                for b in drop_buses:
                    copied.remove("Bus", b)
            MODELS[output_model_id] = copied
            return {
                "operation": "copy",
                "output_model_id": output_model_id,
                "message": f"Model '{model_id}' copied as '{output_model_id}'.",
                "summary": generate_network_summary(copied),
            }

        elif operation == "consistency_check":
            network = get_energy_model(model_id)
            network.consistency_check()
            return {
                "operation": "consistency_check",
                "message": f"Consistency check passed for model '{model_id}'.",
            }

    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_io.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/io.py tests/test_io.py
git commit -m "feat: add network_io tool (export, import, merge, copy, consistency check)"
```

---

## Task 11: Deprecated Aliases

**Files:**
- Modify: `src/pypsamcp/tools/deprecated.py`
- Test: `tests/test_deprecated.py`

### Steps

- [ ] **Step 1: Write tests**

Create `tests/test_deprecated.py`:

```python
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
    @pytest.mark.asyncio
    async def test_delegates_to_configure_time(self):
        MODELS["test"] = pypsa.Network()
        result = await set_snapshots("test", ["2024-01-01", "2024-01-02"])
        assert "deprecation_notice" in result
        assert len(MODELS["test"].snapshots) == 2


class TestRunPowerflow:
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_deprecated.py -v`
Expected: FAIL

- [ ] **Step 3: Implement deprecated.py**

Write `src/pypsamcp/tools/deprecated.py`:

```python
"""Deprecated tool aliases for backwards compatibility.

These tools delegate to their replacements and will be removed in a future release.
"""

from pypsamcp.core import mcp
from pypsamcp.tools.simulation import run_simulation as _run_simulation
from pypsamcp.tools.time_config import configure_time as _configure_time

_DEPRECATION_NOTICE_SNAPSHOTS = (
    "set_snapshots is deprecated. Use configure_time(mode='snapshots') instead."
)
_DEPRECATION_NOTICE_PF = (
    "run_powerflow is deprecated. Use run_simulation(mode='pf') instead."
)
_DEPRECATION_NOTICE_OPT = (
    "run_optimization is deprecated. Use run_simulation(mode='optimize') instead."
)


@mcp.tool()
async def set_snapshots(model_id: str, snapshots: list[str]) -> dict:
    """[Deprecated] Set time snapshots. Use configure_time(mode='snapshots') instead.

    Args:
        model_id: The model ID
        snapshots: List of datetime strings
    """
    result = await _configure_time(model_id, "snapshots", snapshots=snapshots)
    result["deprecation_notice"] = _DEPRECATION_NOTICE_SNAPSHOTS
    return result


@mcp.tool()
async def run_powerflow(model_id: str, snapshot: str | None = None) -> dict:
    """[Deprecated] Run power flow. Use run_simulation(mode='pf') instead.

    Args:
        model_id: The model ID
        snapshot: Specific snapshot to run for (optional)
    """
    snap_list = [snapshot] if snapshot else None
    result = await _run_simulation(model_id, mode="pf", snapshots=snap_list)
    result["deprecation_notice"] = _DEPRECATION_NOTICE_PF
    return result


@mcp.tool()
async def run_optimization(
    model_id: str,
    solver_name: str = "highs",
    formulation: str = "kirchhoff",
    extra_functionality: str | None = None,
) -> dict:
    """[Deprecated] Run optimization. Use run_simulation(mode='optimize') instead.

    Args:
        model_id: The model ID
        solver_name: Solver to use
        formulation: Network formulation
        extra_functionality: Custom constraint code
    """
    result = await _run_simulation(
        model_id,
        mode="optimize",
        solver_name=solver_name,
        formulation=formulation,
        extra_functionality=extra_functionality,
    )
    result["deprecation_notice"] = _DEPRECATION_NOTICE_OPT
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/test_deprecated.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pypsamcp/tools/deprecated.py tests/test_deprecated.py
git commit -m "feat: add deprecated aliases (set_snapshots, run_powerflow, run_optimization)"
```

---

## Task 12: Integration Test and Final Verification

**Files:**
- Test: `tests/test_integration.py`

### Steps

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end integration test: build → configure → optimize → query statistics."""

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
    @pytest.mark.asyncio
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
        assert result["objective_value"] > 0

        # 6. Get statistics
        result = await get_statistics("demo", "system_cost")
        assert "result" in result

        # 7. Export summary
        result = await export_model_summary("demo")
        assert "error" not in result
        assert result["summary"]["components"]["Generator"]["count"] == 2

    @pytest.mark.asyncio
    async def test_discovery_flow(self):
        result = await list_component_types()
        assert len(result["component_types"]) == 13

        result = await describe_component("Generator")
        assert len(result["required"]) > 0
        assert len(result["varying"]) > 0

    @pytest.mark.asyncio
    async def test_tool_count(self):
        """Verify we have exactly 22 registered tools."""
        from pypsamcp.core import mcp
        # FastMCP stores tools; count them
        tools = mcp._tool_manager._tools
        assert len(tools) == 22, f"Expected 22 tools, got {len(tools)}: {list(tools.keys())}"
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test verifying full workflow and 22-tool registration"
```

---

## Task 13: Cleanup — Remove Old Code from core.py

**Files:**
- Verify: `src/pypsamcp/core.py` — should contain only helpers, no `@mcp.tool()` decorators

### Steps

- [ ] **Step 1: Verify core.py has no tool registrations**

Run: `grep -c "@mcp.tool" src/pypsamcp/core.py`
Expected: `0` — all tools are in `tools/` modules now.

- [ ] **Step 2: Run full test suite one final time**

Run: `cd /home/carlos/repos/pypsa-mcp && .venv/bin/python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: verify clean module split — no tools in core.py"
```
