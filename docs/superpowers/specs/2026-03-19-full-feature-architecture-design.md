# PyPSA MCP — Full Feature Architecture Design

**Date:** 2026-03-19
**Status:** Approved
**Target:** PyPSA 1.1.2, 22 registered tools (19 primary + 3 deprecated aliases)

---

## Overview

Refactor pypsa-mcp from 11 tools in a single `core.py` to 19 composable tools + 3 deprecated aliases across a modular file structure. Every PyPSA 1.1.2 capability is exposed through parameterized, multi-mode tools.

The 19 primary tools include 4 convenience wrappers (add_bus, add_generator, add_load, add_line) that are part of the core inventory but marked deprecated. The 3 additional deprecated aliases (set_snapshots, run_powerflow, run_optimization) bring the total to 22 registered tools.

## Module Structure

```
src/pypsamcp/
├── __init__.py
├── server.py                  # Entry point (unchanged)
├── core.py                    # mcp, MODELS, helpers (imports tools/ at bottom)
├── tools/
│   ├── __init__.py            # Imports all tool modules to trigger @mcp.tool() registration
│   ├── management.py          # create_energy_model, list_models, delete_model, export_model_summary
│   ├── discovery.py           # list_component_types, describe_component
│   ├── components.py          # add_component, update_component, remove_component, query_components
│   ├── convenience.py         # add_bus, add_generator, add_load, add_line (thin delegates, deprecated)
│   ├── time_config.py         # configure_time
│   ├── simulation.py          # run_simulation
│   ├── statistics.py          # get_statistics
│   ├── clustering.py          # cluster_network
│   ├── io.py                  # network_io
│   └── deprecated.py          # set_snapshots, run_powerflow, run_optimization (aliases)
```

**Registration wiring:** `core.py` defines `mcp` and helpers, then at the bottom imports `pypsamcp.tools` to trigger all `@mcp.tool()` decorators. `server.py` imports `mcp` from `core` as before — no change needed.

## Design Decisions

1. **Module split (not monolith):** core.py would grow to 1500-2000+ lines. Split by domain keeps files focused and editable.
2. **`pypsa>=1.1.0`:** New tools use 1.x APIs (statistics accessor, cluster accessor, set_investment_periods, optimize.optimize_mga). Clean break from 0.x.
3. **Thin convenience wrappers:** `add_bus`, `add_generator`, `add_load`, `add_line` repackage args and delegate to `add_component`. Marked deprecated — will be removed in a future release.
4. **Deprecated aliases:** `set_snapshots` → `configure_time(mode="snapshots")`, `run_powerflow` → `run_simulation(mode="pf")`, `run_optimization` → `run_simulation(mode="optimize")`. Each returns a `deprecation_notice` field.
5. **Helper rename:** `get_model` → `get_energy_model` throughout.
6. **CamelCase canonical for component_type:** `"Bus"`, `"Generator"`, etc. match PyPSA docs and `network.add("Bus", ...)`. Build a lookup dict mapping both CamelCase and lowercase list_name forms.
7. **No `**kwargs` in tool signatures:** MCP tools require fully typed parameters for JSON Schema generation. Multi-mode tools declare all mode-specific params explicitly with `None` defaults.
8. **`optimize()` returns `(status, condition)` tuple:** Capture return value directly, do not rely on `network.results`.
9. **Exclude internal component types:** `SubNetwork` and `Shape` are managed internally by PyPSA. Excluded from `list_component_types` and rejected by `add_component`.

## What Does NOT Change

- `stdout_to_stderr` context manager
- `convert_to_serializable` helper
- `MODELS` dict
- `mcp` FastMCP object (`"pypsa-mcp"`, `on_duplicate="error"`)
- `server.py` entry point
- All tools remain `async def` returning `dict`

## Return Shape Convention

All tools return either:
- **Success:** dict with at minimum a `"message"` key, plus domain-specific fields
- **Error:** `{"error": str}` with an actionable error message

---

## Tool Inventory (19 primary + 3 deprecated = 22 total)

### GROUP 1 — Model Management (4 tools) — `tools/management.py`

#### `create_energy_model(model_id, name=None, override=False) → dict`
Creates a new PyPSA network and stores in MODELS. Unchanged from current.

#### `list_models() → dict`
Lists all models with summaries. Unchanged from current.

#### `delete_model(model_id) → dict`
Removes a model. Unchanged from current.

#### `export_model_summary(model_id) → dict`
Extended to report: `has_investment_periods`, `has_scenarios`, `has_risk_preference`, `investment_periods`, `scenarios` in addition to existing fields.

### GROUP 2 — Discovery (2 tools) — `tools/discovery.py`

#### `list_component_types() → dict`
Returns catalog of 13 user-facing PyPSA component types (excluding internal SubNetwork and Shape). No arguments, no model required. Always fast.

CamelCase `type` is the canonical form used across all tools. `list_name` provided for reference.

```json
{
  "component_types": [
    {"type": "Bus", "list_name": "buses", "description": "Electrical node connecting components"},
    {"type": "Generator", "list_name": "generators", "description": "Power source with cost and capacity parameters"},
    {"type": "Load", "list_name": "loads", "description": "Power demand at a bus"},
    {"type": "Line", "list_name": "lines", "description": "AC transmission line between two buses"},
    {"type": "Link", "list_name": "links", "description": "Controllable branch: HVDC, heat pumps, electrolyzers, sector coupling"},
    {"type": "StorageUnit", "list_name": "storage_units", "description": "Coupled power-energy storage: battery, pumped hydro, reservoir"},
    {"type": "Store", "list_name": "stores", "description": "Energy reservoir decoupled from power: H2 tank, heat store"},
    {"type": "Transformer", "list_name": "transformers", "description": "Voltage-level coupling with tap ratio and phase shift"},
    {"type": "ShuntImpedance", "list_name": "shunt_impedances", "description": "Shunt conductance/susceptance at a bus"},
    {"type": "Carrier", "list_name": "carriers", "description": "Energy carrier with CO2 emissions, color, growth limits"},
    {"type": "GlobalConstraint", "list_name": "global_constraints", "description": "System-wide constraint: CO2 cap, primary energy limit, capacity target"},
    {"type": "LineType", "list_name": "line_types", "description": "Standard AC line type library (r/x/c per km)"},
    {"type": "TransformerType", "list_name": "transformer_types", "description": "Standard transformer type library"}
  ]
}
```

#### `describe_component(component_type, include_defaults=True) → dict`
Returns full input-parameter schema for one component type, introspected live from the PyPSA component attribute system.

**Note on PyPSA 1.x API:** The `.attrs` accessor is deprecated in favor of `.defaults`. Implementation should use the non-deprecated API or suppress the warning. The filtering logic (status, varying columns) applies to whichever accessor is used.

**Internal logic:**
1. Use any existing model from MODELS, or instantiate a throwaway `pypsa.Network()`.
2. Validate `component_type` against the 13 user-facing types; return error with valid list if unknown.
3. Load component attributes. Filter to rows where `status` starts with `"Input"`.
4. Partition into:
   - `required`: `status == "Input (required)"` — always `name` and bus reference(s).
   - `static`: `status == "Input (optional)"` and `varying == False`.
   - `varying`: `status == "Input (optional)"` and `varying == True`.
5. For each param: `attr`, `default` (if include_defaults), `unit`, `description`, `typ`.
6. Include a `note` field:
   ```
   "Pass 'static' and 'varying' params to add_component(params={}). Pass time series for 'varying' params to add_component(time_series={})."
   ```

### GROUP 3 — Component CRUD (4 tools) — `tools/components.py`

#### `add_component(model_id, component_type, component_id, params, time_series) → dict`
Adds any PyPSA component with full parameter coverage.

**Arguments:**
- `model_id: str`
- `component_type: str` — CamelCase canonical form
- `component_id: str`
- `params: dict | None` — static/scalar input parameters
- `time_series: dict | None` — `{attr: [v1, v2, ...]}` for varying input fields

**Validation (fail-fast, in order):**
1. Model exists.
2. `component_type` valid (one of 13 user-facing types).
3. All `params` keys are `Input` attributes (not `Output`). Return named bad keys + suggest `describe_component`.
4. Bus references resolve. Bus attr names per type:
   - Single `bus`: Generator, Load, StorageUnit, Store, ShuntImpedance.
   - `bus0`+`bus1`: Line, Transformer.
   - `bus0`+`bus1`+optional `bus2`…`busN` in params: Link.
   - No check: Carrier, GlobalConstraint, LineType, TransformerType, Bus.
5. `component_id` not already in component DataFrame index.
6. All `time_series` keys are `varying=True` input attributes.
7. If `time_series` non-empty, `len(network.snapshots) > 0`. Return actionable error if not.

**Execution:**
1. Type-cast each param from attribute type info. Handle `bool` explicitly.
2. `network.add(component_type, component_id, **params)`.
3. For each `(attr, values)` in `time_series`: assign `pd.Series(values, index=network.snapshots[:len(values)])` to the appropriate dynamic DataFrame.
4. Return full serialized component row + count.

#### `update_component(model_id, component_type, component_id, params, time_series) → dict`
Same validation as `add_component` but step 5 inverted (component must exist). Overwrites only supplied keys.

#### `remove_component(model_id, component_type, component_id) → dict`
Removes a component. Returns remaining count.

#### `query_components(model_id, component_type, filters) → dict`
Read component state. `filters` is optional `{attr: value}` dict. Returns serialized DataFrame rows + count.

**Arguments (all explicit, no kwargs):**
- `model_id: str`
- `component_type: str`
- `filters: dict | None = None`

### GROUP 3b — Convenience Wrappers (4 tools, deprecated) — `tools/convenience.py`

`add_bus`, `add_generator`, `add_load`, `add_line` — narrow signatures, delegate internally to `add_component`. Each returns a `deprecation_notice` field. Will be removed in a future release. These are counted within the 19 primary tools.

### GROUP 4 — Time & Investment Structure (1 tool) — `tools/time_config.py`

#### `configure_time(model_id, mode, ...) → dict`

All parameters declared explicitly with `None` defaults (no `**kwargs`):

```python
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
```

**Mode behavior:**
- `"snapshots"` — `network.set_snapshots(pd.to_datetime(snapshots))`. Optional weightings.
- `"investment_periods"` — `network.set_investment_periods(periods)`. Snapshots must be set first.
- `"scenarios"` — `network.set_scenarios(scenarios)`.
- `"risk_preference"` — `network.set_risk_preference(alpha, omega)`. Requires has_scenarios.

**Return:** mode, message, has_investment_periods, has_scenarios, has_risk_preference, snapshot_count, investment_period_weightings.

### GROUP 5 — Simulation (1 tool) — `tools/simulation.py`

#### `run_simulation(model_id, mode, ...) → dict`

All parameters declared explicitly (no `**kwargs`):

```python
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
```

**Mode → PyPSA method mapping:**

| Mode | PyPSA method |
|------|-------------|
| `"pf"` | `network.pf()` |
| `"lpf"` | `network.lpf()` |
| `"optimize"` | `network.optimize()` |
| `"mga"` | `network.optimize.optimize_mga()` |
| `"security_constrained"` | `network.optimize.optimize_security_constrained()` |
| `"rolling_horizon"` | `network.optimize.optimize_with_rolling_horizon()` |
| `"transmission_expansion_iterative"` | `network.optimize.optimize_transmission_expansion_iteratively()` |
| `"optimize_and_pf"` | `network.optimize.optimize_and_run_non_linear_powerflow()` |

**Result extraction:** `network.optimize()` returns `(status, termination_condition)` tuple. Capture directly — do not rely on `network.results` dict.

All modes wrapped in `stdout_to_stderr()`. `extra_functionality` uses same `textwrap.indent` + `exec` pattern as current.

**Return:** mode, status, termination_condition, objective_value, summary (dispatch, expansion, flows, shadow prices).

### GROUP 6 — Statistics & Results (1 tool) — `tools/statistics.py`

#### `get_statistics(model_id, metric, ...) → dict`

All parameters declared explicitly:

```python
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
```

**19 metrics** (all map to `network.statistics.<metric>(**kwargs)`): system_cost, capex, opex, fom, overnight_cost, installed_capacity, optimal_capacity, expanded_capacity, installed_capex, expanded_capex, capacity_factor, curtailment, energy_balance, supply, withdrawal, revenue, market_value, prices, transmission.

**Special metric `"all"`:** Loops over all 19 metrics and returns combined dict. Not a PyPSA method — implemented as iteration.

**Requires:** model must be solved. Return clear error if not.

### GROUP 7 — Clustering / Reduction (1 tool) — `tools/clustering.py`

#### `cluster_network(model_id, domain, method, output_model_id, ...) → dict`

All parameters declared explicitly:

```python
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
```

**`domain="spatial"`** methods: kmeans, hac, greedy_modularity.
- `bus_weightings`: if `None`, auto-compute from load at each bus.

**`domain="temporal"`** methods: resample, downsample, segment, snapshot_map.

Result stored as new model in MODELS[output_model_id].

### GROUP 8 — I/O & Network Operations (1 tool) — `tools/io.py`

#### `network_io(model_id, operation, ...) → dict`

All parameters declared explicitly:

```python
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
```

**Operations:** export_netcdf, export_csv, import_netcdf, import_csv, merge, copy, consistency_check.

**Removed from original spec:**
- `export_hdf5` — requires extra dependencies (`tables`/`h5py`) not in pyproject.toml.
- `slice` — `network.slice()` does not exist in PyPSA 1.1.2. Use `copy` with `snapshots`/`buses` params instead.

---

## Deprecated Aliases (3 tools) — `tools/deprecated.py`

| Old name | Delegates to | Notes |
|----------|-------------|-------|
| `set_snapshots(model_id, snapshots)` | `configure_time(model_id, mode="snapshots", snapshots=snapshots)` | Returns deprecation_notice |
| `run_powerflow(model_id, snapshot)` | `run_simulation(model_id, mode="pf", snapshots=[snapshot])` | Returns deprecation_notice |
| `run_optimization(model_id, solver_name, formulation, extra_functionality)` | `run_simulation(model_id, mode="optimize", ...)` | Returns deprecation_notice |

---

## Progressive Disclosure Flow

```
Orientation
  └─ list_component_types()       "What can I model?"
  └─ describe_component(type)     "What params does StorageUnit take?"

Build
  └─ create_energy_model()
  └─ configure_time(mode="snapshots")
  └─ configure_time(mode="investment_periods")
  └─ add_bus / add_generator / add_load / add_line   ← happy path (deprecated)
  └─ add_component(type, id, params, time_series)    ← full power

Simulate
  └─ run_simulation(mode="optimize")
  └─ run_simulation(mode="mga")
  └─ run_simulation(mode="security_constrained")

Analyse
  └─ get_statistics(metric="system_cost")
  └─ get_statistics(metric="energy_balance", groupby="carrier")
  └─ get_statistics(metric="all")

Reduce
  └─ cluster_network(domain="temporal", method="segment", num_segments=12)
  └─ cluster_network(domain="spatial",  method="kmeans",  n_clusters=50)

Persist
  └─ network_io(operation="export_netcdf", path="model.nc")
  └─ network_io(operation="import_netcdf", path="model.nc", model_id="loaded")
  └─ network_io(operation="merge", other_model_id="network_b", output_model_id="merged")
```
