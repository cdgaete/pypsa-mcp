# PyPSA MCP — Full Feature Architecture Design

**Date:** 2026-03-19
**Status:** Approved
**Target:** PyPSA 1.1.2, ≤ 20 registered tools (19 primary + 3 deprecated aliases)

---

## Overview

Refactor pypsa-mcp from 11 tools in a single `core.py` to 19 composable tools + 3 deprecated aliases across a modular file structure. Every PyPSA 1.1.2 capability is exposed through parameterized, multi-mode tools that stay within the 20-tool context window budget.

## Module Structure

```
src/pypsamcp/
├── __init__.py
├── server.py                  # Entry point (unchanged)
├── core.py                    # mcp, MODELS, helpers
├── tools/
│   ├── __init__.py            # Imports all tool modules to trigger registration
│   ├── management.py          # create_energy_model, list_models, delete_model, export_model_summary
│   ├── discovery.py           # list_component_types, describe_component
│   ├── components.py          # add_component, update_component, remove_component, query_components
│   ├── convenience.py         # add_bus, add_generator, add_load, add_line (thin delegates)
│   ├── time_config.py         # configure_time
│   ├── simulation.py          # run_simulation
│   ├── statistics.py          # get_statistics
│   ├── clustering.py          # cluster_network
│   ├── io.py                  # network_io
│   └── deprecated.py          # set_snapshots, run_powerflow, run_optimization (aliases)
```

## Design Decisions

1. **Module split (not monolith):** core.py would grow to 1500-2000+ lines. Split by domain keeps files focused and editable.
2. **`pypsa>=1.1.0`:** New tools use 1.x APIs (statistics accessor, cluster accessor, set_investment_periods, optimize.optimize_mga). Clean break from 0.x.
3. **Thin convenience wrappers:** `add_bus`, `add_generator`, `add_load`, `add_line` repackage args and delegate to `add_component`. Marked deprecated.
4. **Deprecated aliases:** `set_snapshots` → `configure_time(mode="snapshots")`, `run_powerflow` → `run_simulation(mode="pf")`, `run_optimization` → `run_simulation(mode="optimize")`. Each returns a `deprecation_notice` field.
5. **Helper rename:** `get_model` → `get_energy_model` throughout.

## What Does NOT Change

- `stdout_to_stderr` context manager
- `convert_to_serializable` helper
- `MODELS` dict
- `mcp` FastMCP object (`"pypsa-mcp"`, `on_duplicate="error"`)
- `server.py` entry point
- All tools remain `async def` returning `dict`

---

## Tool Inventory (19 + 3 deprecated)

### GROUP 1 — Model Management (4 tools)

#### `create_energy_model(model_id, name=None, override=False) → dict`
Creates a new PyPSA network and stores in MODELS. Unchanged from current.

#### `list_models() → dict`
Lists all models with summaries. Unchanged from current.

#### `delete_model(model_id) → dict`
Removes a model. Unchanged from current.

#### `export_model_summary(model_id) → dict`
Extended to report: `has_investment_periods`, `has_scenarios`, `has_risk_preference`, `investment_periods`, `scenarios` in addition to existing fields.

### GROUP 2 — Discovery (2 tools)

#### `list_component_types() → dict`
Returns catalog of all 13 PyPSA component types with one-line descriptions. No arguments, no model required. Always fast.

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
Returns full input-parameter schema for one component type, introspected live from `network.components[component_type]['attrs']`.

**Internal logic:**
1. Use any existing model from MODELS, or instantiate a throwaway `pypsa.Network()`.
2. Validate `component_type`; return error with valid list if unknown.
3. Load attrs. Filter to rows where `status` starts with `"Input"`.
4. Partition into:
   - `required`: `status == "Input (required)"` — always `name` and bus reference(s).
   - `static`: `status == "Input (optional)"` and `varying == False`.
   - `varying`: `status == "Input (optional)"` and `varying == True`.
5. For each param: `attr`, `default` (if include_defaults), `unit`, `description`, `typ`.
6. Include a `note` field:
   ```
   "Pass 'static' and 'varying' params to add_component(params={}). Pass time series for 'varying' params to add_component(time_series={})."
   ```

### GROUP 3 — Component CRUD (4 tools)

#### `add_component(model_id, component_type, component_id, params, time_series) → dict`
Adds any PyPSA component with full parameter coverage.

**Validation (fail-fast, in order):**
1. Model exists.
2. `component_type` valid.
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
1. Type-cast each param from `attrs.loc[attr, 'typ']`. Handle `bool` explicitly.
2. `network.add(component_type, component_id, **params)`.
3. For each `(attr, values)` in `time_series`: assign `pd.Series(values, index=network.snapshots[:len(values)])` to `getattr(network, f"{list_name}_t")[attr][component_id]`.
4. Return full serialized component row + count.

#### `update_component(model_id, component_type, component_id, params, time_series) → dict`
Same validation as `add_component` but step 5 inverted (component must exist). Overwrites only supplied keys.

#### `remove_component(model_id, component_type, component_id) → dict`
Removes a component. Returns remaining count.

#### `query_components(model_id, component_type, filters) → dict`
Read component state. `filters` is optional `{attr: value}` dict. Returns serialized DataFrame rows + count.

### GROUP 3b — Convenience Wrappers (4 tools, deprecated)

`add_bus`, `add_generator`, `add_load`, `add_line` — narrow signatures, delegate internally to `add_component`. Each returns a `deprecation_notice` field. Will be removed in a future release.

### GROUP 4 — Time & Investment Structure (1 tool)

#### `configure_time(model_id, mode, **kwargs) → dict`

**`mode="snapshots"`**
- `snapshots: list[str]` — ISO datetime strings.
- `weightings: dict | None` — optional, keys `"objective"`, `"generators"`, `"stores"`.

**`mode="investment_periods"`**
- `periods: list[int]` — e.g. `[2025, 2030, 2035]`.
- Snapshots must be set first.

**`mode="scenarios"`**
- `scenarios: dict | list | None`

**`mode="risk_preference"`**
- `alpha: float` — CVaR confidence level.
- `omega: float` — weight on CVaR term.
- Requires `has_scenarios == True`.

**Return:** mode, message, has_investment_periods, has_scenarios, has_risk_preference, snapshot_count, investment_period_weightings.

### GROUP 5 — Simulation (1 tool)

#### `run_simulation(model_id, mode, **kwargs) → dict`

**Modes:**
- `"pf"` — Non-linear AC power flow. kwargs: snapshots, distribute_slack, slack_weights, x_tol.
- `"lpf"` — Linear power flow. kwargs: snapshots.
- `"optimize"` — Standard optimization. kwargs: solver_name, formulation, multi_investment_periods, transmission_losses, linearized_unit_commitment, assign_all_duals, compute_infeasibilities, solver_options, extra_functionality.
- `"mga"` — Modelling-to-Generate-Alternatives. kwargs: slack, sense, weights, solver_name.
- `"security_constrained"` — N-1 security. kwargs: branch_outages, solver_name.
- `"rolling_horizon"` — Rolling horizon dispatch. kwargs: horizon, overlap, solver_name.
- `"transmission_expansion_iterative"` — Iterative transmission expansion. kwargs: msq_threshold, min_iterations, max_iterations, solver_name.
- `"optimize_and_pf"` — Optimize then non-linear power flow.

All modes wrapped in `stdout_to_stderr()`. `extra_functionality` uses same `textwrap.indent` + `exec` pattern as current.

**Return:** mode, status, termination_condition, objective_value, summary (dispatch, expansion, flows, shadow prices).

### GROUP 6 — Statistics & Results (1 tool)

#### `get_statistics(model_id, metric, **kwargs) → dict`

**19 metrics:** system_cost, capex, opex, fom, overnight_cost, installed_capacity, optimal_capacity, expanded_capacity, installed_capex, expanded_capex, capacity_factor, curtailment, energy_balance, supply, withdrawal, revenue, market_value, prices, transmission, all.

**Shared kwargs:** components, carrier, bus_carrier, groupby, aggregate_across_components, nice_names, drop_zero.

**Requires:** `network.is_solved == True`.

### GROUP 7 — Clustering / Reduction (1 tool)

#### `cluster_network(model_id, domain, method, output_model_id, **kwargs) → dict`

**`domain="spatial"`** methods: kmeans, hac, greedy_modularity. kwargs: n_clusters, bus_weightings, line_length_factor, affinity, linkage. Shared cluster_options: with_time, aggregate_generators_weighted, scale_link_capital_costs.

**`domain="temporal"`** methods: resample, downsample, segment, snapshot_map. kwargs: offset, stride, num_segments, solver, snapshot_map.

Result stored as new model in MODELS[output_model_id].

### GROUP 8 — I/O & Network Operations (1 tool)

#### `network_io(model_id, operation, **kwargs) → dict`

**Operations:** export_netcdf, export_csv, export_hdf5, import_netcdf, import_csv, merge, copy, slice, consistency_check.

Each operation has specific kwargs per the spec. Import operations create new models. Merge/copy/slice store results in MODELS[output_model_id].

---

## Deprecated Aliases (3 tools)

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
