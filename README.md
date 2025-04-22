# PyPSA MCP

PyPSA MCP is a Model Context Protocol (MCP) server for creating, analyzing, and optimizing energy system models using PyPSA (Python for Power System Analysis).

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables Large Language Models (LLMs) like Claude to interact with PyPSA for energy model creation and analysis via natural language.

## Demo Example

Below is a demo video showing how to use PyPSA MCP with Claude. The video demonstrates creating a simple two-bus model, running power flow calculations, and performing optimization.

https://github.com/user-attachments/assets/5633a431-7c3b-4a2f-9a9e-395dcbbb2e29

### Demo Prompt

You can try this exact prompt with Claude to reproduce the example shown in the video:

```text
I'd like to build an energy system model and perform optimization using PyPSA. Please help me with these steps: 
1. Create a simple two-bus model with: 
   1. Two buses at (0,0) and (100,0) with 220 kV nominal voltage 
   2. A generator at bus1 with 100 MW capacity and 50 €/MWh cost 
   3. A load at bus2 with 80 MW demand
   4. 24 hourly snapshots for January 1, 2025
2. Run a power flow calculation to verify the model 
3. Perform optimization with the highs solver using the kirchhoff formulation 
4. Discuss the results
```


## Overview

PyPSA MCP provides a bridge between Large Language Models and PyPSA, allowing you to:

1. Create and manage energy system models through natural language
2. Add network components like buses, generators, and transmission lines
3. Set up time series data for simulation
4. Run power flow and optimization calculations
5. Analyze results

## Features

- **Model Management**
  - Create new PyPSA energy models
  - List and select from available models
  - Export detailed model summaries
  - Delete models when no longer needed

- **Component Creation**
  - Add buses, generators, loads, and other network components
  - Configure component parameters through natural language
  - Modify existing components
  - Organize components into meaningful groups

- **Data and Simulation**
  - Set time snapshots for simulation periods
  - Add time series data for loads and generators
  - Run power flow calculations
  - Perform optimization with various solvers and formulations

- **Results Analysis**
  - Extract key metrics from simulation results
  - Generate summaries of model performance
  - Export data for further analysis

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for easy dependency management)

### Main Installation (PyPI)

```bash
# Install from PyPI
pip install pypsamcp

# Or using uv (recommended)
uv pip install pypsamcp
```

### Running PyPSA MCP

```bash
# Run using the installed package
pypsamcp
```

### Configuring in Claude Desktop

1. Locate Claude Desktop's configuration file (typically in `~/.config/Claude/config.json`)

2. Add PyPSA MCP to the `mcpServers` section:

   ```json
   "mcpServers": {
     "PyPSA MCP":{
       "command": "uv",
       "args": [
         "run",
         "--with",
         "pypsamcp",
         "pypsamcp"
       ]
     }
   }
   ```

3. Save the configuration file and restart Claude Desktop

### Development Installation (from GitHub)

For contributors or users who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/cdgaete/pypsa-mcp.git
cd pypsa-mcp

# Install development dependencies with uv
uv pip install -e ".[dev]"
```

#### Running in Development Mode

```bash
# Run the server directly
python -m pypsamcp.server
```

## Available Tools

The server provides the following MCP tools:

### Model Management

```python
create_energy_model(
    id: str,
    name: str = None,
    description: str = None
)
```

```python
list_models()
```

```python
delete_model(
    id: str
)
```

```python
export_model_summary(
    id: str,
    include_components: bool = True,
    include_parameters: bool = True
)
```

### Component Creation

```python
add_bus(
    model_id: str,
    name: str,
    v_nom: float,
    x: float = 0.0,
    y: float = 0.0,
    carrier: str = "AC"
)
```

```python
add_generator(
    model_id: str,
    name: str,
    bus: str,
    p_nom: float,
    marginal_cost: float = 0.0,
    carrier: str = "generator"
)
```

```python
add_load(
    model_id: str,
    name: str,
    bus: str,
    p_set: float
)
```

```python
add_line(
    model_id: str,
    name: str,
    bus0: str,
    bus1: str,
    x: float,
    r: float = 0.0,
    g: float = 0.0,
    b: float = 0.0,
    s_nom: float = 0.0
)
```

```python
add_storage(
    model_id: str,
    name: str,
    bus: str,
    p_nom: float,
    max_hours: float,
    efficiency_store: float = 1.0,
    efficiency_dispatch: float = 1.0,
    standing_loss: float = 0.0
)
```

### Data and Simulation

```python
set_snapshots(
    model_id: str,
    start_time: str,
    end_time: str,
    freq: str = "H"
)
```

```python
run_powerflow(
    model_id: str,
    snapshot: str = None
)
```

```python
run_optimization(
    model_id: str,
    solver_name: str = "glpk",
    formulation: str = "kirchhoff"
)
```

## Example Prompts

Here are some examples of how to use PyPSA MCP with Claude:

```text
Create a new energy system model with three buses, two generators, and a load.
```

```text
Add a wind generator with 100 MW capacity to bus "bus1" with a marginal cost of 10.
```

```text
Run a power flow calculation on the current model and show me the results.
```

```text
Optimize the model using the GLPK solver and summarize the key findings.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [PyPSA](https://github.com/PyPSA/PyPSA) for power system modeling
- Uses [FastMCP](https://github.com/jlowin/fastmcp) for the Model Context Protocol implementation
- Inspired by the need to make energy system modeling more accessible through natural language interfaces
