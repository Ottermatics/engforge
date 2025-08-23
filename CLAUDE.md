# EngForge - Claude Development Guide

## Overview
EngForge is a sophisticated engineering systems framework that provides tabulation, analysis, and solver capabilities for complex engineering problems. It's built on a component-system architecture with dynamic programming optimizations and integrated reporting.

## Core Architecture

### Key Design Patterns

1. **Component-System Architecture**: The framework uses a hierarchical design where:
   - `Component`: Basic building blocks that encapsulate engineering calculations and properties
   - `System`: Orchestrates multiple components, manages data flow, and provides solver capabilities
   - Both inherit from `SolveableInterface` which provides tabulation, configuration, and solving capabilities

2. **Configuration-Based Design**: Uses the `@forge` decorator (similar to `@attrs.define`) to automatically configure classes with:
   - Attribute validation and transformation 
   - Property change tracking
   - Signal and slot management
   - Automatic name generation

3. **Cached Property System**: Implements dynamic programming through cached properties:
   - `@system_property`: Basic cached properties
   - `@cached_system_property`: Only recalculates when dependencies change
   - Properties automatically track dependencies and invalidate when inputs change

4. **Signal-Slot Pattern**: Data flow between components is managed via:
   - `Signal.define(source, target, mode)`: Defines data flow connections
   - `Slot.define(ComponentType)`: Defines component mounting points
   - Supports pre/post execution modes and control signals

### Core Classes and Concepts

#### Configuration (`engforge/configuration.py`)
- `@forge` decorator: Main class decorator that configures all EngForge classes
- `Configuration`: Base class providing attribute management and property change callbacks
- Automatic name generation using randomized technical terms
- Property change tracking with `property_changed` callback

#### Component (`engforge/components.py`) 
- `SolveableInterface`: Common base providing tabulation, configuration, and solving
- `Component`: Basic building block with evaluation and caching capabilities
- `DynamicsMixin`: Provides time-based integration capabilities

#### System (`engforge/system.py`)
- `System`: Orchestrates components, manages solver execution, provides plotting
- Inherits from `SolverMixin`, `SolveableInterface`, `PlottingMixin`, `GlobalDynamics`
- Manages component slots and signal routing
- Provides solver execution with constraint handling

#### Solver Framework (`engforge/solver.py`, `engforge/attr_solver.py`)
- `SolverMixin`: Core solver capabilities using scipy optimizers
- `Solver.define(dependent, independent)`: Defines solver equations
- Constraint system with min/max limits and custom functions
- Supports multiple solver combos and variable sets

#### Attributes System (`engforge/attributes.py`, `engforge/attr_*.py`)
- `ATTR_BASE`: Base attribute system with dependency tracking
- `Time`: Time integration attributes for dynamics
- `Signal`: Data flow attributes
- `Slot`: Component mounting attributes  
- `Plot`/`Trace`: Plotting and visualization attributes
- All attributes support instance-level customization and caching

### Property and Caching System

The framework implements sophisticated caching through several property decorators:

```python
@system_property
def calculated_value(self) -> float:
    """Basic cached property"""
    return expensive_calculation()

@cached_system_property  
def dynamic_value(self) -> float:
    """Only recalculates when dependencies change"""
    return self.input_a * self.input_b
```

Properties automatically:
- Cache results until dependencies change
- Track inter-property dependencies
- Support type validation and conversion
- Integrate with the solver system for constraint handling

### Data Flow and Tabulation

The framework provides comprehensive data capture through:

1. **System References** (`engforge/system_reference.py`): Lightweight references to nested attributes
2. **DataframeMixin** (`engforge/dataframe.py`): Automatic pandas DataFrame generation
3. **TabulationMixin** (`engforge/tabulation.py`): Data collection and organization

All component attributes and system properties are automatically captured into structured DataFrames suitable for analysis and reporting.

## Engineering Domain Libraries

### Fluid Mechanics (`engforge/eng/fluid_material.py`)
- `FluidMaterial`: Base class for fluid property calculations
- `CoolPropMaterial`: Integration with CoolProp thermodynamic database
- Built-in materials: Air, Water, Steam, Hydrogen, Oxygen
- Support for mixtures and ideal gas approximations

### Solid Mechanics (`engforge/eng/solid_materials.py`)
- `SolidMaterial`: Material property definitions
- Standard materials: Steel (SS_316, ANSI_4130/4340), Aluminum, Carbon Fiber, Concrete
- Stress/strain calculations and safety factors

### Thermodynamics (`engforge/eng/thermodynamics.py`)
- Component models: Compressor, Turbine, Heat Exchanger, Pump
- Cycle analysis capabilities
- Heat transfer and pressure drop correlations

### Structural Analysis (`engforge/eng/structure_beams.py`, `engforge/eng/geometry.py`)
- Beam analysis with PyNite FEA integration
- Cross-sectional property calculations
- Geometric primitives and section properties

### Pipe Networks (`engforge/eng/pipes.py`)
- Flow network modeling with pressure drop calculations
- Pump and fitting models
- System-level flow analysis

### Cost Analysis (`engforge/eng/costs.py`)
- Engineering economics with time value of money
- Cost categorization and breakdown analysis
- Integration with system design parameters

## Testing Framework

The project uses Python's built-in `unittest` framework:

### Test Structure
- Tests located in `engforge/test/` directory
- Import tests in `test_modules.py` verify all modules load correctly
- Domain-specific tests for components, solver, dynamics, etc.

### Running Tests
```bash
# Install package in development mode
pip install -e .

# Run all tests
python -m unittest discover -s engforge/test -p "test_*.py" -v

# Run specific test module
python -m unittest engforge.test.test_modules -v
```

### Test Dependencies
The full test suite requires all optional dependencies. Core functionality can be tested with:
- `attrs>=23.2.0` - Core attribute system
- `numpy>=1.24.3` - Numerical computations  
- `scipy` - Solver algorithms
- `matplotlib>=3.8.1` - Plotting
- `pandas` - Data handling

## Examples and Usage Patterns

### Air Filter Analysis Example
Located in `examples/air_filter.py`, demonstrates:
- Component definition with `@forge` decorator
- System assembly with slots and signals
- Solver configuration for flow balancing
- Plotting integration and data analysis

### Spring-Mass-Damper Example  
Shows dynamic system analysis with:
- Time integration using `Time.integrate()`
- Solver constraints and variable handling
- Multi-parameter studies with `run()` method

### Analysis and Reporting
The `Analysis` class provides:
- System wrapping with additional post-processing
- Multiple reporter integration (CSV, Excel, plots)
- Automated data collection and storage

## Development Guidelines

### Code Quality Focus
- **Only modify non-scientific code**: Algorithmic and engineering calculations should not be altered
- **Follow existing patterns**: Use `@forge`, inherit from appropriate base classes
- **Maintain property caching**: Ensure expensive calculations are properly cached
- **Preserve signal-slot architecture**: Data flow should use defined patterns

### Build-Test-Fix Process
The project uses a comprehensive CI/CD pipeline that can be validated locally:

#### Local Validation Steps
1. **Create clean test environment:**
   ```bash
   cd /tmp && python -m venv test_engforge_env
   source test_engforge_env/bin/activate
   pip install --upgrade pip
   ```

2. **Install package and dependencies:**
   ```bash
   cd <project_root>
   source /tmp/test_engforge_env/bin/activate
   pip install -e .                    # Core dependencies only
   pip install -e .[all]               # All optional dependencies
   pip install -e .[database]          # Database functionality only
   pip install -e .[google,cloud]      # Specific optional groups
   ```

3. **Test version extraction (CI compatibility):**
   ```bash
   python -c "
   try:
       import tomllib
   except ImportError:
       import tomli as tomllib
   with open('pyproject.toml', 'rb') as f:
       print('✅ Version:', tomllib.load(f)['project']['version'])
   "
   ```

4. **Run tests:**
   ```bash
   python -m unittest discover -s engforge/test -p "test_*.py" -v
   ```

5. **Check code formatting:**
   ```bash
   black --check --verbose ./engforge
   ```

#### Common Issues and Solutions
- **tomllib ImportError**: Fixed with `tomli>=1.2.0;python_version<'3.11'` dependency
- **SQLAlchemy missing**: Auto-installation will prompt when importing datastores, or install manually with `pip install engforge[database]`
- **Black formatting**: All files should pass formatting checks; use `black ./engforge` to auto-format

#### Optional Dependencies & Auto-Installation
The project uses modern pyproject.toml optional dependencies with intelligent auto-installation:

**Dependency Groups:**
- **`[database]`**: SQLAlchemy, PostgreSQL, disk caching
- **`[google]`**: Google Sheets and Drive integration
- **`[cloud]`**: AWS boto3 integration  
- **`[distributed]`**: Ray distributed computing
- **`[all]`**: All optional dependencies combined

**Auto-Installation Behavior:**
- When importing `engforge.datastores`, missing dependencies trigger an auto-install prompt
- Reads pyproject.toml directly to determine required packages
- Works in development mode (editable installs) by detecting project root
- Graceful fallback to manual installation instructions

**Manual Installation:**
```bash
pip install engforge[database,cloud]  # Specific groups
pip install engforge[all]             # All optional dependencies
pip install -e .[all]                 # Development mode with all deps
```

### Key Files to Understand
1. `configuration.py` - Core class configuration system
2. `components.py` - Component base classes  
3. `system.py` - System orchestration
4. `properties.py` - Property and caching decorators
5. `attributes.py` - Attribute system foundation

### Common Patterns
```python
@forge  # Always use this decorator
class MyComponent(Component):
    # Use attrs.field for inputs
    input_value: float = attrs.field(default=1.0)
    
    # Use system_property for calculated outputs
    @system_property 
    def output_value(self) -> float:
        return self.input_value * 2.0

@forge
class MySystem(System):
    # Define component slots
    comp: MyComponent = Slot.define(MyComponent)
    
    # Define data flow
    set_input = Signal.define('external_param', 'comp.input_value')
    
    # Define solver if needed
    solver = Solver.define('constraint_equation', 'control_variable')
```

The framework emphasizes declarative configuration over imperative programming, extensive use of caching for performance, and clear separation between data flow definition and execution.