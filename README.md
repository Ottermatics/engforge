[![build](https://github.com/Ottermatics/engforge/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Ottermatics/engforge/actions/workflows/build.yml)
[![docs](https://github.com/Ottermatics/engforge/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/Ottermatics/engforge/actions/workflows/docs.yml)

# EngForge

**Define engineering components as `Y = F(X)` — state in, properties out — compose them, and let the
solver drive the inputs to satisfy your constraints or optimize an objective.**

EngForge is a Python framework for modeling and solving coupled engineering systems: thermofluids,
structures, cost, controls — anything you can write as a function of state.

## Why

Real engineering models are networks of coupled relationships. A fan's pressure rise depends on a
flow that depends on a filter's loss that depends on that same pressure. Writing the solve loop by
hand — guess, evaluate, check, adjust, repeat — is where the bugs live, and it has to be rewritten
every time the problem is re-scoped (design vs. off-design, one site vs. many, steady vs. transient).

EngForge separates the **model** from the **solve**. You declare each component's physics once as
plain methods; you mark which inputs are free, which outputs must balance, and what to optimize; the
framework assembles and runs the solver. Re-scoping is a selection, not a rewrite.

This follows the lineage of **NPSS** (balances and integrators driving residuals to zero) and
NASA's **OpenMDAO** (implicit/explicit components assembled into one system of equations). EngForge
brings that residual-solver model into an ergonomic, class-based Python DSL with first-class data
tabulation and sweeps.

**Who it's for:** engineers and analysts building multidisciplinary models who want a Python-native,
reusable way to express `Y = F(X)` and solve or optimize across coupled components — without hand-rolling
solver glue or MDAO boilerplate.

## How data flows

```
  fields (X)  ──►  @system_property (Y = F(X))  ──►  recorded rows  ──►  Result / DataFrame
     ▲                      │
     └──── solver drives ───┘   (change X until residuals are satisfied / objective is optimal)
```

1. **State `X`** — a component's `field`s: the numbers you set, and the ones the solver is free to change.
2. **Properties `Y`** — `@system_property` methods: derived quantities, `Y = F(X)`, tabulated automatically.
3. **Composition** — components mount into systems through **slots**; **signals** push values between
   them; a system is just a component that owns others.
4. **The solve** — you declare solver **variables**, **constraints** (residuals driven to zero), and
   **objectives**. `run()` assembles the active set and solves it.
5. **Results** — a run returns a `Result` that owns its rows; render it as a pandas `DataFrame`. The
   model objects stay clean — results live with the run, not smeared across the components.

## Concepts

| Concept | Purpose |
|---|---|
| `@forge` `Component` / `System` | the unit of modeling — everything is a component; a `System` owns others |
| `field` | component **state** (`X`): inputs and solver-changeable variables |
| `@system_property` | a derived output (`Y = F(X)`), tabulated into every row |
| `Slot.define(...)` | mount a component into a system (composition); optional/swappable |
| `Signal.define(target, source)` | push a value from one place to another before/after execution |
| `Solver.declare_var(...)` | mark a field (local or `slot.path`) the solver may change |
| `Solver.constraint_equality / constraint_inequality` | residuals the solver drives to zero (balances) and bounds |
| `Solver.objective(...)` | a quantity to minimize or maximize |
| `Time.integrate(var, rate)` | make a variable a dynamic state for transient runs |
| **combos** | the selection lever — tag declarations and pick them at `run(combos=...)` to **re-scope** one model (design vs. off-design, …) without rewriting it |
| `run(**inputs)` → `Result` | assemble the active problem, solve, sweep an input matrix, return tabulated results |
| `Analysis` | wrap a system with post-processing, plots, and reporters |

**Problem types this expresses:** feed-forward design studies · balanced networks (jet engines, pipe
nets, drivetrains) · coupled multidisciplinary fixed points (aero↔structure) · multipoint / multi-site
design · steady **and** transient (integrated states) · optimization under constraints. The same
residuals, re-partitioned by combos, cover design and off-design.

## Hello world

Grow a tank until its volume hits a target — the solver changes `height` until `volume == target_volume`:

```python
from engforge import forge, System, field
from engforge.attr_solver import Solver
from engforge.properties import system_property

@forge
class Tank(System):
    area: float = field(default=2.0)            # X: set by you
    target_volume: float = field(default=10.0)  # X: the goal
    height: float = field(default=1.0)          # X: the solver changes this

    @system_property
    def volume(self) -> float:                  # Y = F(X)
        return self.area * self.height

    grow = Solver.declare_var("height")                       # free variable
    grow.add_var_constraint(0.0, "min")                       # height >= 0
    fill = Solver.constraint_equality("volume", "target_volume")  # drive volume - target -> 0

tank = Tank()
result = tank.run(revert_last=False, revert_every=False)
print(tank.height, tank.volume)   # 5.0  10.0
result.dataframe                  # the run, tabulated
```

Change `Solver.constraint_equality` to a `Solver.objective`, add a slot for a pump, sweep
`area=[1,2,3]`, or declare a `Time.integrate` to make it transient — the model stays the same; what
you ask of it changes.

## Install

```bash
pip install git+https://github.com/Ottermatics/engforge.git
```

## Learn more

- **[Guide](docs/GUIDE.md)** — running models: input matrices and jointly-varying groups, `save_mode`,
  the `Result` object, change events, and the command-line entry point.
- **[Tutorials](docs/TUTORIALS.md)** — step-by-step walkthroughs end to end.
- **Runnable examples** — by path:

  | file | shows |
  |---|---|
  | [`examples/air_filter.py`](examples/air_filter.py) | throttle sweep, signals, a solver balance, plotting |
  | [`examples/spring_mass.py`](examples/spring_mass.py) | dynamics via `Time.integrate`; `simulate` vs `sim_matrix` |
  | [`examples/transient_control.py`](examples/transient_control.py) | a PID controller as a component; grouped inputs |
  | [`examples/field_area.py`](examples/field_area.py) | constrained optimization with objectives |
  | [`examples/design_offdesign.py`](examples/design_offdesign.py) | one model, re-scoped by combos |

- **[Design (engforge 1.0)](docs/ENGFORGE_1.0.md)** — the forward-looking engine architecture and the
  record of what has landed.
- **[API docs](https://ottermatics.github.io/engforge/)**
