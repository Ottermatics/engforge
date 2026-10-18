# EngForge Guide — running models

This guide covers the mechanics of running a model: the input matrix, what a run returns, how rows
are recorded, change events, and the command-line entry point. For the concepts and a hello-world,
see the [README](../README.md); for end-to-end walkthroughs, [TUTORIALS.md](TUTORIALS.md).

## `run()` and the input matrix

`System.run(**inputs)` assembles the active problem (per the current combos), solves it, and returns
a `Result`. Inputs can sweep:

- **Plain keys cross as an outer product:** `run(x=[1, 2, 3], y=[1, 2])` → 6 points.
- **A comma key groups subkeys that vary jointly** (rows are zipped, not crossed), then crosses the
  remaining factors — a *partial* factorial:

  ```python
  res = system.run(**{"kp,ki": [(2, 1), (8, 4)], "x": [1, 2, 3]})
  len(res.cases)   # 6 — two tuned gain-sets × three x-values, NOT 12
  ```

  Subkeys may be dotted, tying a system field to a component field:

  ```python
  system.run(**{"x,comp.y": [(1, 10), (2, 20)]})   # 2 points: (1,10) and (2,20)
  ```

- **A `sequence=[{...}, {...}]`** channel is crossed innermost and overrides on key collision.

The expansion is pure stdlib in `engforge/inputs.py`; `count_points(...)` is the dry run before an
expensive sweep.

By default a run records each point's state and then reverts the live objects to where they started
(`revert_last` / `revert_every`), so multiple systems can share a component. Pass
`revert_last=False, revert_every=False` to leave the solved state on the objects.

## What `run()` returns

`run()` returns a `Result` (`engforge.results.Result`) — the explicit owner of the run's rows:

```python
res = system.run(x=[1, 2, 3])

res.cases                 # list[Case], one per input point
res.cases[0].dataframe    # just that point's rows
res.dataframe             # the whole run, combined
```

`Result` is also a `Mapping` over two reserved keys so the legacy shape keeps working:

```python
res["output"][0]          # the solver output of input-point 0
res["input_sets"][0]      # -> {'x': 1}
```

`Result.__getitem__` is deliberately **not** column-like (`res["x"]` raises `KeyError`, because
`output` is a legitimate variable name in real models) — reach for `res.dataframe["x"]`.

## `save_mode` — what each row holds

`run()` / `simulate()` take `save_mode=` to choose what each recorded row captures:

| mode | records |
|---|---|
| `vars` | the run's variable refs only |
| `nums` | numeric attributes + properties |
| `all` | every system reference (default) |
| `prob` | only the solver's problem variables (objective refs excluded) |
| `sparse` | the same refs as `all`, but each row stores only the **delta** vs the prior row |

`sparse` pays off on transients. Read sparse rows back through `case.dense_rows` /
`case.dense_dataframe`, which forward-fill the deltas.

## Transients

Two entry points, and they differ:

- **`simulate(dt, endtime, ...)`** runs **one** transient on a copy of the system and returns the
  `DataFrame`. It does not sweep.
- **`sim_matrix(dt=…, endtime=…, **inputs)`** is the sweep — it expands the input matrix and returns
  a `Result`.

Make a variable a dynamic state with `Time.integrate(var, rate)`.

## Change events

Attribute writes, slot swaps, and collection mutations emit events on a process-wide bus
(`engforge/events.py`). Emitters are free when nobody listens; `watch()` scopes a subscription:

```python
from engforge.events import VALUE_CHANGED, watch

seen = []
with watch(VALUE_CHANGED, lambda inst, name, old, new: seen.append((name, old, new))):
    system.run(x=[1.0, 2.0])
```

| event | payload |
|---|---|
| `VALUE_CHANGED` | `(instance, name, old, new)` |
| `SLOT_CHANGED` | `(instance, name, old, new)` — a slot's component was swapped |
| `COLLECTION_CHANGED` | `(collection, key, item)` |

## Command line

`Analysis.cli()` is an argparse entry point — put it at the bottom of an analysis script:

```python
if __name__ == "__main__":
    MyAnalysis(system=MySystem()).cli()
```

`--input-json` accepts a flat document (`{"x": [1, 2], "kp,ki": [[2, 1], [8, 4]]}`) or a sectioned one
(`{"inputs": {...}, "settings": {...}, "sequence": [...]}`). Precedence: file → `--set KEY=VALUE`
(JSON-parsed, repeatable) → `sequence`. `--list-inputs` prints the system's fields and slots.

## Reporting

`Analysis` wraps a top-level system and saves data for each interval through table and plot reporters
(CSV, Excel, and disk-stored plots). See [`examples/air_filter.py`](../examples/air_filter.py) for a
reporter setup.

## Imports

The newer names import from their modules:

```python
from engforge.results import Result, Case
from engforge.state import SystemState
from engforge.inputs import count_points
from engforge import events
```
