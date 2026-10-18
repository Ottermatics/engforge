# engforge Tutorials

Task-oriented walkthroughs of the run/result/state/event layer. Every code block in this
document was executed against branch `engforge_1.0`, and every output block is the real
output of that code. The suite behind it: `Ran 129 tests ... OK`
(`conda run -n pace python -m unittest discover -s engforge/test -p "test_*.py"`).

**How to run anything here**

```bash
cd /path/to/engforge
conda run -n pace python my_script.py
```

**Two conventions, so the listings stay readable**

1. Each snippet assumes the shared model below is importable (save it next to your script
   as `heatsink.py`).
2. engforge logs at INFO by default and the transient integrator prints the raw scipy
   `OdeResult`. Every script shown here starts with

   ```python
   from engforge import change_all_log_levels
   change_all_log_levels(50)          # errors only
   ```

   and the outputs below have the remaining log lines and the solver's progress dots
   (`...`) stripped. Nothing else is edited.

**Contents**

| # | Tutorial |
|---|---|
| [T1](#t1--running-a-system-and-what-you-get-back) | Running a system, and what you get back |
| [T2](#t2--input-matrices-that-are-not-outer-products) | Input matrices that are not outer products |
| [T3](#t3--reading-results-result-case-and-where-tabulation-went) | Reading results: `Result`, `Case`, and where tabulation went |
| [T4](#t4--recording-less-save_mode) | Recording less: `save_mode` |
| [T5](#t5--explicit-state-systemstate-trees-and-episodes) | Explicit state: `SystemState`, trees, and episodes |
| [T6](#t6--change-events) | Change events |
| [T7](#t7--driving-a-study-from-the-command-line) | Driving a study from the command line |
| [T8](#t8--parallel-problem-contexts-what-works-and-what-does-not) | Parallel problem contexts: what works and what does not |
| [+](#transients-simulate-vs-sim_matrix) | Addendum: `simulate` vs `sim_matrix` |
| [—](#import-paths-cheat-sheet) | Import-paths cheat sheet · [migration note](#migrating-from-0x) · [examples index](#where-to-look-next) |

---

## The shared model

You are sizing a forced-air heatsink for a power stage. A fin extrusion gives you area;
airflow gives you a convection coefficient; the junction temperature is whatever makes the
heat balance close. That last part is a solve, not a formula — which is the whole reason to
write it as a `System`.

```python
"""heatsink.py — the shared tutorial model: a forced-air heatsink."""

from engforge import System, Component, forge, Slot, Solver, Time
from engforge.properties import system_property
from attrs import field


@forge
class Fins(Component):
    """an extruded fin stack, 50 mm deep"""

    n: float = field(default=12.0)          # fin count
    height: float = field(default=0.020)    # m
    thickness: float = field(default=0.0010)  # m

    @system_property
    def area(self) -> float:
        """convective area, both sides, m2"""
        return 2 * self.n * self.height * 0.050

    @system_property
    def mass(self) -> float:
        """aluminium fin mass, kg"""
        return 2700.0 * self.n * self.height * self.thickness * 0.050


@forge
class HeatSink(System):
    """solve for the junction temperature that balances the heat load"""

    q_watt: float = field(default=25.0)      # W dissipated
    air_vel: float = field(default=2.0)      # m/s
    t_air: float = field(default=30.0)       # degC
    t_junction: float = field(default=60.0)  # degC — solved for

    fins: Fins = Slot.define(Fins)

    temp = Solver.declare_var("t_junction", combos="thermal")
    temp.add_var_constraint(0.0, "min", combos="thermal")
    balance = Solver.constraint_equality("q_balance", 0, combos="thermal")

    @system_property
    def h_conv(self) -> float:
        """W/m2K, forced-convection fit"""
        return 10.0 + 12.0 * self.air_vel**0.6

    @system_property
    def q_removed(self) -> float:
        return self.h_conv * self.fins.area * (self.t_junction - self.t_air)

    @system_property
    def q_balance(self) -> float:
        return self.q_watt - self.q_removed


@forge
class SinkWarmup(System):
    """the same sink, warming up: lumped capacitance"""

    q_watt: float = field(default=25.0)
    air_vel: float = field(default=2.0)
    t_air: float = field(default=30.0)
    t_metal: float = field(default=30.0)
    c_therm: float = field(default=6.0)      # J/K

    fins: Fins = Slot.define(Fins)

    warm = Time.integrate("t_metal", "dTdt")

    @system_property
    def h_conv(self) -> float:
        return 10.0 + 12.0 * self.air_vel**0.6

    @system_property
    def q_out(self) -> float:
        return self.h_conv * self.fins.area * (self.t_metal - self.t_air)

    @system_property
    def dTdt(self) -> float:
        return (self.q_watt - self.q_out) / self.c_therm
```

`combos="thermal"` tags the solver variable and the constraint. Nothing solves unless you
ask for that combo — that is why every `run()` below passes `combos="thermal"`.

---

## T1 — Running a system, and what you get back

**The situation.** You want one number: how hot does the junction get at 1, 2 and 4 m/s?

### One point

```python
from engforge import change_all_log_levels
change_all_log_levels(50)
from heatsink import HeatSink, Fins

hs = HeatSink(fins=Fins())
res = hs.run(combos="thermal")
print(res)
print("solved t_junction  :", round(res.dataframe["t_junction"][0], 3))
print("system t_junction  :", hs.t_junction, "(reverted)")
```

```
Result(1 cases)
solved t_junction  : 66.953
system t_junction  : 60.0 (reverted)
```

Read that last line carefully: **`run()` puts the system back where it found it.** The
answer lives in the result, not on the object. (`revert_last=True` is the default; pass
`revert_last=False` if you want the solved state left in place.)

### A sweep

```python
res = hs.run(air_vel=[1.0, 2.0, 4.0], combos="thermal")

print("type        :", type(res).__name__, "|", res)
print("Mapping keys:", list(res.keys()))
print("input_sets  :", res["input_sets"])
print("output[0]   :", list(res["output"][0]))
print("  success   :", res["output"][0]["success"], "Xans", res["output"][0]["Xans"])
print("cases       :", res.cases)
print("df.shape    :", res.dataframe.shape)
print("case[0].df  :", res.cases[0].dataframe.shape)
print("last_result :", hs.last_result is res)
print("hs.dataframe:", hs.dataframe.shape)
print(res.dataframe[["air_vel", "h_conv", "t_junction", "q_removed"]].to_string())
try:
    res["t_junction"]
except KeyError as e:
    print("res['t_junction'] ->", type(e).__name__, e)
```

```
type        : Result | Result(3 cases)
Mapping keys: ['output', 'input_sets']
input_sets  : {0: {'air_vel': 1.0}, 1: {'air_vel': 2.0}, 2: {'air_vel': 4.0}}
output[0]   : ['Xstart', 'Ystart', 'Xans', 'success', 'Yobj', 'Ycon', 'ans', 'weights', 'constraints']
  success   : True Xans {'t_junction': 77.348484848484}
cases       : [Case(1 rows, inputs=['air_vel']), Case(1 rows, inputs=['air_vel']), Case(1 rows, inputs=['air_vel'])]
df.shape    : (3, 17)
case[0].df  : (1, 17)
last_result : True
hs.dataframe: (3, 17)

   air_vel     h_conv  t_junction  q_removed
0      1.0  22.000000   77.348485       25.0
1      2.0  28.188599   66.953474       25.0
2      4.0  37.568761   57.726937       25.0

res['t_junction'] -> KeyError 't_junction'
```

### What a `Result` is

`run()` returns `engforge.results.Result`. It is a `collections.abc.Mapping` over exactly
two reserved keys, plus an object interface:

| access | meaning |
|---|---|
| `res["output"][i]` | case *i*'s solver return (the `Xstart/Xans/success/...` dict) |
| `res["input_sets"][i]` | case *i*'s input point |
| `res.cases` | the `Case` list, in run order |
| `res.dataframe` | every case's rows, one frame, column-formatted |
| `res.numeric_rows` | the same rows as a list of dicts, list/dict/tuple values dropped |
| `system.last_result` | the same object, reachable later |

`res["t_junction"]` raising `KeyError` is **deliberate**, not an oversight. The keys are
pinned to `('output', 'input_sets')` so a model that happens to have a property called
`output` can never shadow the run contract. Columns come from a dataframe; the mapping is
the legacy `{"output": ..., "input_sets": ...}` shape and nothing more.

> **Docstring warning.** `solver.py:157` and `solveable.py:354` still say `run()` returns
> "the system" / "system or list of systems". They predate `Result`. Trust this page.

---

## T2 — Input matrices that are not outer products

**The situation.** Your extruder sells two stock profiles: 12 fins at 1.0 mm, or 24 fins at
0.6 mm. Those are *two options*, not four. If you sweep `fins.n` and `fins.thickness`
independently you will also build a 12-fin/0.6 mm sink and a 24-fin/1.0 mm sink — parts
that do not exist — and pay for 12 solves instead of 6.

A **comma key** says "these vary together".

### Dry-run the matrix before you launch it

`engforge.inputs` is pure stdlib, so you can size a sweep without touching a system.

```python
from engforge.inputs import (
    is_group_key, split_group_key, expand_point_factors, count_points,
)

STOCK = [(12.0, 0.0010), (24.0, 0.0006)]        # extrusion A, extrusion B

singles = {"air_vel": [1.0, 2.0, 4.0]}
groups = [(["fins.n", "fins.thickness"], STOCK)]

print("grouped  :", count_points(singles, groups), "points")
print("crossed  :", count_points({**singles, "fins.n": [12.0, 24.0],
                                  "fins.thickness": [0.0010, 0.0006]}, []), "points")
for i, p in enumerate(expand_point_factors(singles, groups)):
    print(f"  {i}: {p}")
```

```
grouped  : 6 points
crossed  : 12 points
  0: {'air_vel': 1.0, 'fins.n': 12.0, 'fins.thickness': 0.001}
  1: {'air_vel': 1.0, 'fins.n': 24.0, 'fins.thickness': 0.0006}
  2: {'air_vel': 2.0, 'fins.n': 12.0, 'fins.thickness': 0.001}
  3: {'air_vel': 2.0, 'fins.n': 24.0, 'fins.thickness': 0.0006}
  4: {'air_vel': 4.0, 'fins.n': 12.0, 'fins.thickness': 0.001}
  5: {'air_vel': 4.0, 'fins.n': 24.0, 'fins.thickness': 0.0006}
```

Six, not twelve. The group is **one factor** of zipped rows that crosses the other factors
— a partial factorial.

```python
print("is_group_key('fins.n,fins.thickness') :", is_group_key("fins.n,fins.thickness"))
print("is_group_key('air_vel')               :", is_group_key("air_vel"))
print("split_group_key('air_vel, fins.n')    :", split_group_key("air_vel, fins.n"))
```

```
is_group_key('fins.n,fins.thickness') : True
is_group_key('air_vel')               : False
split_group_key('air_vel, fins.n')    : ['air_vel', 'fins.n']
```

Subkeys may be dotted (`"air_vel, fins.n"` mixes a system field and a component field), are
whitespace-tolerant, must be unique, and every row must have exactly one value per subkey —
otherwise you get an assertion naming the offending row.

### On the system

Because the key contains a comma you must pass it through `**{...}`:

```python
hs = HeatSink(fins=Fins())
res = hs.run(
    **{"fins.n,fins.thickness": STOCK, "air_vel": [1.0, 2.0, 4.0]},
    combos="thermal",
)
print(res)
print("input_sets[0]:", res["input_sets"][0])
cols = ["fins.n", "fins.thickness", "air_vel", "fins.area", "fins.mass", "t_junction"]
print(res.dataframe[cols].to_string())
```

```
Result(6 cases)
input_sets[0]: {'air_vel': 1.0, 'fins.n': 12.0, 'fins.thickness': 0.001}
   fins.n  fins.thickness  air_vel  fins.area  fins.mass  t_junction
0    12.0          0.0010      1.0      0.024    0.03240   77.348485
1    24.0          0.0006      1.0      0.048    0.03888   53.674242
2    12.0          0.0010      2.0      0.024    0.03240   66.953476
3    24.0          0.0006      2.0      0.048    0.03888   48.476737
4    12.0          0.0010      4.0      0.024    0.03240   57.726938
5    24.0          0.0006      4.0      0.048    0.03888   43.863469
```

That table is the answer to the real question: extrusion B doubles the area for 20 % more
mass and buys ~24 °C at 1 m/s. The group key is also what makes the comparison
*well-formed* — each row is a buildable part.

For contrast, the independent sweep:

```python
hs2 = HeatSink(fins=Fins())
res2 = hs2.run(
    **{"fins.n": [12.0, 24.0], "fins.thickness": [0.0010, 0.0006],
       "air_vel": [1.0, 2.0, 4.0]},
    combos="thermal",
)
print(res2, "->", res2.dataframe.shape[0], "rows")
```

```
Result(12 cases) -> 12 rows
```

### `sequence=` — the innermost factor, and it wins ties

`sequence` is a list of override dicts applied inside every other factor:

```python
hs3 = HeatSink(fins=Fins())
res3 = hs3.run(
    air_vel=[1.0, 4.0],
    sequence=[{"q_watt": 25.0}, {"q_watt": 40.0}],
    combos="thermal",
)
print(res3)
print(res3.dataframe[["air_vel", "q_watt", "t_junction"]].to_string())
```

```
Result(4 cases)
   air_vel  q_watt  t_junction
0      1.0    25.0   77.348485
1      1.0    40.0  105.757576
2      4.0    25.0   57.726937
3      4.0    40.0   74.363100
```

Note the ordering: `air_vel` is the outer loop, the sequence row varies fastest. And when a
sequence row names a key you also swept, the row **overwrites** it:

```python
for i, p in enumerate(expand_point_factors({"air_vel": [1.0, 2.0]}, [],
                                           [{"air_vel": 9.0}])):
    print(f"  {i}: {p}")
```

```
  0: {'air_vel': 9.0}
  1: {'air_vel': 9.0}
```

Two points, both 9.0. The point *count* still comes from the sweep; the values come from
the sequence. That is intentional (legacy `_iterate_input_matrix` semantics) and it is the
single easiest way to silently waste a sweep — if two points come back identical, check for
a colliding sequence key first.

### Choosing between the three

| you want | use |
|---|---|
| every combination of independent knobs | plain keys — `x=[...], y=[...]` |
| a fixed set of *configurations* (a part number, a tuned gain set, an operating point) crossed with knobs | comma key — `"a,b": [(…), (…)]` |
| the same override list applied inside everything else (load cases, duty points) | `sequence=[{…}, {…}]` |

Call `count_points()` first on anything that will run for more than a minute.

See `examples/transient_control.py` for the same pattern on tuned PID gain sets, and
`examples/design_offdesign.py` / `examples/balanced_network.py` for design-point and
network variants.

---

## T3 — Reading results: `Result`, `Case`, and where tabulation went

**The situation.** You ran a sweep and now you need the rows — per point, not just one
blended frame. In 0.x, rows accumulated in a dict on the context singleton and transient
cases keyed by *time* overwrote each other. Now each input point owns its rows.

### `Case` on its own

```python
from engforge.results import Case, Result

c = Case(inputs={"air_vel": 2.0})
print("stored:", c.record({"t": 60.0, "q": 25.0}))
print("stored:", c.record({"t": 60.0, "q": 25.0}, changed=False))
print("stored:", c.record({"t": 61.0, "q": 25.0}, force=True))
print(c)
print(c.dataframe.to_string())
```

```
stored: True
stored: False
stored: True
Case(2 rows, inputs=['air_vel'])
      t     q  index
0  60.0  25.0      0
1  61.0  25.0      1
```

`Case.record` holds the whole dedup policy in one line: store if `force`, or if this is the
first row, or if the caller reports `changed`. Returns whether it stored. `index` defaults
to an auto-incrementing counter and is written into the row.

### `Result` owns the cases

```python
r = Result()
a = r.new_case({"leg": "A"})
b = r.new_case({"leg": "B"})
a.record({"x": 1.0})
b.record({"x": 2.0})
print(r, "| cases:", r.cases)
print("r['input_sets'] :", r["input_sets"])
print("r.numeric_rows  :", r.numeric_rows)
print(r.dataframe.to_string())
```

```
Result(2 cases) | cases: [Case(1 rows, inputs=['leg']), Case(1 rows, inputs=['leg'])]
r['input_sets'] : {0: {'leg': 'A'}, 1: {'leg': 'B'}}
r.numeric_rows  : [{'x': 1.0, 'index': 0}, {'x': 2.0, 'index': 0}]
     x  index
0  1.0      0
1  2.0      0
```

`index` restarts per case — it is a *within-case* row number, not a global one. Group the
combined frame by an input column, not by `index`.

### On a real run

```python
hs = HeatSink(fins=Fins())
res = hs.run(air_vel=[1.0, 2.0, 4.0], combos="thermal")

print("res.cases            :", len(res.cases))
print("res.dataframe        :", res.dataframe.shape)
print("per-case shapes      :", [c.dataframe.shape for c in res.cases])
print("case[1].inputs       :", res.cases[1].inputs)
print("case[1].rows keys    :", list(res.cases[1].rows))
print("hs.last_result is res:", hs.last_result is res)
print("hs.dataframe shape   :", hs.dataframe.shape)
print("hs.last_context.dataframe shape:", hs.last_context.dataframe.shape)
print("ctx.data (active case):", len(hs.last_context.data), "rows")
print("ctx.numeric_data (run):", len(hs.last_context.numeric_data), "rows")
```

```
res.cases            : 3
res.dataframe        : (3, 17)
per-case shapes      : [(1, 17), (1, 17), (1, 17)]
case[1].inputs       : {'air_vel': 2.0}
case[1].rows keys    : [0]
hs.last_result is res: True
hs.dataframe shape   : (3, 17)
hs.last_context.dataframe shape: (3, 17)
ctx.data (active case): 1 rows
ctx.numeric_data (run): 3 rows
```

### Where tabulation actually lives now

| you reach for | you get | note |
|---|---|---|
| `system.last_result` | the `Result` | **the owner.** Start here. |
| `result.cases[i].dataframe` | one input point's rows | the per-case frame |
| `result.dataframe` | the whole run | column-formatted via the system |
| `system.dataframe` | the whole run | a three-level fallback *shim* (`tabulation.py:85`): last context's frame → a patched frame → empty |
| `ProblemExec.numeric_data` | the whole run's rows | |
| `ProblemExec.data` | **the active case's** rows | one letter from `numeric_data`, very different meaning |

`ProblemExec.data` vs `.numeric_data` is the trap: in the output above, `.data` is 1 row
and `.numeric_data` is 3. If you used `.data` in 0.x to get a whole run, you now get the
last point only.

`TabulationMixin`'s module docstring and `CLAUDE.md`'s "Data Flow and Tabulation" section
still describe the old automatic-capture model. The mechanism is now
`ProblemExec.output_state` → `Case.record` → `Result`.

### Recording your own rows

Sometimes the number you want is not a `system_property` — it belongs to the *study*, not
the part. Open a context, drive the inputs yourself, and inject it:

```python
from engforge import ProblemExec

hs2 = HeatSink(fins=Fins())
with ProblemExec(hs2, {}, level_name="scan", save_mode="all") as px:
    for v in (1.0, 2.0, 4.0):
        px.begin_case({"air_vel": v})
        hs2.air_vel = v
        px.save_data(force=True, fan_watt=0.6 * v**3)
    out = px.result
    print(out, "->")
    print(out.dataframe[["air_vel", "h_conv", "q_balance", "fan_watt"]].to_string())
    px.exit_to_level("scan", revert=True)
```

```
Result(3 cases) ->
   air_vel     h_conv  q_balance  fan_watt
0      1.0  22.000000   9.160000       0.6
1      2.0  28.188599   4.704209       4.8
2      4.0  37.568761  -2.049508      38.4
```

Three things to copy from that:

- `px.begin_case({...})` opens a new `Case`; everything recorded afterwards lands in it.
  (It also drops a pristine, empty default case so you do not accumulate blank frames.)
- `save_data(force=True, **add_data)` merges your keyword values into the row. Keys are
  lowercased by the column formatter — `margin_C` arrives as `margin_c`.
- `exit_to_level(name, revert=True)` puts the system back.

This is exactly the mechanism `Analysis.run` uses for its own callback
(`analysis.py:68`), and `Case.record_sparse` is the same door for delta rows (T4).

---

## T4 — Recording less: `save_mode`

**The situation.** A 3-point steady sweep stores 51 values. The same model as a 10-second
transient stores 95,114. At some point you stop wanting every string column of every
component on every row.

`save_mode` picks the ref set each row is built from. Five modes:
`vars`, `nums`, `all`, `prob`, `sparse` (`problem_context.py:150`). It is a run kwarg.

```python
for mode in ("vars", "nums", "all", "prob", "sparse"):
    hs = HeatSink(fins=Fins())
    res = hs.run(air_vel=[1.0, 2.0, 4.0], combos="thermal", save_mode=mode)
    df = res.dataframe
    stored = sum(len(r) for c in res.cases for r in c.rows.values())
    print(f"{mode:>7}: df={str(df.shape):<9} stored={stored:<4} cols={list(df.columns)}")
```

```
   vars: df=(3, 2)    stored=6    cols=['t_junction', 'index']
   nums: df=(3, 15)   stored=45   cols=['q_watt', 'air_vel', 't_air', 't_junction', 'fins.n', 'fins.height', 'fins.thickness', 'h_conv', 'q_removed', 'q_balance', 'converged', 'run_id', 'fins.area', 'fins.mass', 'index']
    all: df=(3, 17)   stored=51   cols=['name', 'q_watt', 'air_vel', 't_air', 't_junction', 'fins.name', 'fins.n', 'fins.height', 'fins.thickness', 'h_conv', 'q_removed', 'q_balance', 'converged', 'run_id', 'fins.area', 'fins.mass', 'index']
   prob: df=(3, 2)    stored=6    cols=['t_junction', 'index']
 sparse: df=(3, 17)   stored=51   cols=['name', 'q_watt', 'air_vel', 't_air', 't_junction', 'fins.name', 'fins.n', 'fins.height', 'fins.thickness', 'h_conv', 'q_removed', 'q_balance', 'converged', 'run_id', 'fins.area', 'fins.mass', 'index']
```

| mode | ref set | on this model | use it when |
|---|---|---|---|
| `all` | every system reference (default) | 17 cols | you don't know yet what you'll plot |
| `nums` | attribute + property refs, flattened | 15 cols — `all` minus the two `name` strings | you want a purely numeric frame |
| `vars` | the active variable refs | 2 cols | you only care about what moved |
| `prob` | the solver's problem variables (settable ones) | 2 cols | large models, many sweep points, you only want the solve answer |
| `sparse` | same refs as `all`, stored as deltas | 17 cols | **transients** |

Two things the mode names don't tell you:

- `prob` deliberately **excludes** objective refs (`sesh.Yref`). A compiled objective is a
  control-flow wrapper; calling one outside the solver loop re-enters the solver and raises
  `ProblemExitAtLevel`. If you need the objective's value as a column, record the
  underlying property with `all` or `nums`.
- `sparse` cannot compress a single-row case — above, it is byte-for-byte `all`. It only
  pays on multi-row cases, which in practice means transients.

### `sparse` on a transient

`SinkWarmup` integrated for 10 s, recorded both ways:

```python
from heatsink import SinkWarmup, Fins

cases = {}
for mode in ("all", "sparse"):
    sw = SinkWarmup(fins=Fins())
    df = sw.simulate(dt=0.01, endtime=10.0, save_mode=mode)
    case = sw.last_result.cases[-1]
    stored = sum(len(r) for r in case.rows.values())
    print(f"{mode:>7}: rows={len(case.rows):<4} stored_values={stored:<6} "
          f"df={df.shape} nan={int(case.dataframe.isna().sum().sum()):<5} "
          f"dense={case.dense_dataframe.shape} "
          f"dense_nan={int(case.dense_dataframe.isna().sum().sum())}")
    cases[mode] = case
```

```
    all: rows=5006 stored_values=95114  df=(5006, 19) nan=10012 dense=(5006, 19) dense_nan=10012
 sparse: rows=5006 stored_values=24044  df=(5006, 19) nan=71072 dense=(5006, 19) dense_nan=10012
```

**95,114 → 24,044 stored values: 75 % fewer**, same 5,006 rows, same 19 columns.

The stored rows show why — row 0 is the full baseline, everything after it is a delta:

```python
for i in sorted(cases["sparse"].rows)[:4]:
    print("  ", i, cases["sparse"].rows[i])
```

```
   0.0 {'name': 'sinkwarmup-formal-radius', 'q_watt': 25.0, 'air_vel': 2.0, 't_air': 30.0, 't_metal': 30.0, 'c_therm': 6.0, 'fins.name': 'fins-pink-inference', 'fins.n': 12.0, 'fins.height': 0.02, 'fins.thickness': 0.001, 'h_conv': 28.188598798124776, 'q_out': 0.0, 'dTdt': 4.166666666666667, 'converged': None, 'run_id': None, 'fins.area': 0.024, 'fins.mass': 0.032400000000000005, 'time': 0.0, 'index': 0.0}
   0.002 {'t_metal': 30.008333333333333, 'q_out': 0.005637719759624634, 'dTdt': 4.165727046706729, 'time': 0.002, 'index': 0.002}
   0.003 {'t_metal': 30.01249788585509, 'q_out': 0.008455149364653627, 'dTdt': 4.165257475105891, 'time': 0.003, 'index': 0.003}
   0.008 {'t_metal': 30.033318307889676, 'q_out': 0.022540713929627285, 'dTdt': 4.162909881011729, 'time': 0.008, 'index': 0.008}
```

(Row keys are times; `name` values are the auto-generated instance names, so they differ
run to run.)

### Reading it back: `dataframe` vs `dense_dataframe`

```python
print("sparse .dataframe       NaN:", int(cases["sparse"].dataframe.isna().sum().sum()))
print("sparse .dense_dataframe NaN:", int(cases["sparse"].dense_dataframe.isna().sum().sum()))
print("all    .dataframe       NaN:", int(cases["all"].dataframe.isna().sum().sum()))
```

```
sparse .dataframe       NaN: 71072
sparse .dense_dataframe NaN: 10012
all    .dataframe       NaN: 10012
```

- `Case.dataframe` renders the stored rows as-is: full of holes. Useful for inspecting what
  was actually recorded, useless for plotting.
- `Case.dense_dataframe` forward-fills the deltas and lands on **exactly** the NaN count of
  the `all` run. `Result` has no `dense_*` accessor — forward-filling is per-case by
  definition, so go through the cases.

Column-by-column, the dense sparse frame equals the `all` frame (max abs diff `0.0`) for
every numeric column — `t_metal`, `q_out`, `dTdt`, `h_conv`, `fins.*`, `q_watt`, `air_vel`,
`t_air`, `c_therm`, `index` — with one exception.

> **The `time` column caveat.** `time` differs on **1000 of 5006 rows**, by at most
> 1.11e-3 (one integrator substep):
>
> ```
> rows where time differs: 1000 of 5006
>   row 5 all.time 0.01 sparse.time 0.008888888888888889 | index 0.01 0.01
>   row 10 all.time 0.02 sparse.time 0.01888888888888889 | index 0.02 0.02
> ```
>
> Cause: rows are keyed by time, and at each `t_eval` output point the row index gets
> written twice — **1001 of 5006 indices are written twice** (6007 `record` calls for 5006
> stored rows). Under `all` the second write replaces the row wholesale; under `sparse` the
> second write stores a delta that omits `time` when it matches the previously recorded
> dense state, so the forward fill carries the earlier substep's clock. If you need an
> exact time axis from a sparse run, use the row `index` (identical in both, max abs diff
> `0.0`) rather than the `time` column.

### Rules of thumb

- Steady sweeps: leave it at `all` unless the frame is genuinely too wide, then `prob`.
- Transients you intend to keep: `sparse`, and read `dense_dataframe`.
- Anything feeding a numeric pipeline: `nums`.
- Never mix modes inside one comparison — the stored key sets differ.

---

## T5 — Explicit state: `SystemState`, trees, and episodes

**The situation.** You want to try a configuration, look at it, and get back *exactly* where
you were — including slot components, including the clock. In 0.x this was three implicit
dicts (`record_state`, `x_start`, `temp_state`) and restore logic smeared across the
context. Now it is one object with four operations.

```python
SystemState(values: dict = None, time: float = 0.0)      # __slots__ = ('values','time')

SystemState.capture(context) -> SystemState              # classmethod
state.apply(context, restore_time: bool = False)
state.updated(point: dict) -> SystemState                # a NEW state
state.copy() -> SystemState
state.diff(other) -> {key: (mine, theirs)}
```

### Capture, derive, apply, undo

```python
from engforge import ProblemExec
from engforge.state import SystemState

hs = HeatSink(fins=Fins())
with ProblemExec(hs, {}, level_name="inspect") as px:
    base = SystemState.capture(px)
    print("base       :", base)
    print("keys       :", sorted(base.values))
    print("t_junction :", base.values["t_junction"], "| air_vel", base.values["air_vel"])

    hot = base.updated({"air_vel": 6.0, "fins.n": 30.0})
    print("new object :", hot.values is base.values)
    print("diff       :", base.diff(hot))

    hot.apply(px)
    print("after apply: air_vel", hs.air_vel, "fins.n", hs.fins.n,
          "h_conv", round(hs.h_conv, 3))

    base.apply(px)
    print("after undo : air_vel", hs.air_vel, "fins.n", hs.fins.n)
    print("round trip :", base.diff(SystemState.capture(px)))
    px.exit_to_level("inspect", revert=True)
```

```
base       : SystemState(10 refs, t=0)
keys       : ['air_vel', 'fins', 'fins.height', 'fins.n', 'fins.name', 'fins.thickness', 'name', 'q_watt', 't_air', 't_junction']
t_junction : 60.0 | air_vel 2.0
new object : False
diff       : {'air_vel': (2.0, 6.0), 'fins.n': (12.0, 30.0)}
after apply: air_vel 6.0 fins.n 30.0 h_conv 45.162
after undo : air_vel 2.0 fins.n 12.0
round trip : {}
```

Four facts worth internalizing:

1. `capture` reads **every settable ref through the session**, including dotted component
   fields and the slot bindings themselves (`fins` is in the key list).
2. `updated()` returns a **new** node — `hot.values is base.values` is `False`. States are
   a persistent tree, not a mutable cursor. That is what makes branching safe.
3. `diff()` is your debugger for "why didn't my system come back". An empty dict is a
   clean round trip.
4. `apply` writes through `Ref.refset_input(..., fail=False)` — it is tolerant of refs that
   have disappeared, so a state captured before a slot swap won't explode after one.

### State trees for a variation study

**The situation.** Two fin-stock options, and for each you want a sweep. That is a tree:
trunk (as-built) → one branch per option → one leaf per airflow. Branch, record, come back
to the branch root, repeat; then come back to the trunk.

```python
STOCK = {"extr_A": {"fins.n": 12.0, "fins.thickness": 0.0010},
         "extr_B": {"fins.n": 24.0, "fins.thickness": 0.0006}}

hs2 = HeatSink(fins=Fins())
with ProblemExec(hs2, {}, level_name="study", save_mode="all") as px:
    root = SystemState.capture(px)
    for name, cfg in STOCK.items():
        front = root.updated(cfg)                 # branch node
        for vel in (1.0, 2.0, 4.0):
            px.begin_case({"stock": name, "air_vel": vel})
            front.updated({"air_vel": vel}).apply(px)   # leaf
            px.save_data(force=True, stock_id=name)
        front.apply(px)                           # back to the branch root
    root.apply(px)                                # back to the trunk
    res = px.result
    print(res)
    print(res.dataframe[["stock_id", "fins.n", "air_vel", "fins.area",
                         "fins.mass", "q_balance"]].to_string())
    print("round trip :", root.diff(SystemState.capture(px)))
    px.exit_to_level("study", revert=True)
```

```
Result(6 cases)
  stock_id  fins.n  air_vel  fins.area  fins.mass  q_balance
0   extr_A    12.0      1.0      0.024    0.03240   9.160000
1   extr_A    12.0      2.0      0.024    0.03240   4.704209
2   extr_A    12.0      4.0      0.024    0.03240  -2.049508
3   extr_B    24.0      1.0      0.048    0.03888  -6.680000
4   extr_B    24.0      2.0      0.048    0.03888 -15.591582
5   extr_B    24.0      4.0      0.048    0.03888 -29.099015
```

```
round trip : {}
```

Depth-2 branching round-trips exactly. Note this context did **not** run the solver, so
`q_balance` is the residual at the default `t_junction` — the point here is state handling,
not thermal design. Reach for a state tree over a grouped `run()` (T2) when the branches
need *different sweep grids*; if the grids are the same, the grouped matrix key is simpler
and gives you one frame (T8 closes on that recipe).

### Slot components are state too

```python
hs3 = HeatSink(fins=Fins())
orig = hs3.fins
with ProblemExec(hs3, {}, level_name="slots") as px:
    snap = SystemState.capture(px)
    print("captured 'fins' ->", type(snap.values["fins"]).__name__)
    hs3.fins = Fins(n=40.0)
    print("swapped   fins.n:", hs3.fins.n, "| is orig:", hs3.fins is orig)
    snap.apply(px)
    print("restored  fins.n:", hs3.fins.n, "| is orig:", hs3.fins is orig)
    px.exit_to_level("slots", revert=True)
```

```
captured 'fins' -> Fins
swapped   fins.n: 40.0 | is orig: False
restored  fins.n: 12.0 | is orig: True
```

`apply` restores the **original object identity**, not a copy of its values. That matters
for anything holding a reference to the component.

### Episodes and `restore_time`

A transient is an *episode*: it brackets itself with a `SystemState` and restores fields
**and the clock** on exit. This is the one place `restore_time=True` is used
(`dynamics.py:622-671`).

```python
sw = SinkWarmup(fins=Fins())
print("before: time", sw.time, "t_metal", sw.t_metal)
df = sw.simulate(dt=0.01, endtime=2.0)
print("after : time", sw.time, "t_metal", sw.t_metal)
print("df    :", df.shape, "| last t_metal", round(df["t_metal"].values[-1], 4),
      "| last time", df["time"].values[-1])
```

```
before: time 0 t_metal 30.0
after : time 0 t_metal 30.0
df    : (1001, 19) | last t_metal 37.4605 | last time 2.0
```

The sink warmed to 37.46 °C over 2 s and the system is back at t=0, T=30 °C. The run's
history is in the frame.

Why the flag exists at all: **solver reverts must not touch the clock.** Integration
advances time across nested context exits, so `revert_to_start` calls
`apply(..., restore_time=False)` (`problem_context.py:1904`). Only the episode bracket
passes `True`. If you write your own bracket around a transient, pass `restore_time=True`;
around a solve, don't.

> **Gotcha: `endtime` and `dt` must land on a grid.** `integrate` builds
> `np.arange(t0, endtime + dt, dt)` and hands it to `solve_ivp` as `t_eval`. Floating-point
> overshoot makes scipy reject it:
>
> ```
> endtime=10.0 dt=0.01: len=1001 last=10.0 last>endtime=False
> endtime=20.0 dt=0.01: len=2002 last=20.01 last>endtime=True
> ```
>
> `endtime=20.0, dt=0.01` raises `ValueError: Values in 't_eval' are not within 't_span'`.
> Nudge `endtime` or `dt`. Also note `dt` is clamped to `max_step_dt` (default `0.01`) with
> a warning, so `dt=0.5` does **not** give you a coarse run.

`examples/state_variations.py` is the companion example for state-tree variation studies;
`examples/spring_mass.py` is the minimal transient.

---

## T6 — Change events

**The situation.** You want to know when the model changes — to invalidate a cache, to log
what a solver actually touched, or to drive sparse recording. `engforge.events` is a
process-wide pyee bus with three events and a scoped-subscription context manager.

```python
from engforge import events
from engforge.events import watch, VALUE_CHANGED, SLOT_CHANGED, COLLECTION_CHANGED
```

| event | payload | fires on |
|---|---|---|
| `VALUE_CHANGED` | `(instance, name, old, new)` | an attrs field changed value |
| `SLOT_CHANGED` | `(instance, name, old, new)` | a slot's **component** was swapped |
| `COLLECTION_CHANGED` | `(collection, key, item)` | a collection member was set or appended |

Emission is guarded on `bus.listeners(...)`, so with nobody subscribed the cost is one
dict lookup:

```python
print("has_listeners(VALUE_CHANGED):", events.has_listeners(VALUE_CHANGED))
```

```
has_listeners(VALUE_CHANGED): False
```

### The caveat that will bite you first

```python
hs = HeatSink(fins=Fins())
seen = []
with watch(VALUE_CHANGED, lambda i, n, o, v: seen.append((n, o, v))):
    print("  has_listeners:", events.has_listeners(VALUE_CHANGED))
    print("  hs._anything_changed:", hs._anything_changed)
    hs.air_vel = 5.0
print("  seen:", seen)
```

```
  has_listeners: True
  hs._anything_changed: True
  seen: []
```

The listener is attached, the write happened, **nothing fired.** `property_changed`
(`configuration.py:160-169`) short-circuits when there is no active `ProblemExec` session
*and* the instance's `_anything_changed` is already `True` — and a freshly constructed
object starts dirty, so there is nothing new to report.

The fix is to watch from inside a session, which is where you actually want to watch:

```python
hs = HeatSink(fins=Fins())
seen = []
with ProblemExec(hs, {}, level_name="watch") as px:
    with watch(VALUE_CHANGED, lambda i, n, o, v: seen.append((i.classname, n, o, v))):
        hs.air_vel = 5.0
        hs.fins.n = 20.0
        hs.air_vel = 5.0        # unchanged -> silent
    px.exit_to_level("watch", revert=True)
print("  seen:", seen)
```

```
  seen: [('heatsink', 'air_vel', 2.0, 5.0), ('fins', 'n', 12.0, 20.0)]
```

Component writes report too (the payload's `instance` is the `Fins`, not the system), and
re-writing the same value is silent.

(If you must watch outside a session, clear the flag yourself: `obj._anything_changed =
False` immediately before the write. It is per-instance — clearing it on the system does
nothing for its components.)

### A whole run

```python
hs2 = HeatSink(fins=Fins())
seen3 = []
with watch(VALUE_CHANGED, lambda i, n, o, v: seen3.append((n, o, v))):
    hs2.run(air_vel=[1.0, 4.0], combos="thermal")
print("  events:", len(seen3))
print("  by name:", {n: sum(1 for e in seen3 if e[0] == n)
                     for n in sorted({e[0] for e in seen3})})
print("  air_vel:", [e for e in seen3 if e[0] == "air_vel"])
```

```
  events: 42
  by name: {'air_vel': 3, 't_junction': 39}
  air_vel: [('air_vel', 2.0, 1.0), ('air_vel', 1.0, 4.0), ('air_vel', 4.0, 2.0)]
```

A two-point sweep is 42 value changes: two input writes, 39 solver iterations on
`t_junction`, and one final write putting `air_vel` back to 2.0 — **the revert is an event
too.** This is the cheapest window into what a solve actually did.

### `SLOT_CHANGED` and its identity guard

```python
hs3 = HeatSink(fins=Fins())
slots = []
with ProblemExec(hs3, {}, level_name="slot") as px:
    with watch(SLOT_CHANGED, lambda i, n, o, v: slots.append((n, o.n, v.n))):
        newfins = Fins(n=36.0)
        hs3.fins = newfins
        hs3.fins = newfins      # same object -> silent
    px.exit_to_level("slot", revert=True)
print("  seen:", slots)
```

```
  seen: [('fins', 12.0, 36.0)]
```

`SLOT_CHANGED` has an identity guard (`configuration.py:210`, `if value is not cur`) that
`VALUE_CHANGED` does not. Re-assigning the same component is deliberately silent —
assignment of an unchanged component dominates the write volume during a solve.

### `COLLECTION_CHANGED`

```python
from engforge import forge
from engforge.component_collections import ComponentIterator

@forge
class FinBank(ComponentIterator):
    component_type = Fins

bank = FinBank(component_type=Fins)
coll = []
with watch(COLLECTION_CHANGED,
           lambda c, k, item: coll.append((k, type(item).__name__, item.n))):
    bank.append(Fins(n=10.0))
    bank.append(Fins(n=20.0))
    bank[0] = Fins(n=30.0)
print("  seen:", coll)
print("  current_item after mutation:", bank.current_item)
```

```
  seen: [(0, 'Fins', 10.0), (1, 'Fins', 20.0), (0, 'Fins', 30.0)]
  current_item after mutation: None
```

`ComponentIterator.append` and `ComponentIter.__setitem__` both type-check the item, reset
the reference cache (`current_item` back to `None`), and emit.

### What it is actually for

The one consumer in core today is the session itself: `ProblemExec` subscribes to
`SLOT_CHANGED`/`COLLECTION_CHANGED`, sets `comp_changed`, and rebinds its reference graph
on the next refresh (`problem_context.py:690-732`). That is what makes `full_update=False`
correct instead of fast-and-wrong.

```python
hs4 = HeatSink(fins=Fins())
with ProblemExec(hs4, {}, level_name="refresh") as px:
    px.full_update = False
    print("  listening on SLOT_CHANGED:", events.has_listeners(SLOT_CHANGED))
    print("  comp_changed            :", px.comp_changed)
    fresh = Fins(n=48.0)
    hs4.fins = fresh
    print("  after slot swap         :", px.comp_changed)
    px.min_refresh()
    print("  after min_refresh       :", px.comp_changed)
    ref = px.sesh.all_comps_and_vars["fins.n"]
    print("  ref rebound to new comp :", ref.comp is fresh, "| value", ref.value())
    px.exit_to_level("refresh", revert=True)
print("  listeners after exit    :", events.has_listeners(SLOT_CHANGED),
      events.has_listeners(COLLECTION_CHANGED))
```

```
  listening on SLOT_CHANGED: True
  comp_changed            : False
  after slot swap         : True
  after min_refresh       : False
  ref rebound to new comp : True | value 48.0
  listeners after exit    : False False
```

The swap flags `comp_changed`; `min_refresh()` consumes the flag and rebinds `fins.n` to
the new component. Note the last line: the session unhooks its own listeners on exit. `watch()` does the same
for yours — the bus is process-wide, so a listener you forget is a listener that fires
during someone else's run. Always use the context manager.

Three honest limits: the bus is **global**, not per-session; it is **not** thread-safe (see
T8); and `VALUE_CHANGED` tells you a field moved, not that a dependent property is stale —
dependency-graph invalidation is the next consumer, not a current one.

---

## T7 — Driving a study from the command line

**The situation.** The sweep is working in a notebook. Now a colleague needs to run it with
different numbers, and you do not want to be the person who edits the script for them.

`Analysis.cli()` is an argparse entry point: a JSON input-set file in, a `Result` out.

```python
"""sink_study.py"""
from engforge import Analysis, forge
from heatsink import HeatSink, Fins


@forge
class SinkStudy(Analysis):
    """Heatsink sizing study — junction temp vs airflow and fin stock."""

    def post_process(self, *args, **kwargs):
        df = self.system.last_result.dataframe
        if "t_junction" in df.columns:
            print(f"worst case {df['t_junction'].max():.1f} C over {len(df)} rows")


if __name__ == "__main__":
    res = SinkStudy(system=HeatSink(fins=Fins())).cli()
    print("cli() returned:", type(res).__name__, res)
    if res is not None:
        cols = ["fins.n", "fins.thickness", "air_vel", "q_watt", "t_junction"]
        df = res.dataframe
        print(df[[c for c in cols if c in df.columns]].to_string())
```

```
$ conda run -n pace python sink_study.py --help

usage: sinkstudy [-h] [--input-json PATH] [--set KEY=VALUE]
                 [--save-mode {vars,nums,all,prob,sparse}] [--no-plots]
                 [--list-inputs]

Heatsink sizing study — junction temp vs airflow and fin stock.

options:
  -h, --help            show this help message and exit
  --input-json PATH     JSON input-set file: flat {key: value|[values]} or
                        sectioned {inputs, settings, sequence}; a comma key
                        ('a,b') declares a jointly-varying group with rows as
                        lists
  --set KEY=VALUE       set one input (JSON-parsed value, repeatable;
                        overrides the file)
  --save-mode {vars,nums,all,prob,sparse}
                        row recording mode
  --no-plots            skip plot display
  --list-inputs         print the system's input fields and exit
```

The program name comes from the class name; the description is the **first line of the
class docstring**. Write one.

### Discover the inputs

```
$ conda run -n pace python sink_study.py --list-inputs

heatsink-bronze-net inputs:
  air_vel: float = 2.0
  dynamic_input_vars: list = []
  dynamic_output_vars: list = []
  dynamic_state_vars: list = []
  name: str = 'heatsink-bronze-net'
  q_watt: float = 25.0
  t_air: float = 30.0
  t_junction: float = 60.0
  fins: <slot> = <heatsink.Fins object at 0x178b6ec60>
cli() returned: NoneType None
```

`--list-inputs` prints and returns `None` — the one case where `cli()` is not a `Result`.
It lists the *system's* fields and slots; component fields are reachable dotted
(`fins.n`) but are not enumerated here.

### Two JSON shapes

**Flat** — the whole document is the inputs dict. Shortest thing that works:

```json
{
  "air_vel": [1.0, 4.0],
  "q_watt": 25.0,
  "combos": "thermal"
}
```

```
$ conda run -n pace python sink_study.py --input-json quick.json --no-plots

worst case 77.3 C over 4 rows
cli() returned: Result Result(2 cases)
   fins.n  fins.thickness  air_vel  q_watt  t_junction
0    12.0           0.001      1.0    25.0   77.348485
1    12.0           0.001      1.0    25.0   77.348485
2    12.0           0.001      4.0    25.0   57.726937
3    12.0           0.001      4.0    25.0   57.726937
```

A flat document has nowhere to put run settings, so solver kwargs like `combos` ride the
same stream — `run()` separates them from inputs.

**Sectioned** — `inputs` / `settings` / `sequence`, and nothing else. This is the form to
use for anything real:

```json
{
  "inputs": {
    "air_vel": [1.0, 2.0, 4.0],
    "fins.n,fins.thickness": [[12.0, 0.0010], [24.0, 0.0006]]
  },
  "settings": {
    "combos": "thermal",
    "save_mode": "all",
    "revert_last": true
  },
  "sequence": [
    {"q_watt": 25.0},
    {"q_watt": 40.0}
  ]
}
```

```
$ conda run -n pace python sink_study.py --input-json sweep.json --no-plots

worst case 105.8 C over 24 rows
cli() returned: Result Result(12 cases)
    fins.n  fins.thickness  air_vel  q_watt  t_junction
0     12.0          0.0010      1.0    25.0   77.348485
1     12.0          0.0010      1.0    25.0   77.348485
2     12.0          0.0010      1.0    40.0  105.757576
3     12.0          0.0010      1.0    40.0  105.757576
4     24.0          0.0006      1.0    25.0   53.674242
5     24.0          0.0006      1.0    25.0   53.674242
6     24.0          0.0006      1.0    40.0   67.878788
7     24.0          0.0006      1.0    40.0   67.878788
8     12.0          0.0010      2.0    25.0   66.953474
     ...                                                     (rows 9-21 elided)
22    24.0          0.0006      4.0    40.0   52.181549
23    24.0          0.0006      4.0    40.0   52.181549
```

12 cases: 3 airflows × 2 stock rows (grouped, not crossed) × 2 sequence rows. Three things
to notice:

- **JSON has no tuples**, so group rows are lists of lists. `normalize_group_rows` accepts
  them unchanged.
- An unknown top-level key next to `inputs` is a hard error, not a silent ignore — a
  typo'd section name is almost always a bug. `RESERVED_SECTIONS = ('inputs', 'settings',
  'sequence')`.
- Keys are **not** validated at load time. Field existence and value types are checked by
  `parse_run_kwargs` when the matrix is parsed, so the error message carries the system's
  context instead of the file's.

### `--set` precedence

`--set` is repeatable, values are JSON-parsed (falling back to a bare string), and it
**replaces** the file's entry for that key:

```
$ conda run -n pace python sink_study.py --input-json sweep.json --set "air_vel=[3.0]" --no-plots

worst case 80.2 C over 8 rows
cli() returned: Result Result(4 cases)
   fins.n  fins.thickness  air_vel  q_watt  t_junction
0    12.0          0.0010      3.0    25.0   61.377218
1    12.0          0.0010      3.0    25.0   61.377218
2    12.0          0.0010      3.0    40.0   80.203548
3    12.0          0.0010      3.0    40.0   80.203548
4    24.0          0.0006      3.0    25.0   45.688609
5    24.0          0.0006      3.0    25.0   45.688609
6    24.0          0.0006      3.0    40.0   55.101775
7    24.0          0.0006      3.0    40.0   55.101775
```

But `sequence` is the innermost factor, so a sequence row still wins over `--set`:

```
$ conda run -n pace python sink_study.py --input-json sweep.json \
      --set "q_watt=99.0" --set "air_vel=[2.0]" --no-plots

worst case 89.1 C over 8 rows
cli() returned: Result Result(4 cases)
   fins.n  fins.thickness  air_vel  q_watt  t_junction
0    12.0          0.0010      2.0    25.0   66.953475
1    12.0          0.0010      2.0    25.0   66.953475
2    12.0          0.0010      2.0    40.0   89.125559
3    12.0          0.0010      2.0    40.0   89.125559
4    24.0          0.0006      2.0    25.0   48.476737
5    24.0          0.0006      2.0    25.0   48.476737
6    24.0          0.0006      2.0    40.0   59.562780
7    24.0          0.0006      2.0    40.0   59.562780
```

`air_vel=[2.0]` took effect; `q_watt=99.0` never appears — the sequence overwrote it on
every point. **Precedence, outermost to innermost: file `inputs` < `--set` < `sequence`
rows.** If a `--set` seems to do nothing, look for the key in `sequence`.

`--save-mode` overrides `settings.save_mode`:

```
$ conda run -n pace python sink_study.py --input-json quick.json --save-mode prob --no-plots

worst case 77.3 C over 4 rows
cli() returned: Result Result(2 cases)
   t_junction
0   77.348485
1   77.348485
2   57.726937
3   57.726937
```

`--no-plots` sets `self.show_plots = False` before the run, which suppresses `fig.show()`
in `report_results` — the flag you want on a headless box or in CI.

> **Known wart: `Analysis` doubles your rows.** Look at the counts above —
> `Result(2 cases)` but 4 rows, `Result(12 cases)` but 24 rows. `Analysis.run` installs
> `cb=lambda *a, **kw: self.system.last_context.save_data(force=True)`
> (`analysis.py:68`), which forces a second, identical row on top of the automatic one.
> Measured directly:
>
> ```
> direct  : Result(2 cases) rows/case [1, 1] df (2, 17)
> analysis: Result(2 cases) rows/case [2, 2] df (4, 17)
> ```
>
> Harmless for `max`/`min`, wrong for counts, means, and sums. De-duplicate in
> `post_process` (`df.drop_duplicates()`) until the callback is fixed.

Also note `Analysis.run()` now **returns** the `Result` (it used to return nothing), and
`Analysis.dataframe` forwards to `self.system.dataframe`.

---

## T8 — Parallel problem contexts: what works, and what does not

**The situation.** Two Pareto fronts to explore, four cores sitting idle. Natural instinct:
run them at the same time. Here is what actually happens, measured.

### Why simultaneous contexts cannot work today

`ProblemExec.class_cache` is `ProblemExec` itself, and `__enter__` does
`self.class_cache.session = self`. The session is a **plain class attribute on a
module-level class**:

```python
print("has 'session' before:", "session" in ProblemExec.__dict__)
with ProblemExec(hs, {}, level_name="a") as px:
    print("has 'session' inside:", "session" in ProblemExec.__dict__)
    print("class_cache is ProblemExec:", ProblemExec.class_cache is ProblemExec)
    px.exit_to_level("a", revert=True)
print("has 'session' after :", "session" in ProblemExec.__dict__)
```

```
has 'session' before: False
has 'session' inside: True
class_cache is ProblemExec: True
has 'session' after : False
```

And there is no isolation machinery anywhere in the module
(`grep -c` on `engforge/problem_context.py`):

```
  'threading.local'    in problem_context.py: False
  'contextvars'        in problem_context.py: False
  'ContextVar'         in problem_context.py: False
  'threading.Lock'     in problem_context.py: False
  'import threading'   in problem_context.py: False
```

Two consequences.

**Nesting is bookkeeping, not isolation.** A second context on the *same* instance hands
you back the first object:

```python
with ProblemExec(hs, {}, level_name="outer") as outer:
    with ProblemExec(hs, {}, level_name="inner") as inner:
        print("inner is outer       :", inner is outer)
        print("same Result          :", inner.result is outer.result)
```

```
inner is outer       : True
same Result          : True
```

Levels scope *exits and reverts*. They do not give you a second problem.

**A different instance is refused.** The guard fires at construction:

```python
with ProblemExec(a, {}, level_name="A") as pa:
    try:
        with ProblemExec(b, {}, level_name="B") as pb:
            print("no error?!")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
```

```
IllegalArgument: somethings wrong! change of comp! <heatsink.HeatSink object at 0x157eeaf00> -> <heatsink.HeatSink object at 0x168fe3890>
```

### Threads

Two barrier-synchronized threads, two separate `HeatSink` instances, `run()` on each:

```python
import threading

errs, oks = [], []

def work(sysobj, vals):
    try:
        r = sysobj.run(air_vel=vals, combos="thermal")
        oks.append(tuple(round(v, 4) for v in r.dataframe["t_junction"]))
    except Exception as e:
        errs.append(type(e).__name__)

for trial in range(10):
    errs.clear(); oks.clear()
    s1 = HeatSink(fins=Fins())
    s2 = HeatSink(fins=Fins(n=24.0))
    bar = threading.Barrier(2)
    def t1(): bar.wait(); work(s1, [1.0, 2.0])
    def t2(): bar.wait(); work(s2, [1.0, 2.0])
    th = [threading.Thread(target=t1), threading.Thread(target=t2)]
    [t.start() for t in th]
    [t.join() for t in th]
    print(f"  trial {trial}: errors={errs} ok={len(oks)}")
```

```
  trial 0: errors=['IllegalArgument'] ok=1
  trial 1: errors=['IllegalArgument'] ok=1
  trial 2: errors=['IllegalArgument'] ok=1
  trial 3: errors=['IllegalArgument'] ok=1
  trial 4: errors=['IllegalArgument'] ok=1
  trial 5: errors=['IllegalArgument'] ok=1
  trial 6: errors=['IllegalArgument'] ok=1
  trial 7: errors=['IllegalArgument'] ok=1
  trial 8: errors=['IllegalArgument'] ok=1
  trial 9: errors=['IllegalArgument'] ok=1
```

**0 of 10 clean.** One thread wins the class attribute, the other raises. The guard only
fires when a session exists at *construction* time, so the failure mode is
scheduling-dependent: construct both contexts first and then enter them, and the second
silently clobbers the attribute instead of raising. Each context caches `inst_sesh` during
`__init__`, so from the inside the divergence is invisible — the numbers just come back
wrong, with no exception. In other words: **threaded engforge either raises or lies, and
which one you get depends on timing.** Do not thread it.

### If sessions were isolated, you would still need separate entities

Worth being clear about why thread-local storage alone would not be enough. State lives in
live attrs objects; `SystemState` is a snapshot *of* them, not a store the context reads
from. Two explorations of one instance alias each other. The right copy mechanism:

```python
src = HeatSink(fins=Fins())
dup = src.copy_config_at_state()
print("same system  :", dup is src)
print("same fins    :", dup.fins is src.fins)
print("dup.fins.parent is dup:", dup.fins.parent is dup)
dup.fins.n = 99.0
print("src.fins.n after dup mutation:", src.fins.n)
```

```
same system  : False
same fins    : False
dup.fins.parent is dup: True
src.fins.n after dup mutation: 12.0
```

`copy_config_at_state()` (`configuration.py:433`) is a recursive `attrs.evolve` with an
identity map: a genuinely independent graph with parent links intact. Prefer it to
`System.clone()` — `clone()` deep-copies through `TabulationMixin.__getstate__`, which pops
`parent`, leaving the clone's components with no `parent` attribute at all.

### Processes: pickle decides, and economics vetoes

A `System` carrying solver attributes does not pickle **at all** — not even freshly
constructed. The `Solver.declare_var` machinery generates a class per variable, and that
class does not exist at module scope for the unpickler to find:

```
  fresh system          : PicklingError: Can't pickle <class 'engforge.attributes.Solver_var_t_junction_2ea438bbf46b46ba'>: attribute lookup
  a bare Component      : 324 bytes
  system after a run    : PicklingError: Can't pickle <class 'engforge.attributes.Solver_var_t_junction_2ea438bbf46b46ba'>: attribute lookup
  Result (system attached): PicklingError: Can't pickle <class 'engforge.attributes.Solver_var_t_junction_2ea438bbf46b46ba'>: attribute lookup
  dataframe             : 1477 bytes
  SystemState           : 594 bytes
  SystemState (numeric) : 228 bytes
```

(Error strings truncated to 100 chars by the probe. `SystemState` pickles because this
model's slot holds a plain `Component`; a slot holding a solver-bearing `System` would
not.)

So the shape of a working fan-out is forced: **children build their own system from a
plain spec and return plain data** (a dataframe, a `SystemState`, a dict):

```python
import multiprocessing as mp

def job(spec):                      # runs in the child
    from heatsink import HeatSink, Fins
    hs = HeatSink(fins=Fins(**spec["fins"]))
    res = hs.run(**spec["inputs"], combos="thermal")
    return spec["tag"], res.dataframe[["air_vel", "t_junction"]].to_dict("list")

SPECS = [
    {"tag": f"n{n}", "fins": {"n": n, "thickness": t},
     "inputs": {"air_vel": [1.0, 2.0, 3.0, 4.0], "q_watt": [20.0, 25.0, 30.0, 40.0]}}
    for n, t in [(12.0, 0.0010), (18.0, 0.0008), (24.0, 0.0006), (30.0, 0.0005)]
]

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        par = pool.map(job, SPECS)
```

That works end to end — and it is slower:

```
sequential  : 0.97s  (4 jobs)
Pool(4)     : 5.40s  -> speedup 0.18x
bare Pool(4): 0.17s (spawn only, no work)
same answers: [True, True, True, True]
```

The answers match; the wall time is 5.6× worse. The reason is in the third line: spawning
is cheap (0.17 s), but every child has to `import engforge`, which measured **3.60 s** in
this environment. A job here costs ~0.24 s. Process fan-out only pays once a single job
costs *seconds*, not milliseconds.

### What to do instead: make it one run and segment the frame

For two fronts over the same grid, the grouped matrix key (T2) is the answer — one context,
one `Result`, one frame you slice:

```python
import time

STOCK = [(12.0, 0.0010), (24.0, 0.0006)]
drive = HeatSink(fins=Fins())
t0 = time.perf_counter()
res = drive.run(
    **{"fins.n,fins.thickness": STOCK,
       "air_vel": [1.0, 2.0, 3.0, 4.0],
       "q_watt": [25.0, 40.0]},
    combos="thermal",
)
dt = time.perf_counter() - t0
df = res.dataframe
print(f"{len(res.cases)} cases in {dt:.3f}s -> df {df.shape}")
A = df[df["fins.n"] == 12.0]
B = df[df["fins.n"] == 24.0]
print("front A rows:", len(A), "max Tj", round(A["t_junction"].max(), 2),
      "mass", A["fins.mass"].iloc[0])
print("front B rows:", len(B), "max Tj", round(B["t_junction"].max(), 2),
      "mass", B["fins.mass"].iloc[0])
```

```
16 cases in 0.130s -> df (16, 17)
front A rows: 8 max Tj 105.76 mass 0.032400000000000005
front B rows: 8 max Tj 67.88 mass 0.03888
```

Both fronts, 16 solves, **0.130 s.** There is no throughput problem to solve here.

### Decision table

| situation | do this |
|---|---|
| N fronts, same sweep grid | one `run()` with a grouped comma key; slice the frame (above) |
| N fronts, *different* grids | one context, an explicit `SystemState` tree, one `begin_case` per front (T5) |
| one front's run costs seconds | process fan-out: children build from a spec, return dataframes/`SystemState`s |
| anything threaded | don't |

### Honest limits of the above

- The thread findings are about the **session attribute**, which is enough to rule threads
  out. The solver's own thread-safety under a hypothetical isolated session is untested —
  scipy `minimize` paths, transients (`dxdt=True`), `Signal`s, `ComponentIterator`, and the
  pyee bus (also process-wide) were not exercised concurrently.
- A thread-local session is a ~15-line core change (route `session` through a
  metaclass-backed thread-local *class*, so `isinstance(self, self.class_cache)` at
  `problem_context.py:827` stays legal) plus one documented invariant: **one entity graph
  per context**. That is a storage change, not a conceptual one, and it is the right
  eventual fix. It is not in the tree today.
- Whether `attrs.evolve`-based copies preserve `setup_global_dynamics` state, plot
  attributes, or `Analysis` report wiring is unverified.
- Timings are one macOS machine and one trivial model, and they **do not reproduce
  closely** — a re-measurement on the same machine gave sequential 0.23 s, `Pool(4)`
  2.14 s (0.11× speedup), bare `Pool(4)` 1.51 s, and `import engforge` 0.95 s in-process /
  1.75 s per child under a 4-way spawn rather than 3.60 s. (The 0.17 s "spawn only" figure
  above is also lower than this script can achieve: with `spawn`, each child re-imports
  `__main__`, which imports engforge.) What is stable across both passes is the *sign*:
  the spawn+import floor is seconds and a job here is ~0.05–0.25 s, so fan-out loses by
  roughly an order of magnitude. Treat every number in this section as an order of
  magnitude and measure your own model before you parallelize.

---

## Transients: `simulate` vs `sim_matrix`

A short addendum, because the two entry points return different things.

```python
sw = SinkWarmup(fins=Fins())
df = sw.simulate(dt=0.01, endtime=10.0)
print("simulate returns :", type(df).__name__, df.shape)
print("last_result      :", sw.last_result, "| cases", len(sw.last_result.cases))

sw2 = SinkWarmup(fins=Fins())
r = sw2.sim_matrix(dt=0.01, endtime=10.0, air_vel=[1.0, 4.0])
print("sim_matrix returns:", type(r).__name__, r)
for i, c in enumerate(r.cases):
    print(f"  case {i}: inputs={c.inputs} rows={len(c.rows)} "
          f"output={type(c.output).__name__}")
print("input_sets:", r["input_sets"])
```

```
simulate returns : DataFrame (5006, 19)
last_result      : Result(1 cases) | cases 1

sim_matrix returns: Result Result(2 cases)
  case 0: inputs={'air_vel': 1.0} rows=5006 output=DataFrame
  case 1: inputs={'air_vel': 4.0} rows=5006 output=DataFrame
input_sets: {0: {'air_vel': 1.0}, 1: {'air_vel': 4.0}}
```

- `simulate()` returns the episode's **DataFrame** directly. It runs on a *copy* of the
  system, so the context never attaches — but the episode's `Result` is handed back
  explicitly, so `system.last_result` still works (`Result(1 cases)`).
- `sim_matrix()` returns a **`Result`**, and its docstring ("return a dictionary of
  dataframes", `dynamics.py:553`) is wrong.

> **One case per point, and that took a fix.** Two layers open a case per sweep point —
> the input-matrix loop (`solveable.py:402`) and each episode (`dynamics.py:640`) — which
> used to emit **2N** cases for N points, half of them empty with `output=None`.
> `ProblemExec.begin_case` (`problem_context.py:500-518`) now *collapses* a trailing case
> that never recorded a row and merges the new inputs into it, so a point's case and its
> episode's case are the same `Case`: one case per point, holding both the rows and the
> frame. Earlier drafts of this page, the README, and `ENGFORGE_1.0.md` Part V all
> described the 2N behaviour; it is gone.

Reading it by filtering on what you actually need is still the robust habit — it survives
either shape:

```python
import pandas
runs = [(c.inputs, c.output) for c in r.cases
        if isinstance(c.output, pandas.DataFrame)]
for inputs, d in runs:
    print(f"  {inputs} -> {d.shape} final t_metal {d['t_metal'].values[-1]:.4f}")

df = r.dataframe
print("combined:", df.shape, "| run_id groups:", df["run_id"].nunique())
print(df.groupby("air_vel")["t_metal"].max())
```

```
  {'air_vel': 1.0} -> (5006, 19) final t_metal 57.7091
  {'air_vel': 4.0} -> (5006, 19) final t_metal 51.5572

combined: (10012, 19) | run_id groups: 0
air_vel
1.0    57.709142
4.0    51.557214
Name: t_metal, dtype: float64
```

Going from 1 to 4 m/s pulls 6.2 °C out of the 10-second peak. Note `run_id.nunique()` is
`0` — `run_id` is all `None` here, so the `df.groupby('run_id')` idiom from the README does
not apply to `sim_matrix` output. **Group by the swept input column.**

---

## Import-paths cheat sheet

`engforge/__init__.py` exports none of the new names. Use these paths:

```python
from engforge.results import Result, Case
from engforge.state import SystemState
from engforge.events import (bus, watch, has_listeners,
                             VALUE_CHANGED, SLOT_CHANGED, COLLECTION_CHANGED)
from engforge.inputs import (is_group_key, split_group_key, normalize_group_rows,
                             expand_point_factors, count_points,
                             load_input_sets, parse_cli_value, parse_set_overrides,
                             RESERVED_SECTIONS)
from engforge.problem_context import ProblemExec, save_modes
from engforge import Analysis, System, Component, forge, Slot, Solver, Signal, Time
```

## Migrating from 0.x

| 0.x | now |
|---|---|
| `run()` handed back the system | returns a `Result`; the system is **reverted** |
| `system.dataframe` was the primary store | it is a fallback shim; the owner is `system.last_result` |
| rows accumulated implicitly per `save_data()` | `Case.record` on an explicit `begin_case(...)` |
| `ProblemExec.data` was the run's rows | it is the **active case's** rows; use `.numeric_data` for the run |
| `record_state` / `x_start` / `temp_state` dicts | `SystemState.capture/apply/updated/diff`; `x_start` survives as a labelled legacy dict view |
| every input kwarg crossed with every other | still true *unless* a comma key groups them (T2) |
| no row-size control | `save_mode` (T4) |
| no change notification | `engforge.events` (T6) |
| scripts hand-rolled argparse | `Analysis.cli()` (T7) |

## Where to look next

| path | what it shows |
|---|---|
| `examples/air_filter.py` | the canonical `Solver.declare_var` / `constraint_equality` / `Signal.define` idiom, plus reporters |
| `examples/spring_mass.py` | the minimal transient (`Time.integrate`, `simulate`) |
| `examples/transient_control.py` | grouped comma keys on tuned PID gain sets × profile components (T2 on a real model) |
| `examples/design_offdesign.py` | design-point vs off-design operation |
| `examples/balanced_network.py` | a solved flow/balance network |
| `examples/state_variations.py` | state-tree variation study (companion to T5) |
| `engforge/test/test_input_groups.py` | the contract tests for T2 |
| `engforge/test/test_state_results.py` | the contract tests for T3 / T5 |
| `engforge/test/test_events.py` | the contract tests for T6 |
| `engforge/test/test_cli_inputs.py` | the contract tests for T7 |
| `docs/ENGFORGE_1.0.md` | the design record: why these structures exist and where they go in 1.0 |
| `engforge/inputs.py`, `state.py`, `results.py`, `events.py` | the module docstrings are accurate and short — read them |
