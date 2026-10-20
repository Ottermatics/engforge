# engforge 1.0 — Design Document

*The unified design for engforge 1.0. Parts I–IV are the drawing-board foundation: the update
cycles our problem classes require, what the current system executes, the measured gap, and the
minimal engine changes — expressed through existing engforge idioms (decorator marking + combos,
interface overrides, argument-passed bindings). Parts V–VII record what has already landed or been
measured; Part VIII is the test strategy. Part IX is the compiled-graph engine design that
implements it all. Everything is a Component; the declaration API stays
Component / Slot / Signal / Solver / Time / @system_property.*

*Evidence: every coverage claim below carries either a `file:line` mechanic or a probe number
from `sandbox/cycles/run_probes.py` (P-A … P-H).*

**Part index**

| part | what it is | status |
|---|---|---|
| I | the problem classes and their update cycles | design |
| II | what engforge executes today (verified) | measurement |
| III | coverage matrix (probe-backed) | measurement |
| IV | minimal engine changes, through the levers | design |
| **V** | **Landed: explicit State & Result** — the staging of Part IX | **shipped code** |
| **VI** | **Parallel problem contexts** | **analysis only — no code landed** |
| **VII** | **The ref space: where the DAG meets the solver matrix** | **measurement + prototype; 4 cache fixes landed** |
| VIII | test & evaluation strategy | design |
| IX | the compiled-graph engine design | design |
| **X** | **worst-case stress tests & core purpose** — MPC-in-transient, CFD/FEA black boxes | **critical analysis** |

*Companion: [`SCALING.md`](SCALING.md) — parallel execution (measured feasibility, Ray mapping,
executor seam), the size problem, and the cloud commercial analysis.*

---

# PART I — The problem classes and their update cycles

## I.1 The cycle alphabet

Every target problem class is a nesting of eight primitive cycle elements:

| element | meaning |
|---|---|
| EVAL | one pass through a value DAG (properties) |
| SWEEP | ordered one-pass over parts (components/stations/cells) |
| ITERATE | repeat to a fixed point (‖Δ‖ < tol) — Gauss-Seidel/Jacobi |
| SOLVE | residual block → linear algebra → update, to tol (Newton/direct) |
| MAP | independent evaluation over set members (parallelizable) |
| REDUCE | aggregate member results into parent values |
| MARCH | time/space stepping (each step contains one of the above) |
| OPTIMIZE | driver loop over design vars |

## I.2 Class 1 — feed-forward design study (hinge fit, airfilter)

```
for X in input_sets:                       # MAP (outer product / cases)
    set_inputs(system, X)
    for node in topo_order(value_dag):     # EVAL — order from dependencies
        node.value = f(node.deps)
    record_row()
# optional driver:
while not converged:                       # OPTIMIZE
    f, g = EVAL(X);  X = step(X, grad)
```
No inner iteration. Correctness = evaluation order; cost = one DAG pass per point.

## I.3 Class 2 — balanced network (jet engine, pipe nets, drivetrain)

```
u = guess(flows, speeds, PRs)              # the balance unknowns
while norm(r) > tol:                       # SOLVE (Newton, bounded)
    for comp in flow_order(components):    # SWEEP along the flow topology
        comp.out = comp.compute(comp.in(u))
    r = residuals(u)                       # continuity, energy, shaft power = 0
    J = dr/du                              # small, sparse
    u = u + solve(J, -r)
# design mode:    u = {W, FAR, PR_turbine}
# off-design:     u = {W, FAR, N_mech}     # SAME residuals, different unknown partition
```

**The multi-domain requirement (flagship case).** A real engine model integrates several domains
with flexible solving: top-level design/off-design performance runs on **component maps**, while
each component (compressor / turbine / fan) also carries its own **internal blade-geometry
analysis** and **structural analysis**. All domains are defined upfront in the classes — that's
the expected legwork — and **combos is the segmentation tool** that selects which domain/fidelity
solves in a given run:

```python
run(combos='cycle,design')      # map-based design point
run(combos='cycle,offdesign')   # same residuals, different unknowns  (probe P-E: works TODAY)
run(combos='blade')             # blade geometry analysis on the same components
run(combos='struct')            # structural checks
```

## I.4 Class 3 — coupled disciplines, fixed point (platform↔turbine, aero↔structure)

```
while not opt_converged:                   # OPTIMIZE over design d
    while norm(delta) > tol:               # ITERATE (Gauss-Seidel MDA)
        a = A.compute(d, b)
        b = B.compute(d, a)
    f, g = objectives(a, b)                # gradients valid ONLY at a consistent (a,b)
    d = step(d, g)
# alternative (IDF): lift (a,b) to the solver with residuals a-A(d,b)=0, b-B(d,a)=0
```
The inner loop must converge **every** outer evaluation, or the gradients lie.

## I.5 Class 4 — continuum mesh (FEA/CFD)

```
# steady implicit:
while norm(R) > tol:                       # SOLVE (Newton / linear)
    for cell in cells:                     # MAP (parallel) over members
        local = gather(u, neighbors[cell]) #   sparse stencil read
        R[cell], Jrow = cell.residual(local)
    J = assemble_sparse(Jrows)
    u = u + sparse_solve(J, -R)
# matrix-free variant:  ITERATE sweeps of local cell updates until norm(du) < tol
# transient:            MARCH t: each step = one SOLVE (implicit) or one sweep (explicit)
```
Signatures: locality (stencil), sparse structure, N ≫ 1e3, assembly = gather/scatter.

## I.6 Class 5 — sets × sites / multipoint design

```
while not opt_converged:                   # OPTIMIZE over SHARED design d
    for site in sites:                     # MAP (independent → parallel)
        y[site] = evaluate(system, d, site)
    F = reduce(weights, y)                 # REDUCE: weighted sum / worst-case
    d = step(d, grad(F))
# "run at one site" = the same cycle, MAP over one site
# coupled arrays (wakes): the MAP becomes an ordered SWEEP or a SOLVE
```

## I.7 Class 6 — stochastic optimization (random upstream output)

```
while not opt_converged:                   # OPTIMIZE over design d
    for s in range(n_samples):             # MAP over random draws (seeded)
        w = draw(dist, seed=base_seed + s) # CRN: SAME seeds every outer iteration
        y[s] = evaluate(system, d, w)
    F = statistic(y)                       # REDUCE: mean / quantile / CVaR
    d = step(d, grad(F))                   # valid only because seeds are common
```
Disciplines: common random numbers (else FD gradients are noise — probe P-F), sample size vs
tolerance, statistics as first-class outputs. Hook today: `@system_property(stochastic=True)`.

## I.8 Class 7 — closed-loop modeling on a real data timeseries

```
# replay / correlation:
for t, row in timeseries:                  # MARCH driven by RECORDED data
    set_measured_inputs(system, row)
    step(system, dt)
    err[t] = outputs(system) - row.measured_outputs
report(err)
# parameter estimation (system ID):
while not fit_converged:                   # OPTIMIZE over params p
    E = replay(timeseries, p)              # the whole MARCH is one objective eval
    p = step(p, grad(sum(E^2)))
```
Precedents in-repo: `test_dynamics` ω-fit; client rig-correlation data (Triton
`test_analysis/data_cache` + `testpoints`). Probe P-G demonstrates both halves today.

---

# PART II — What engforge executes today (verified)

```
run(**kw):
    for X in product(input_lists) x narrow_iter_frames:   # MAP — independent points only
        enter ProblemExec                                  # re-walks/recaches ALL refs
        apply_signals(declaration_order)                   # ┐ sweeps with NO dependency
        run_update_refs(declaration_order, one_level)      # ┘ sort and NO tol contract
        minimize(SLSQP, vars=all_selected, cons=all)       # the ONLY iterative construct
            # per objective/constraint eval: re-enter context; set x; sweep again;
            # lazily EVAL touched props.  con_eq -> minimize(1 + r^2) fallback
        post_signals(); save_row(); revert()

simulate(): MARCH via solve_ivp over name-concatenated per-component states
    # per step: one sweep + per-component block-diagonal rates (no cross-component
    # residual incidence); runs on a COPY and returns the dataframe
reduce:    parent @system_property over wide-collection refs
```

In alphabet terms: `MAP_product( SWEEP_decl ∘ OPTIMIZE_flat( EVAL_lazy ) )` + a separate
`MARCH( SWEEP )`.

Topology facts (each verified by execution or probe):
- Ownership is a tree; the lazy property DAG is correct per-read.
- All cross-component flow lives in declaration-ordered sweeps. **Measured (P-D): `update()` is
  called 6× per `run()` (12 after two runs) — there is no execution-count contract, so
  non-idempotent updates (accumulators, cost tallies) are broken by design; and under strong
  coupling the answer is declaration-order dependent (swap two class-body lines → results differ
  by 0.80, mirrored: (8.62, 9.19) vs (9.19, 8.62), both far from the true (20, 20)).**
- **Accidental iteration (new finding, P-B): `mode="both"` signals re-apply at every nested
  context entry, so one `run()` performs ~8 unplanned Gauss-Seidel sweeps.** Weak couplings
  therefore converge *by accident*; a strong coupling (ρ=0.9025) retains error 12.14 after a run
  and creeps across re-runs (4.35 → 1.56 → 0.56) — convergence with no tolerance contract.
  Reaching 1e-10 needs 232 controlled GS iterations (the ITERATE engforge lacks).
- Solver declarations resolve **downward through slots only** (P-A: a child residual referencing
  `parent.turb.torque` → `AssertionError: invalid parent`): couplings must be declared at a
  common ancestor.
- Collection members are tabulation-only: their `declare_var`/`con_eq`/updates are silently
  ignored (P-C: solver saw `['plain.x']` only; member fields untouched at 3.0).
- `con_eq` has no Newton: residuals become a `1+r²` objective under flat SLSQP. On a small clean
  2×2 balance that works well (P-A: converged in 4 iterations / 13 evals, residual 1.2e-9,
  matching hand-Newton's answer) — the weakness is robustness/scaling and the missing
  flow-ordered sweep, not the small case.

---

# PART III — Coverage matrix (probe-backed)

| class | covered? | evidence for | evidence against |
|---|---|---|---|
| 1 feed-forward | **native** | the entire current feature set | sweep order is declaration luck (P-D); 6× redundant update calls per run (P-D) |
| 2 balanced network | **partial+** | flat SLSQP solved the 2×2 balance exactly (P-A: 4 it/13 evals, r=1.2e-9); **design/off-design unknown re-partition WORKS TODAY via combos** (P-E: `combos='design'` → PR=4.0 with Nmech untouched; `combos='offdesign'` → Nmech=6.0 with PR untouched) | downward-only declarations (P-A assertion); no Newton on residual blocks (robustness at scale unproven); no flow-ordered sweep |
| 3 coupled fixed point | **accidental only** | ~8 unplanned sweeps per run converge weak couplings (P-B) | no ITERATE contract: strong coupling leaves err 12.14, creeps across re-runs, no tolerance, order-dependent (P-B, P-D) |
| 4 continuum mesh | **not covered** | — | members invisible to solver (P-C); no neighbor refs; no assembly; per-object Python dispatch at N |
| 5 sets/multipoint | **partial** | MAP via input product & narrow iter; REDUCE via wide refs | member solver content ignored (P-C); no member coupling; no shared-design cross-point objective in one solve |
| 6 stochastic opt | **partial** | `stochastic=True` + list-sweep MC exist | no CRN: same design, 5 runs → 5 objectives (spread ~0.12); FD "gradients" spanned **109** vs true −2.0; hand-seeded CRN recovers −2.0256 exactly ×5 (P-F) — the gap is seed plumbing + statistics-REDUCE, not math |
| 7 closed-loop on data | **partial (all user code)** | replay RMSE 2e-5 at true params; system-ID recovered (k,c)=(9.9996, 0.5999) from (10, 0.6) in 86 evals (P-G) | measured-input injection = hand-written interpolation inside `update_state_constants` + `nonlinear=True` to defeat matrix caching; error accumulation and staleness are user code |

---

# PART IV — Minimal engine changes, through the levers

## IV.1 The levers (all existing idioms)

| lever | precedent | role |
|---|---|---|
| L1 decorator marking + **combos** selection | `@system_property`, `@cost_property(category=,mode=)`; `combo_filter`/`filter_vals` | methods become cycle participants, activated by `run(combos=…)` — **combos is the house segmentation tool for types of analysis**. Solver-scoped naming: **`@Solver.residual(combos=…)`** (method-local residual, consistent with `Solver.con_eq`), `@Solver.update(combos=…, before=/after=…)` for ordered sweep participation |
| L2 interface overrides | `update()`, `create_state_matrix()`, `_solver_override` | a component owns its block's inner solve (the `_solver_override` pattern generalized to SCC blocks); collection-level gather/assembly overrides |
| L3 args → collection fields | `run(base=[None, MetalBase()])` | member addressing via the input channel: broadcast `cells.T0=5`, per-member arrays, member sweeps; site sets `sites.current=[s1,s2,s3]` |
| L4 args → normal fields | `run(x=5, **{'comp.x': […]})` | the universal mutation channel — unchanged |

## IV.2 The changes

| # | change (engine-level) | unlocks | status |
|---|---|---|---|
| C1 | SWEEP in value-graph topological order with a **declared execution contract** (exactly-once per phase, or iterate-to-tol) — replaces declaration order and the accidental re-sweeps | correctness for 1–3; prereq for all | probe-proven need (P-B/P-D) |
| C2 | detect coupling cycles → ITERATE/SOLVE the block to tolerance inside each outer eval (SCC → GS/Aitken or Newton); `con_eq` blocks get real Newton | class 3; robustness for 2 | P-B shows 232-iteration ITERATE need; P-A shows Newton parity |
| C3 | generalized ref reach for existing string refs (sibling/parent/member) — resolver-only; largely obviated where `@Solver.residual` methods make deps visible by tracing | declare couplings where they live | P-A shows today's block |
| C4 | collection members become full problem citizens (vars/residuals/updates collected; parent set; copy fixed) | class 5; doorway to 4 | P-C proves the gap |
| C5 | **indexed peer Slots + member-block assembly**: a Slot may hold multiple indexed references to same-type components in N dimensions — structured sugar `i,j,k ±` for square cells (`self.nbr[+1,0,0].T`), index lists for unstructured; or **fixed-arity node slots** (a cube element holding 8 shared Node components — classic element↔node incidence). Peer slots are references, not owners (single-parent rule relaxed). Stencil properties (`dTdx = (nbr[+i].T − nbr[−i].T)/(nbr[+i].x − nbr[−i].x)`) read through them; identical-form members evaluate as one vectorized block whose Jacobian sparsity IS the slot-index pattern | class 4 | design (P-H shows the stencil Jacobian rows symbolically) |
| C6 | ~~unknown re-partition~~ — **covered today** by combo-selected vars over shared residuals | class 2 design/off-design | **P-E: done, no change needed** |
| C7 | cross-point aggregate: MAP subproblems sharing declared vars + statistics/weights REDUCE objective inside one problem; seeded-draw MAP for stochastic (CRN owned by the engine) | class 5 multipoint; class 6 | P-F quantifies the CRN requirement |
| C8 | measured-input channel for MARCH (timeseries → inputs per step) + error-accumulation hooks | class 7 ergonomics | P-G shows the contortion that works today |

Model code is unchanged by C1–C3 (same declarations, better engine); C4–C5 upgrade the existing
collection/Slot primitives; C7–C8 are assembly/plumbing. **No new user-facing classes.**

---

# PART V — Landed: explicit State & Result (the staging of Part IX)

*Implemented this pass. Model code is unchanged; what moved is WHERE mutable state and
result rows live.*

**Why.** A `sim_matrix` bug chase produced four stacked patches (row-slicing, data-owner
scopes, an owner stack, a time-restore) — all symptoms of one disease: **state scattered
across live objects + a recursive context singleton, and rows accumulating implicitly as a
side effect of property evaluation.** Every nested scope had to mutate-then-restore shared
state, and every restore path was a new bug surface.

**What landed.**

| structure | file | owns |
|---|---|---|
| `SystemState` | `engforge/state.py` | the typed snapshot: `capture / apply / updated / copy / diff`, plus the episode clock. **Backs `x_start` with a `SystemState`** — `set_checkpoint` (`problem_context.py:1886-1892`) builds `self.x_state = SystemState.capture(self)` and keeps `x_start = self.x_state.values` as an explicitly labelled legacy dict view; `record_state` (`:1787`) and `temp_state` (`:449, :479, :1793, :1912`) remain as readers. `apply(restore_time=)` separates *solver revert* (fields only — `revert_to_start` passes `restore_time=False`) from *episode restore* (fields + clock). |
| `Case` / `Result` | `engforge/results.py` | one input-point's inputs/rows/output; the run's cases. Recording is an explicit `case.record(...)` with the dedup policy (`force or not rows or changed`) moved out of the context. `Result` is a `Mapping`, so `run()["output"][i]` / `["input_sets"][i]` are unchanged. |

- `ProblemExec` keeps **no run-level row store**: `begin_case()` starts a point, `save_data`
  records on the active case, `numeric_data/dataframe` read the Result. Precisely: the row
  *store* is gone, but the `_data: list` class annotation survives (`problem_context.py:237`)
  along with `data=None` in `root_defined` (`:139`), and **`ProblemExec.data` now returns the
  ACTIVE case's rows** (`:529-532`), not the run's. `.data` (active case) and `.numeric_data`
  (whole run, `:2208-2211`) are one letter apart in name and very different in scope — a reader
  using `.data` to get the whole run gets one case.
- `simulate()` is now an **episode**: captures a `SystemState`, owns its `Case`, restores state
  *and* clock on exit. Transient cases keyed by time can no longer collide — the four patches
  are deleted, not kept.
- `system.last_result` exposes the Result; `system.dataframe` stays as the combined view.
  `simulate()` evaluates on a COPY, so its context never attaches to the original — it hands the
  Result back explicitly via `_last_result`, which `TabulationMixin.last_result`
  (`tabulation.py:104-121`) prefers, so "where did my rows go" has one answer regardless of
  entry point. (Verified: `Result(1 cases)` after a bare `simulate()`.) `TabulationMixin.dataframe`
  (`:84-102`) is now a three-level fallback shim — `last_context.dataframe` → `_patch_dataframe`
  → empty frame.

**Also landed with the input layer** (`engforge/inputs.py`, pure stdlib, no engforge imports):

- grouped matrix inputs — `"a,comp.b": [(1,2),(3,4)]`, joint (zipped) rows crossing the other
  factors. Verified partial factorial: `run(**{"kp,ki":[(2,1),(8,4)], "x":[1,2,3]})` → **6 cases,
  not 12**; dotted subkeys work (`{"x,comp.y":[(1,10),(2,20)]}` → 2 cases).
- `sequence=` validated; crossed innermost, overriding on key collision.
- `is_group_key` / `split_group_key` / `normalize_group_rows` / `expand_point_factors` /
  `count_points` — the last being the dry-run point count before an expensive sweep.
- **JSON input sets**: `load_input_sets(path)` accepts a flat document (the whole object is the
  inputs dict) or a sectioned one restricted to `RESERVED_SECTIONS = ('inputs','settings','sequence')`
  — an unknown top-level key next to `inputs` fails loudly rather than being silently dropped.
  `parse_cli_value` (JSON first, bare string fallback) and `parse_set_overrides` back the CLI.
- genuine bug fixes: `dxdt` clobber, `integrate` X0 merge, `sim_matrix` copy_system, and a typed
  `ComponentIterator.append` (the raw `UserList.append` bypassed both the type check and the
  reference-cache reset).

**Also landed: the change-event bus** (`engforge/events.py`) — the first concrete piece of
**M8 (§9, incremental recompute)**. §9's first bullet asks for attrs `on_setattr`
(`property_changed`) to be "repointed from the global boolean to the graph → eager dirty push";
the bus is that repointing, minus the graph that does not exist yet. Writes now *emit* instead of
only setting `_anything_changed`, so the dirty-push consumer can be added without touching the
DSL again.

| event | payload | emitted from |
|---|---|---|
| `VALUE_CHANGED` | `(instance, name, old, new)` | `configuration.property_changed` (`:199-213`) |
| `SLOT_CHANGED` | `(instance, name, old, new)` | same, when either side `isinstance Configuration` |
| `COLLECTION_CHANGED` | `(collection, key, item)` | `ComponentIter.__setitem__` (`component_collections.py:78-87`), `ComponentIterator.append` (`:219-230`) |

`bus` is a process-wide `pyee.EventEmitter`; emitters guard on `has_listeners`, so there is
**zero cost with no listeners**. `watch(event, fn)` is a context manager, so listeners are scoped
and unhooked on exit (verified: `has_listeners` back to `False` after the block). The one consumer
today is `ProblemExec._on_structural_change` → `comp_changed` → `min_refresh`.

Two entry conditions the 1.0 graph layer must respect (both measured):

1. **`VALUE_CHANGED`/`SLOT_CHANGED` do not fire on every write.** `property_changed`
   short-circuits (`configuration.py:164-168`) when there is **no active `ProblemExec` session**
   *and* `instance._anything_changed` is already `True` — and a freshly constructed system has
   that flag set. So a naive `with watch(VALUE_CHANGED, fn): s.x = 5` outside a run sees nothing;
   clearing the flag first yields `[('x', 1.0, 5.0)]`, and inside `run(x=[1.0,2.0])` the same
   watch sees 2 events. This is exactly the "tree-global boolean" §9 is replacing, now visible
   as an event-delivery gap rather than only a caching one.
2. **`SLOT_CHANGED` has an identity guard** (`configuration.py:210-211`, `if value is not cur`)
   that `VALUE_CHANGED` does not: re-assigning the *same* component is deliberately silent,
   because `is_different` is `True` for every non-scalar assignment and that case dominates call
   volume. `COLLECTION_CHANGED` is emitted from the collection methods directly and is **not**
   gated by `_anything_changed` at all.

**Also landed: two new `save_mode` values.** `save_modes = ["vars","nums","all","prob","sparse"]`
(`problem_context.py:150`; default `"all"`, `:124`), dispatched in `ProblemExec.output_state`
(`:1795-1827`).

| mode | ref set | measured on a 3-point sweep, 1 slot, 1 solver var |
|---|---|---|
| `vars` | the run's variable refs | `(3, 2)` |
| `nums` | numeric attributes + properties (str fields dropped) | `(3, 8)` |
| `all` | every system reference (default) | `(3, 10)` |
| `prob` | **new** — only the solver's problem vars. Objective refs (`Yref`) are deliberately **excluded**: a compiled objective is a control-flow wrapper, and calling one outside the solver loop re-enters the solver and raises `ProblemExitAtLevel`. Previously declared-but-`NotImplementedError`. | `(3, 2)` |
| `sparse` | **new** — same ref set as `all`, but each row stores only the **delta** vs the case's prior row (`Case.record_sparse`); `dense_rows`/`dense_dataframe` forward-fill them | `(3, 10)` — single-row cases cannot compress |

`sparse` pays off on transients, which is where the row counts live. Measured on a 2012-row
spring-mass simulation (`dt=0.02`, `endtime=4`, same model both runs):

| mode | rows | stored values | `dataframe` | NaN holes |
|---|---|---|---|---|
| `all` | 2012 | 46276 | (2012, 23) | 4024 |
| `sparse` | 2012 | **19323** | (2012, 23) | 26955 |
| `sparse` → `dense_dataframe` | 2012 | — | (2012, 23) | **4024** |

**58% fewer stored values for the same frame**, and the round trip is faithful: `dense_dataframe`'s
NaN count matches `all`'s exactly, and the integrated states agree to 0.0 (`x`, `v`). Only `time`
(1.1e-3) and the time-dependent `Fext` (0.011) differ, an artifact of the two runs being
independent rather than of sparsity. This directly serves §9's "row-save dedup is restated over
the tabulated node set" and is the storage shape a u-vector tabulation (§15.2) will want.

**Also landed: a shell entry point.** `Analysis.cli(args=None)` (`analysis.py:81-164`) is an
argparse front end — `--input-json PATH`, repeatable `--set KEY=VALUE` (JSON-parsed),
`--save-mode {vars,nums,all,prob,sparse}`, `--no-plots`, `--list-inputs`. Precedence is
file → `--set` → `sequence`, verified end to end: a sectioned file of
`{"x":[1,2], "kp,ki":[[2,1],[8,4]], "comp.y":5.0}` + `sequence:[{"x":9}]` under
`--set "x=[3,4]"` yields `Result(4 cases)` with every `input_set` carrying `x=9` — `--set`
replaced the file's `x`, then the `sequence` row overrode it. `--list-inputs` prints fields and
slots and returns `None`. `Analysis.run()` now **returns** the Result (it previously returned
nothing), matching §15.5's "`Analysis.run()` returns the same type".

Note `engforge/__init__.py` exports **none** of the new names — `Result`, `Case`, `SystemState`,
`events`, and the `inputs` helpers are imported from their modules. 1.0 should decide this
deliberately rather than by omission.

**Toolbox decoupling completed:** core carries no `CostModel`/`Economics` knowledge — the slot
auto-override, the collection veto, and `_cost_update` are deleted. Toolboxes declare intent
explicitly (`Slot.define(X, none_ok=True, default_ok=False)`) and hook the update phase by
overriding `update()` on their own classes (engforge-econ does exactly this).

**Verification.** `conda run -n pace python -m unittest discover -s engforge/test -p "test_*.py"`
→ **`Ran 129 tests ... OK`**. (This paragraph previously claimed 92; the suite has grown with the
work below.) Four test files landed with this pass:

| file | tests | covers |
|---|---|---|
| `engforge/test/test_input_groups.py` | 15 | group-key parsing, zipped rows, partial-factorial point counts, `sequence` precedence |
| `engforge/test/test_state_results.py` | 18 | `SystemState` capture/apply/updated/copy/diff; `Case`/`Result` ownership, dedup, sparse round trip |
| `engforge/test/test_events.py` | 12 | the three events, `watch()` scoping, emission entry conditions |
| `engforge/test/test_cli_inputs.py` | 14 | `load_input_sets` both shapes, section validation, `--set` parsing and precedence |

**Also landed: one `Case` per sweep point** (this section previously recorded the 2N-case
defect as unfixed; it is fixed, and `examples/transient_control.py` — which crashed on it —
now runs end to end and reports `ran 4 transients (2 gain-sets x 2 profiles — not 8)`).
**Two** layers open a case per point — `solveable.py:402` (`session.begin_case(icur)` per
input point) and `dynamics.py:640` (`pbx.begin_case(...)` per episode) — so a sweep of N
points used to emit **2N** cases, with `input_sets` duplicated and half of
`result["output"]` as `None`. The fix is in case ownership rather than another patch, per
this part's thesis: `begin_case` (`problem_context.py:500-518`) **collapses** a trailing
case that never recorded a row and merges the new inputs into it, so the matrix point's
case and the episode's case are the same `Case` (an un-recorded case holds nothing worth
keeping, so collapsing is lossless, and merging the inputs keeps the row provenance —
matrix point + episode `X0`). Measured on the README's two-damping spring-mass sweep:

```
sim_matrix(dt=0.01, endtime=2, u=[0.0, 0.1], ...) -> Result(2 cases)
  input_sets  {0:{'u':0.0}, 1:{'u':0.1}}
  output      [DataFrame, DataFrame]
  rows/case   [1012, 1012]
  dataframe   (2024, 23)                                             # 2 x 1012
```

## V.N — Landed since: run input, naming authority, per-root resolution

A second pass landed the 1.0 run-input layer, the solver naming fix, an audit-driven cleanup, and
the first structural step of the per-root Problem (Part IX M-numbers in parens). Full suite green
throughout (151 core, 17 econ, 1 thermo, 7 examples).

**Run input (one structure for code + JSON/CLI).**
- `engforge/run_config.py` — `RunConfig(method, tol, options)`, a serializable (dict round-trip,
  YAML/JSON-safe) optimizer-settings object. It reaches scipy via `solve_min` (`setdefault`, so an
  explicit per-call kwarg still wins); empty default = scipy's own behavior. This gives the former
  dead `min_kw_dflt` a real, user-controllable home on the steady-state path.
- **Structured `run()`** — `run(inputs=, scope=, config=, sequence=, **shorthand)`, matching the
  JSON `{inputs, settings, sequence}` shape; the flat shorthand still routes into the right bucket.
  This makes the **state (inputs) vs settings (everything else)** split structural.

**One naming authority (the dotted-var bug).** `SolverInstance.get_alias` now returns the declared
var string, so a dotted slot var (`declare_var('comp.x')`) composes to the canonical dotted ref name
and actually solves — previously it keyed the leaf (`'x'`), mismatched the writeback space, and the
solver silently reported success at the start value. Leaf declarations are byte-identical.
`declare_var(on_missing='warn'|'error'|'skip')` handles a var whose target slot is empty, applied
once per problem at establishment.

**Audit cleanup (dead forks removed).** `_solver_override` (never set True; dead branch),
`SOLVER_OPTIONS` (no readers), the `Yref`/`sys_solver_objectives` dead store, the unused `norm_x`
X-normalization, and ~219 lines of commented-out root/Jacobian solver were deleted. Known bugs
fixed: the `con_info` double-append that mislabeled `Ycon`, the dead run-timing branch, and the
`_prob_levels` strong-ref leak. Deferred with evidence: `_anything_changed_` removal waits for the
Stage-4 dirty-set (three live cache consumers + cross-repo API); the var/bound combo-default
divergence needs the two combo forks merged (a naive fix regressed slider-crank).

**Per-root Problem — multi-root coexistence (toward the Part IX engine).** Each context carries an
explicit `parent` (the enclosing context; `None` for a root) with `is_root` and a `root` property
that walks the chain. Session resolution (`get_sesh`, `__getattr__` for the ~20 shared
`root_possible` names) follows the context's **own** parent chain to its root, not the process-global
`class_cache.session`. The active-root tracking is now a LIFO **stack**: entering a root pushes the
previous active root and exiting pops it back, so a later root no longer clobbers an earlier one, and
`problems_dict` is keyed by a unique id per context (no more `[True]` eviction). `run(root=True)` /
`ProblemExec(..., root=True)` forces an **independent new root** even while another is active — a
different system is allowed, since a forced root establishes its own. Verified (`test_problem_roots`):
two different-system roots A and B coexist in nested `with` blocks, each resolves to itself, A is
suspended on the stack while B is active, and B's exit restores A. This is the two-Pareto-front
capability.

The **routing principle** (ratified): a child branches from the root's `SystemState` as captured at
`run()` entry, overlays only its own inputs, and derives its own scope — the shared thing across the
tree is *state, not config*, the right model for differential/nonlinear systems.

**Remaining (2c):** a child still inherits the root's `_slv_kw` and does not yet build its own
problem definition, so per-child re-scoping (different combos per child) is the next increment; the
resolution/lifecycle groundwork for it is in place.

**Owned graph + cycle detection (Part IX M2/M4/M10 seed) — landed, report-only.** `engforge/problem_graph.py`
(`ProblemGraph`) builds the dependency DAG over a system's fields and properties, keyed by the dotted ref
name. Edges come from a static bytecode scan of each `@system_property` getter (the `self.<attr>` /
`self.<comp>.<attr>` read chains, 3.12 method-form aware) — an over-approximation, so a reported
"no cycle" is trustworthy. **Strongly-connected components of size > 1 are algebraic update cycles**,
`layers()` gives the feed-forward topological order, and `report()` prints both. Exposed as
`Problem.graph`; `run(cycle_check='warn'|'error')` applies the policy once at establishment (default
off → report-only, zero solve-behavior change). This is the direct answer to "check for update
cycles" — the core 1.0 convergence concern. Verified (`test_graph_cycles`): a feed-forward model
reports none with `comp.twoy` ordered before `s`; a model whose two properties read each other is
flagged as one SCC and `cycle_check='error'` raises on it. Enforcement (layer-ordered
execution replacing declaration order; stochastic-node poisoning; structure-key plan cache) is
Part IX M4/M8, deferred — it changes execution order and must be measured against the examples.

**ProblemState — the per-root state object (Part IX M8 seed) — landed, additive.** `engforge/state.py`
adds `ProblemState`, owned by each root (`Problem.state`): it holds the start checkpoint (a
`SystemState`) and, while its root is active, an **event-driven dirty set** of the
`(component, attr)` pairs changed since the last `clear()` — fed by `VALUE_CHANGED` on the bus,
subscribed at root enter and removed at teardown alongside the structural-event listeners. This is
the explicit, per-tree answer to "what changed" that will replace the tree-walking
`System.anything_changed` (a full recursive walk per read, load-bearing for `solver_cached`). It is
additive for now — it *records* changes; it does not yet gate caching. Removing `_anything_changed_`
still waits for this to drive the cache-invalidation path AND for lockstep migration of
engforge-econ / -structures / -thermo, which read/write the public `anything_changed` API (audit
#iii). Verified (`test_problem_state`): a value write marks its attr dirty, the checkpoint is a
`SystemState`, and the listener is removed when the root exits.

### What remains (behavior-changing / cross-repo — focused work per stage)

The additive per-root foundation is complete and green (165 core, 17 econ, 1 thermo, 7 examples).
The static part of the graph is now computed at **startup**: `TabulationMixin.cls_property_dependencies`
scans each `@system_property` getter's read-chains once per class (in `pre_compile`) and caches them
on the class, so `ProblemGraph` (renamed from the earlier `RefGraph`; `engforge/problem_graph.py`)
assembles the per-problem graph by stitching those cached chains along the instance tree — no
bytecode scanning per run.

**2c — per-child re-scoping: DONE (via `root=True`).** A re-scoping child is a nested root on the
same system with its own combo selection: it checkpoints the system's current state and derives its
own `Xref`/scope, and the outer scope is restored on exit. Verified (`test_problem_roots`): under an
active `combos='design'` context (`Xref={A}`), a nested `ProblemExec(sys, {'combos':'offdesign'},
root=True)` has `Xref={N}` — the routing principle (child branches state, derives own scope) realized
with no new concept.

The remaining stages each change runtime behavior or break an API and so are landed separately,
each measured against the full suite + examples:
- **3b** — layer-ordered execution (the graph's `layers()` drives `update`/signal order, replacing
  declaration order) + stochastic-node poisoning + a structure-key plan cache; changes evaluation
  order, so it is gated on example-by-example equivalence.
- **4 (completion)** — route `anything_changed`/`solver_cached` invalidation through the
  `ProblemState` dirty set and retire `_anything_changed_`, in lockstep with the toolboxes.
- **5** — hard-drop the `data_dict`/`dataframe` object surface (audit) with the econ keyed-assert
  migration to the `Result` path.
- **6** — subproblem hierarchy: a self-solving subsystem as an opaque `Y=F(X)` node, and transient
  inner solves re-entered against a compiled plan instead of recompiled per step.

---

# PART VI — Parallel problem contexts (analysis)

> **This part is ANALYSIS ONLY. No code from it has landed.** Nothing in `engforge/` was changed
> by this investigation; the repo is untouched. Part V above is the shipped record — this part
> is a measured answer to "can two `ProblemExec`s run at once, and do we want them to?" The
> experiment scripts are `e1_state_tree.py`, `e2_two_sessions.py`, `e3_threads.py`, `e3b_race.py`,
> `e3c_runrace.py`, `e4_copies.py`, `e5_pickle.py`, `e6_pareto.py`, `e7_groups.py`,
> `e8_threadlocal.py`, `e8b_tlmeta.py`, `e9_alias_rate.py`, `e10_timing.py` (sandbox/scratch, not
> shipped). The suite is green throughout (`Ran 129 tests ... OK`).

**Motivating case:** two Pareto fronts over one drivetrain (a light motor and a heavy one), each
swept over gear ratio and duty cycle.

## VI.1 What works today: one context, a tree of states

`SystemState` (Part V) is already a persistent-tree node, which is the parallelism primitive we
actually needed. `capture` reads every settable ref through the session, `updated(point)` returns
a **new** node (verified: `root.values is leaf.values` → `False`), and `apply` writes back through
`Ref.refset_input`. So branching and restoring is explicit and cheap:

```python
with ProblemExec(sys, {}, level_name="top") as px:
    root = SystemState.capture(px)              # 6 refs: a, b, comp, comp.name, comp.y, name
    for name, cfg in fronts.items():
        branch = root.updated(cfg)              # this front's root — a NEW node
        px.begin_case({"front": name})          # explicit row ownership (V)
        for gear in gears:
            for duty in duties:
                branch.updated({"gear": gear, "duty": duty}).apply(px)
                px.save_data(force=True)
        branch.apply(px)                        # restore the branch root
    root.apply(px)                              # restore everything
```

Measured: depth-2 branching round-trips **exactly** — `root.diff(after) == {}`, with `a, b, comp.y`
back to `2.0, 3.0, 1.0` and the derived `s` back to `312.0`. Slot components are *in* the state
(key `comp` holds the instance), so after `s.comp = Inner(y=99.0)` an `apply` restored the
**original object** (`s.comp is orig` → `True`, `y` back to `1.0`).

Both routes to the two-front problem were run and agree. One context with one `begin_case` per
front gave 2 cases × 8 rows = 16 combined rows; two *sequential* contexts over two
`copy_config_at_state()` duplicates gave 8 rows each. **The Pareto fronts were identical** —
light `[(1.0, 0.25, 12.5, 7.25) … (4.0, 0.75, 150.0, 24.75)]`, heavy
`[… (4.0, 0.75, 1200.0, 59.75)]`. Wall time for both fronts: **~5 ms**. There is no correctness
motive for parallelism here, only a throughput one, and throughput is not the bottleneck.

## VI.2 Why simultaneous contexts cannot work today

`ProblemExec.class_cache = ProblemExec` (`problem_context.py:2346`) and `__enter__` does
`self.class_cache.session = self` (`:837`). The session is therefore a **plain class attribute on
a module-level class** — not a `threading.local`, not a `ContextVar`. Grepping the module for
`threading.local`, `contextvars`, `ContextVar`, `threading.Lock`, `import threading` returns
**zero hits for all five**. `'session' in ProblemExec.__dict__` is `False` before enter, `True`
inside, `False` after — and a second thread reads the first thread's session (verified directly).

Two failure paths, both observed:

- **Guarded path.** Opening a context on a *different* instance while a session exists raises from
  `__init__:436-438` — `IllegalArgument: somethings wrong! change of comp!`. Nesting on the *same*
  instance does not create a second session at all: `__enter__:834` returns
  `self.class_cache.session`, so `with ProblemExec(A, level_name="child") as px` hands back the
  **top** object (`px is outer_sesh` → `True`), sharing one `Result`. **Levels are bookkeeping,
  not isolation.**
- **Unguarded path.** The guard only fires if a session exists *at construction time*. Construct
  two contexts first, then enter both: both are top-level (`_problem_id is True`, `:462`) and the
  second **silently clobbers** the class attribute. The first context's `__exit__` then does not
  clear it — `clean_context` (`:1067-1073`) only deletes when `session is self` — so the session
  **leaks past exit** and a later unrelated `C.run()` raises `IllegalArgument`. Worse, `inst_sesh`
  is cached lazily by `get_sesh` (`:600`, reached on the first `.sesh`), so each context keeps resolving `.sesh` to itself and the
  divergence is invisible from the inside.

Threads hit both paths at random. 20 barrier-synchronized trials, two threads, two instances:
**16 × `IllegalArgument`**, **3 × `AttributeError: type object 'ProblemExec' has no attribute
'session'`** (from `exit_action`), **1 × no exception but aliased state** — **0 / 20 clean**.
Concurrent `System.run()`, 15 trials: every trial had exactly one thread raise `IllegalArgument`.
The exact tally moves between reruns (the original pass measured 19/1 and 9-of-15); what is stable
is that the clean rate is zero and that *one* of the outcomes is silent corruption rather than an
exception. That is the dangerous one — it passes in dev and fails in CI.

## VI.3 Are duplicate entities required? Yes.

State lives in live attrs objects; `SystemState` is a snapshot *of* them, not a store the context
reads through. So two explorations of one instance alias each other. With sessions already isolated
(thread-local prototype) and a 1 ms window between set and read:

| entities | corrupt reads (per thread, 300 reads) |
|---|---|
| one shared instance | **298 / 300** and **298 / 300** |
| `copy_config_at_state()` duplicates | **0 / 300** and **0 / 300** |

An earlier shared-instance trial showed T1 writing `a=10.0` and reading back `a=20.0, s=30.0` —
wrong numbers, **no exception**. **Session isolation without entity duplication converts loud
failures into silent ones**, which is strictly worse. The invariant to document alongside any
isolation work is *one entity graph per context*.

The copy mechanism to use is `copy_config_at_state()` (`configuration.py:433`, recursive
`attrs.evolve` with an identity map): `B.comp is A.comp` → `False`, mutations do not leak either
direction, and `B.comp.parent is B` → `True`. **Avoid `clone()`** (`system.py:114`,
`copy.deepcopy`): `TabulationMixin.__getstate__` (`tabulation.py:53-58`) pops `parent`, so the
clone and its components end up with **no `parent` attribute at all**
(`hasattr(C.comp,'parent')` → `False`). It still runs, which makes it a landmine rather than an
error. `copy_system=True` inside a context works and leaves the original untouched.

## VI.4 Three options, ranked

**(i) Sequential state tree — available today, zero change.** Already correct and exact (VI.1).
Cost: nothing. Breaks: nothing. The honest answer at current problem sizes.

**(ii) Thread-local `session` + duplicated entities.** Prototyped. A naive
`class_cache = threading.local()` breaks exactly one line — `:827
isinstance(self, self.class_cache)` → `TypeError: isinstance() arg 2 must be a type`. Routing
`session` through a metaclass-backed thread-local **class** keeps `isinstance` legal, and then:
**25 / 25** concurrent `run()` trials exactly correct, 4 threads × 4 copies all correct. Fallout: a
cosmetic `change of execution class!` warning (`:828`), and `level_number` must stay shared because
instances read `self.level_number` for log formatting. Real cost: ~15 lines in core plus a
documented invariant. **This is the right eventual fix** — it changes a storage class, not a
concept.

**(iii) Process-pool fan-out.** Pickle decides it. On **this probe's model** — a plain
`System` with two fields and one slot, *no solver attributes* — a **fresh** system pickles
(534 B), and `SystemState` (434 B; 165 B numeric-only), `Case` (394 B),
`Result(system=None)` (607 B) and `dataframe` (1055 B) all pickle. But **after a run**,
`pickle.dumps(system)` fails — `AttributeError: Can't get local object
'Ref.setup_calls.<locals>.<lambda>'` — via `_last_context`; same failure for `Result` with
`system` attached and for `ProblemExec`.

That "fresh system pickles" result does **not** generalize, and the generalization would be
the dangerous read: a system carrying `Solver` attributes does not pickle at **all**, not
even freshly constructed, because `Solver.declare_var` generates a class per variable that
does not exist at module scope for the unpickler to find (`PicklingError: Can't pickle
<class 'engforge.attributes.Solver_var_<var>_<hash>'>`). Since every system worth fanning
out has solver attributes, the conclusion is the stronger one either way: children must
*build* their own system from a spec and return plain data. (Byte counts above are from one
probe run and are not stable — instance names are auto-generated at random length.) That works (`spawn` Pool
verified end to end). The problem is economics: 4 jobs × 144 points cost **0.70–1.15 s sequential
vs 5.8–6.8 s** in `Pool(4)` — a **0.12–0.17× "speedup"** — because the spawn+import floor is
**~4.8–6.5 s** while one job is **0.14–0.16 s**. (Ranges are across reruns on one macOS machine
with a trivial model; the floor is roughly fixed, so the crossover depends entirely on real model
cost.) Only worth it once a single job runs for seconds.

## VI.5 Judged against the project's rules

| rule | verdict |
|---|---|
| **Everything is a Component** | None of the options add a user-facing class. (ii) changes where `session` is stored; (iii) is a deployment topology. Neutral-to-good. |
| **Concept parsimony** | (i) adds **zero** concepts — `SystemState` already does the work, which is the strongest argument for it. (ii) adds one invariant ("one entity graph per context") and no vocabulary. (iii) adds a spec/serialization boundary, a real new concept, and should be resisted until forced. |
| **Combos as segmentation** | Decisive. Two Pareto fronts are two *segments of one analysis*, and combos + the grouped matrix key are the house tools for exactly that. Reaching for threads to express segmentation would duplicate a mechanism that already exists. |
| **Explicit over implicit** | The current `class_cache.session` singleton is the *most implicit* thing in the execution path — an invisible global whose clobbering is silent (VI.2). (ii) is therefore a parsimony *win*, not just a concurrency feature: it makes "which session am I in" answerable. The state tree (i) is explicit by construction (`capture`/`apply`/`begin_case`). |

Consistent with §15.2 (flat `u` can't carry revert alone) and §15.13 (selection can be
structural): per-context isolation is a **storage** question, which is what makes (ii) small.

## VI.6 Recommendation for the two-front case

**Don't parallelize.** Make it **one run** and segment the frame — the grouped matrix key
(Part V) already expresses "two fronts" exactly. Verified: 16 cases, 16 rows, 8 per front,
numerically identical to both the state-tree and two-copy answers:

```python
res = drive.run(**{
    "motor.mass,motor.kv": [(0.5, 100.0), (4.0, 100.0)],   # the two fronts, joint
    "gear": [1., 2., 3., 4.],
    "duty": [0.25, 0.75],
})
df = res.dataframe                      # run() returns the Result (Part V)

light = df[df["motor.mass"] == 0.5]     # 8 rows
heavy = df[df["motor.mass"] == 4.0]     # 8 rows
```

Escalation ladder: reach for the explicit `SystemState` tree (VI.1) only when fronts need
*different sweep grids*; adopt thread-local sessions (option ii) when wall time actually hurts,
and only together with `copy_config_at_state()` duplicates; fan out to processes only when one
front's run takes seconds, not milliseconds.

## VI.7 What was NOT verified

Thread-safety of the **solver** under option (ii) — the prototype exercised `run`/`eval`/`sys_slvr`,
not scipy `minimize` paths, transients (`dxdt=True`), signals, `ComponentIterator`, or the new pyee
bus (whose `bus` is **also process-wide**, so listeners are a second shared global with the same
class of problem). Whether `attrs.evolve`-based copies preserve dynamics state
(`setup_global_dynamics`), plot attributes, or `Analysis` report wiring. Whether `problems_dict`
(a `WeakValueDictionary`, `:217`) misbehaves under concurrent mutation. Whether the `clone()`
`parent` drop breaks something not exercised here. Timings are one machine, one trivial model.

---

# PART VII — The ref space: where the DAG meets the solver matrix (measured)

> **Status: measurement + git-ignored prototype. Four property/caching bug-fixes landed in core
> with tests (`test_properties_cache.py`).** This part answers one design question — *every component
> is a black box `Y = F(X)` from its state X to its properties Y; where does the "solver matrix" live,
> and can we bring the refs into one space the DAG operates on?* Everything below was run, not
> asserted. Prototypes: `sandbox/ref_space.py`, `sandbox/dp_eval.py`, `sandbox/viz_space.py` (all
> git-ignored, `.gitignore:27`). Measurement harness: background workflow `wf_1e8290be-d7b` plus a
> determinism-classifier pass.

## VII.1 The gap, stated plainly: the solver matrix is not a matrix

`refmin_solve` (`solver_utils.py:219-264`) hands scipy `vars = list(Xref.keys())` — an *ordered dict
of keys* — and calls `sciopt.minimize(Fc, Xo, **kw)` with **no `jac=`, no `hess=`, no sparsity**
(captured at the call: the only kwargs reaching scipy on the steady-state path are
`['bounds','constraints']`). scipy therefore finite-differences everything itself: one-sided 2-point,
step `h ≈ 1.49e-8·max(1,|x|)` (verified empirically — `h=1.49e-8` for `|x|≤1`, `2.73e-7` for `x=18.3`).
No code anywhere computes a Jacobian (`calcJacobean` is commented out, `solver_utils.py:568-613`; the
standing TODO is `:258`). And x-normalization is dead: `norm_x = handle_normalize(...)` is assigned at
`:246` and **never read again** (`inspect.getsource(...).count('norm_x') == 1`), so a variable of SI
magnitude 1e-6 silently gets a **1.5% perturbation reported as a derivative**.

So the refs never enter a space. They are ordered keys into a black-box closure. Meanwhile the
inspected DAG lived in a separate `sandbox/problem_map.py` pass with its *own* node identity, built
from class inspection rather than the live session. That disconnect is exactly what "we were bringing
the refs into a space (or so I thought)" names. VII builds the connection.

## VII.2 The model: `Y = F(X)`, measured

Per component, state X (settable fields + solver unknowns) maps to properties Y through
`@system_property` bodies, composed across slots, collections, and `Signal`/`update()` broadcasts.
The empirical oracle (`SystemState.capture` → perturb one x → re-read every property → bit-exact
diff → `SystemState.apply` restore, verified exact) gives the true dependency matrix `D[y][x]` and its
magnitudes (= the measured Jacobian). Caveats found by perturbation, not assumed:

- **Signals create dependencies no property body contains.** `update()` writes sibling fields
  (`for b in self.branches.data.values(): b.dP = self.dP`) — a cross-component edge invisible to any
  reading of `fget` alone. Perturbation reveals it; a static body scan does not.
- **`run_id` / `converged` read solver *history*, not X** (`system.py:77-95`). They are not `F(X)` and
  must never enter a cached Y. At rest both read `None`; during a run they are history-dependent by
  design.
- **`cache_dXdt` writes the dynamics matrices onto `self` from inside a property read** — a real
  state-in-a-getter smell, flagged for the M8 memoizer to eventually subsume.
- Across **105 properties / 23 classes** (test components + the four examples), **user model math is
  100% deterministic**; every flagged property is framework-inherited.

## VII.3 The bridge: one graph, node identity = the session's dotted ref name

`sandbox/ref_space.py:build_space(ctx)` takes a live `ProblemExec` and returns one `networkx.DiGraph`
whose nodes are keyed by `sesh.all_refs['attributes'] | ['properties']` — the same flattening
`all_system_references` does, one naming authority. Edges come from a **hybrid**: `@system_property`
descriptor-stack tracing for property→property, the perturbation oracle for field→property and
field→field (i.e. `update()`/signals). Roles come from the solver's own ref sets, never re-inspected:
`solver_x ← all_problem_vars`, `residual/inequality ← problem_eq/problem_ineq`, `objective ← Yref`.
**The solver matrix is then the induced submatrix**: predicted Jacobian sparsity = "can this Xref node
reach this residual row in the graph."

Predicted sparsity vs measured finite-difference, in the solver's own `Xref`/`Yref` order:

| model / partition | solver TP/FP/**FN** | full-D TP/FP/**FN** |
|---|---|---|
| CubeSystem `[combos=*, signals OFF]` | 20/0/**0** | 110/0/**0** |
| CubeSystem `[combos=*, signals ON]` | 100/24/**0** | 107/0/**0** |
| SpringMass `@ v=2,a=1,t=0` | 2/0/**0** | 30/3/**0** |
| SpringMass `@ t=0.7` | 2/0/**0** | 33/0/**0** |
| Spool `[combos=DESIGN]` | 2/0/**0** | 36/0/**0** |
| Spool `[combos=OFFDESIGN]` | 4/0/**0** | 36/0/**0** |
| Spool `[combos=design,offdesign]` | 6/0/**0** | 36/0/**0** |
| Manifold `[combos=balance]` | 4/0/**0** | 22/0/**0** |

**Zero false negatives across all 8 configs.** (False *positives* on CubeSystem-signals-on are safe —
they over-cover; a few `wrap_f` function-valued declarations are unnameable and reported, not guessed.)
The combos result is the important one: selecting `DESIGN` vs `OFFDESIGN` on the same Spool gives
**different X nodes (hardware `A_t,S_t` vs operating `N,W`) with the same two residual nodes** — read
from the live session, confirming combos is the segmentation lever (Part IV) with nothing new added.

A defect surfaced building the join: the naive bridge `get_parent_key(decl)+ref.key` points a
cross-component constraint (`x_lim = con_ineq("x","comp.x")`) at the *system's* `x` instead of
`comp.x`. **Any RefSpace must resolve a solver term through `ref.comp`, not the declaration's parent
key.** Caught by predicted-vs-measured, fixed in the prototype.

## VII.4 Inspection: static vs value-callback, and the sound hybrid rule

Two methods, each incomplete alone; the union is complete on the models tested:

- **Static bytecode** (`dis`) is the skeleton, **not a superset**: it over-approximates branches (safe)
  but under-approximates loops, helper calls, and `getattr` (unsafe — misses real edges).
- **Value-callback tracing** is path-exact, **therefore unsound by construction** for any cache that
  outlives a branch flip: its map is valid only under the guard conjunction it was recorded on.

The sound hybrid (what the M2/M3 engine must assume): static skeleton + traced confirmation, with an
explicit **unknown → depends-on-everything fallback** that is **transitive** (poison the forward
closure). A foreign C/ufunc call silently converts a dependency into a constant — force the dep set to
static and void any affine claim. **The residual is `update() ∘ fget`, not `fget`** — trace from the
solver's entry with `update()` inside the traced region, or every sibling-shared unknown is invisible.
**Invalidation must precede `update()`** — no dependency map fixes the wrong order.

## VII.5 Dynamic programming: real in eval-counts, break-even in today's wall clock

`sandbox/dp_eval.py` memoizes evaluation over the DAG and, for one finite-difference probe that
perturbs one x, recomputes only that x's reachable cone. Jacobian agreement with the naive full
re-evaluation is **bit-exact (max |Δ| = 0.0, no tolerance needed)** — and the harness has teeth: three
deliberately-crippled variants produce provably-wrong Jacobians (stale-cache corrupts a field;
non-transitive poison → 1/14 cells wrong; no fallback → 2/14 wrong).

Speedup, counted `fget` calls (the honest metric; `cse` = memoize-within-one-sweep, needs no graph):

| model | naive | cse | dp | dp/naive | dp/cse |
|---|---|---|---|---|---|
| CubeSystem | 246 | 126 | 68 | 3.62× | 1.85× |
| Spool design | 26 | 12 | 4 | **6.50×** | 3.00× |
| Spool offdesign | 26 | 12 | 14 | 1.86× | **0.86×** |
| Manifold N=3 → N=100 | 30→612 | 14→208 | 9→106 | 3.3→**5.8×** | 1.56→1.96× |

DP removes **35–52% of a whole solve's property evaluations** and the advantage **grows with N**
(saturates ~2× over CSE, bounded by the densest column). Spool-offdesign is the honest counter-case:
DP is *worse* than CSE there (0.86×) because `update()` writes `comp.N` then reads `comp.PR`, which DP
correctly recomputes and CSE serves stale. **But wall clock is break-even today** (DP 0.74–1.03×):
the properties are too cheap (bare `fget` median 0.056–0.122 µs) and `pre_execute()`/`update()`'s
attrs-broadcast is ~91% of a probe — a floor DP cannot touch. Measured **crossover ≈ 0.5–1 µs per
property**; above 1 µs DP wins 1.4–2.2× wall-clock, above 10 µs 3.6–3.8×, converging on the count
ratio. Graph build is 0.25–2.95 ms, amortized per *class*.

**Invalidation is event-driven, not statically mapped.** `engforge/events.py` (Part V) already emits
`VALUE_CHANGED` on every attrs write, so the evaluator *observes* `update()`'s sibling writes rather
than trying to predict them — `dP=36.5` on Manifold emits the write plus three broadcasts, and
"invalidate before `update()`" is satisfied for free. Measured unsound modes to respect in core:
mid-solve collection mutation needs `min_refresh` (the `COLLECTION_CHANGED` hook); the clock
(`self.time`) and `last_context` readers have no graph node and must stay never-cached.

## VII.6 Property/caching defects fixed in core (landed)

The determinism pass found four latent bugs by execution; all are fixed with tests
(`engforge/test/test_properties_cache.py`, 7 tests):

1. **`stochastic=True` was dead code.** `Configuration.pre_compile` (a no-op) shadowed
   `TabulationMixin.pre_compile` in the MRO of both `System` and `Component`, and that Tabulation hook
   is the flag's *only* consumer. Fixed with `configuration.run_pre_compile(acls)` running **every
   distinct `pre_compile` hook in the MRO** (deduped by function identity), at both `forge` call sites.
2. **`cached_system_property` crashed on first access** — `gname` was set only by
   `engforge_prop.__init__`, never by the `system_property` family, so `private_var` raised
   `AttributeError`. Fixed by setting `self.gname = fget.__name__` in both `__init__` and `__call__`.
3. **The aliases lied**: `cached_sys_prop`/`cached_system_prop` pointed at the *uncached*
   `system_property`. Repointed at `cached_system_property`.
4. **Class-level access crashed** — `solver_cached`/`instance_cached`/`class_cache` asserted on a
   `None` instance. Fixed with an `if instance is None: return self` guard.

A regression from (1) was root-caused and removed: the old `pre_compile` also assigned
`cls._anything_changed = True`, which on the subclass **clobbered `System`'s aggregating
`_anything_changed` property** and silently killed transient row recording (every row after the first
deduped away — surfaced as `test_dynamics_spaces` recording 1 row instead of 1000). The class default
already provides the flag, so the assignment is gone. The stale `test_tabulation.test_dataframe`
assertion (which vacuously compared a `solver_cached` dict to itself) now honors the live `stochastic`
contract. Full suite: **136 green**; econ 17, thermo 1.

**Determinism policy (recommended, concept-parsimonious).** Honor the existing `stochastic=` kwarg as
the declaration lever — no new ontology. Default: undeclared + statically suspicious ⇒ treated as
stochastic (sound-by-default; FPs cost only performance). A stochastic node is an **always-dirty
source that poisons its descendant cone** through the ref DAG. CRN-freezing = promote the seed to an
explicit X field (probe-F's existing pattern: seeded FD recovered −2.0256 ×5 identical vs a 212-wide
noise span unseeded). Statefulness is a third verdict surfaced as a warning, never a cache policy.

## VII.7 Visualization

`sandbox/viz_space.py` renders the ref space from a live session: per-model PNGs (`viz_cube.png`,
`viz_cube_signals.png`, `viz_spool_design.png`, `viz_spool_offdesign.png`) in a custom multipartite
layout where column = measured dependency depth, with free fields (grey squares), solver unknowns
(blue diamonds), properties (circles), residuals (orange triangles), objectives (aqua hexagons), and
the aqua-dashed `update()`/signal broadcast edges that exist in no property body. Each model also
emits an ASCII sparsity matrix (Y rows × X cols, terminal-readable) and a `.mmd` mermaid export. The
drawn graph's reachability equals the raw measured oracle exactly on all four models.

## VII.8 What it would take to land in core (tied to Part IX, no new concepts)

This is the staging of Part IX's existing milestones, not a new design:

- **M2 (graph extraction, §4)** — promote `ref_space.build_space` into `engforge/core/`: the hybrid
  static-skeleton + oracle, node identity = dotted ref name, roles from the solver's ref sets. The
  `ref.comp` resolution fix (VII.3) is a prerequisite.
- **M3 (value tracer, §5)** — the path-exact tracer with stored guards; the hybrid rule in VII.4 is its
  soundness contract.
- **M6 (derivatives, §7)** — hand scipy the predicted sparsity + semi-analytic Jacobian instead of
  letting it finite-difference blind; this is where the eval-count win becomes a real-model speedup.
- **M8 (incremental recompute, §9)** — the DP layer: dirty-push memoization driven by `events.py`
  (already landed, Part V), with transitive stochastic poisoning and the `COLLECTION_CHANGED →
  min_refresh` invalidation hook.

Combos still segment (design vs offdesign is one graph, two X partitions); everything stays a
component; the two structures this needs — the event bus and `SystemState` — already exist.

---

# PART VIII — Test & evaluation strategy

How candidate approaches graduate into core: reference problems with known answers, measured.

**Reference problems** (one per class): hinge fit (1); 2×2 spool balance, root (0.6, 0.75), with
design/off-design partitions (2 — P-A/P-E are its seed); two-discipline fixed point, ρ=0.9025,
fixed point (20, 20) (3 — P-B); 1-D conduction mesh vs linear profile (4); 3-site weighted design
with hand-computed optimum (5); noisy quadratic, true grad −2.0 at d=2, CRN-recoverable (6 —
P-F); recorded spring-mass trace, recover (k,c)=(10, 0.6) (7 — P-G).

**Candidate definition/extraction approaches under evaluation** (measured head-to-head, P-H):
- (a) **plain Python + value tracer** (baseline): zero user-code constraints; exact dependency
  sets always; exact Jacobian rows for affine terms only (P-H: r2 row {N: 1.1, W: 0.4} exact;
  nonlinear r1 → FD/CS). Branch guards recorded.
- (b) **opt-in symbolic (sympy) definitions**: incidence, coupled-block (SCC) identification, and
  the exact Jacobian fall out *by construction* (P-H: J = [[4N+1/2, 7/10], [11/10, 2/5]], Newton
  6 iterations to the exact root; symbolic stencil rows ±1/(x_p−x_m)). Costs: branches require
  `Piecewise`, sympy-only math, two-world friction. Positioning: **top-down component DSL +
  automatic DAG/cycle/solver identification is ModelingToolkit.jl's territory and empty in
  Python** — symbolic and tracing are the two routes; evaluation decides the mix (expected:
  tracer default, symbolic opt-in per property/residual).
- (c) **decorator-marked methods + combos** (`@Solver.residual(combos=…)`) for cycle
  participation — the house idiom.

**Metrics**: correctness vs reference; convergence robustness (iterations, failure rate under
scaling/perturbation); wall-time at N; model LOC; concept count added.
**Gate**: nothing enters core without reference-problem numbers. Harness:
`sandbox/cycles/run_probes.py` (P-A…P-H), extended per candidate.

---

# PART IX — The compiled-graph engine design

*The engine that implements Parts I–VIII: everything below compiles the declarations into one
problem graph (R(u)=0), which is precisely what delivers C1 (topological order), C2 (SCC blocks
with real solvers), C4/C5 (members and peer slots as graph nodes), and C7 (point-mapped
subproblems). Sections retain their original numbering.*

## 1. Why 1.0

engforge 0.x has one root defect, already diagnosed in the codebase's own TODO
(`solveable.py:636`): **there is no owned problem graph**. Everything recursive is re-derived
from live tree walks with string keys, every time:

- `ProblemExec.__enter__` rebuilds the full reference table with `recache=True`
  (`problem_context.py:631-665`, `full_update` hardcoded True at `:211`) — and a context is
  entered on **every objective and constraint evaluation** (`solver_utils.py:65,200`).
- Signals and updates run in **declaration order** with no dependency graph
  (`problem_context.py:1811-1833`); grandchild `update()` is never collected (`solveable.py:135`).
- Caching gates on one **tree-global boolean** whose read is itself a tree walk (`system.py:126`).
- The whole model flattens into **one scipy SLSQP problem** with
  `finite_diff_rel_step=0.25` (`problem_context.py:79`) — a 25% secant, not a derivative,
  chosen to average over nested-loop noise.
- Mistyped dotted keys are silently dropped (`refset_input(..., fail=False)`,
  `problem_context.py:1780/1791/1797`).

The symptoms — fragile recursive argument handling, surprising signal behavior, slow sweeps,
optimizer wandering — all trace to this. 1.0 keeps what is genuinely differentiated (the attrs
DSL, automatic tabulation, run-time var/combo selection, integrated plotting/reporting/data) and
replaces the execution core with a **compiled problem graph** and a **true hierarchical solver**.

**Research base** for this design: Hwang & Martins, *MAUD*, ACM TOMS 44(4) 2018; Gray et al.,
*OpenMDAO*, SMO 2019 + theory manual; Modelica/ModelingToolkit structural compilation
(matching → SCC/BLT → tearing → Dulmage–Mendelsohn); salsa/Adapton incremental computation;
pyomo `incidence_analysis`; SparseConnectivityTracer.jl. Full citations in §17.

## 2. Architecture overview

```
           attrs DSL (unchanged surface)
  forge / Slot / Signal / Solver / Time / @system_property
                      │  class-decoration + first bind
                      ▼
        ┌─────────── COMPILE ────────────┐
        │ extraction: declared ∪ static  │   M2, M3
        │ (dis/AST) ∪ value-tracer probe │
        │ alias-eliminate signals        │   M4 P2
        │ implicit core: match (HK)      │   M4 P5
        │ Dulmage–Mendelsohn diagnostics │   M4 P6
        │ Tarjan SCC → BLT ordering      │   M4 P7
        │ tearing / solver assignment /  │   M4 P8-P10
        │ coloring                       │
        └──────────── ▼ ────────────────┘
              immutable Plan  (typed steps, printable, cached by structure key)
                      │
        ┌──────────── RUN ───────────────┐
        │ flat u vector + views          │   MAUD §4.4
        │ dirty-push + memoized topo     │   M8 (the DP layer)
        │ eval; per-block solvers        │   M5
        │ (GS sweep / Newton / LU)       │
        │ semi-analytic derivatives via  │   M6 (UDE)
        │ UDE → analytic jac to driver   │
        └──────────── ▼ ────────────────┘
       tabulation = row slices of u (+ shadow)   →  pandas / plots / datastores
```

One sentence: **attrs ergonomics on the outside, `R(u) = 0` on the inside** — exactly the split
OpenMDAO describes for its own implicit transformation, but with the dependency graph *inferred*
from user code instead of hand-declared, which is engforge's structural advantage (properties are
transparent Python on typed attrs classes; OpenMDAO's `compute()` is an opaque black box).

## 3. Foundation: everything is a residual (MAUD)

MAUD concatenates all quantities into one vector `u = (x, y, f)` and gives every variable a
defining residual (Hwang & Martins Eqs. 8–13):

- inputs: `r = x − x*` (running the model at a value *is* a residual — this is what makes
  `df/dx` a block of `(∂R/∂u)⁻¹`)
- explicit states: `r = y − Y(...)`
- implicit states: `r = −𝓡(...)`
- outputs: `r = f − F(...)`

### The engforge mapping (M1)

| engforge construct | residual row | notes |
|---|---|---|
| plain attrs field | `x − x*` | auto-IVC; no special parameter concept |
| `Solver.declare_var` | `x − x*`, `x*` owned by the driver | design variable |
| `@system_property` | `y − F(...)` | explicit; Jacobian row `−∂F` |
| `Signal.define(tgt, src)` | `tgt − src` | a **connection/alias**, not an imperative copy |
| `Solver.con_eq` | user residual | implicit |
| `con_ineq` / objectives | driver-level rows | carried separately, not in `R` |
| `Time` / dynamics | `ẋ − f(x,u)`; A,B,C,D are exact Jacobian blocks | `dynamics.py` already has them |
| callable bounds (`add_var_constraint(lambda s,p: …)`) | `bound` node kind | re-evaluated per outer iteration; see risk §15.8 |
| overridden `update()`/`post_update()` | update node with declared/harvested write set | see risk §15.9 |

Two consequences:
1. **Signals stop being imperative.** A feed-forward chain is satisfied by one topologically
   ordered sweep (same cost as today's copy, now correct by construction and differentiable).
   A signal cycle is detected and becomes a fixed-point block — not a silent mis-ordering.
2. **Properties are pre-causalized** (one output, non-invertible Python), so bipartite
   **matching applies only to the implicit core** — `con_eq` + declared vars + genuine cycles:
   tens of nodes, not thousands. This is the decisive simplification vs a Modelica compiler and
   what makes a pure-Python implementation comfortable.

### Derivatives: the Unified Derivative Equations

All total derivatives come from one linear system and its transpose (MAUD Eq. 33):

```
(∂R/∂u) (du/dr) = I        forward — cost ∝ #selected design vars
(∂R/∂u)ᵀ (du/dr)ᵀ = I      reverse — cost ∝ #selected objectives+constraints
```

Chain rule, direct method, and adjoint are special cases. Users (and the framework's own
extraction) supply only **partials**; totals are one factorized linear solve. Run-time var/combo
selection = choosing **seed vectors**, not re-flattening the problem — only structural changes
recompile (see §9).

## 4. Graph extraction (M2) — hybrid, tracing-first, auditable

| layer | mechanism | role |
|---|---|---|
| **L0 declared** | Slot/Signal/Solver/Time declarations; `@system_property(depends=[...], depends_extra=[...], opaque=True, trace=False)` | ground truth where present; escape hatch |
| **L1 static** | `dis.get_instructions(fget.__code__)` scan for `LOAD_FAST self; LOAD_ATTR…` chains (recurse into `co_consts`); AST pass with beniget def-use for aliasing + helper-method closure; resolve dotted chains against **declared Slot types** | all-branch structural superset; works without source; `getattr`/`vars`/`__dict__` on self ⇒ escalate |
| **L2 value-tracer probe** | see M3 below | exact deps per path + guards + affine rows + units |

Edges = L0 ∪ L1 ∪ L2 with **per-edge provenance**; `problem.explain_graph()` prints every
property's dep-set with its source (declared/static/traced/probed) and flags disagreements —
static-only edge = dead code or unprobed branch; traced-only edge = dynamism the static pass
missed. `verify_graph` mode re-traces every k-th solve out-of-band (default on in tests).

Prior art: nbsafety/ipyflow (dynamic+static hybrid for notebook lineage; dynamic slicing beat
static by 54–66% on real sessions), param/Traits (declaration-only — rejected: 40 hand-maintained
dep lists kills the DSL), OpenMDAO (no inference — forced by opaque components; the moment they
could introspect, with JAX-traceable components in 3.37+, they shipped auto-AD).

## 5. The value tracer (M3) — "refs → linear algebra", made sound

During the probe (never in solves), fields return `Tracer(primal, deps, affine)` — a number type
with dunder arithmetic. Deps ride on **values**, so helper methods, local aliases
(`c = self.box`), and intermediates are captured for free.

- `+ − * /` and numpy ufuncs merge dep sets and compose **affine forms** `(const, {ref: coeff})`.
- **Branch guards recorded**: `__gt__`/`__lt__`/`__bool__` record the comparison's input deps as
  a *guard edge* before returning the primal bool. Per-path tracing stays sound: any input change
  that could flip a branch flows through a recorded guard, which triggers re-trace. Guards also
  tell the compiler which nodes have state-dependent structure.
- **Strict mode** (`trace="global"`): no primal, comparisons raise ⇒ branch-free evaluation ⇒
  recorded structure valid for **all** inputs (SCT.jl's global tracer). For
  sparsity-critical nodes.
- Any nonlinear op (`**`, `sin`, tracer×tracer, …) degrades the output to `NONLINEAR` while
  keeping exact deps.

**Payoff**: affine properties — mass/cost/area/power rollups, moment sums, signal chains — yield
**exact A/B/C/D rows with zero FD and zero AD**, feeding `dynamics.py`'s state-space machinery
directly. One probe pass yields four artifacts: dep edges, guard edges, affine rows, inferred
pint units (§8). ~300 lines; precedent: Pyomo `generate_standard_repn`, CasADi `which_depends` —
but on plain Python floats with zero user-code constraints. Guard-region validity and probe
construction caveats: risks §15.1, §15.10.

Rejected as the *primary* representation: CasADi/sympy/cvxpy (truth-value wall on `if`,
`if_else` Jacobian NaN-poisoning — casadi #1805/#1792, DCP restrictions) and mandatory JAX
(control-flow/mutation constraints, heavy dep). JAX stays as an opt-in exact tier (§7).

## 6. Structural compile pipeline (M4) and hierarchical solver (M5)

### Compile (once per structure key; cached, printable, diffable)

```
P0  name resolution        dotted ids → frozen accessor table (Refs built ONCE)
P1  dependency DAG         from §4
P2  alias elimination      collapse Signal chains (union-find) — the one Modelica
                           symbolic pass worth keeping; kills plumbing-only SCCs
P3  cycle check            explicit-part cycles → fixed-point blocks (not errors)
P4  implicit core          equations F = con_eq + dxdt residuals + P3 loops;
                           unknowns X = declared vars + loop vars + dynamic states;
                           incidence by reverse reachability on the DAG
P5  matching               nx.bipartite.hopcroft_karp_matching  (O(E√V))
P6  Dulmage–Mendelsohn     ALWAYS — the error-message engine (see below)
P7  SCC → BLT              nx.condensation → nx.lexicographical_topological_sort
                           (deterministic; never trust SCC yield order)
P8  tearing                per block > ~8 vars; greedy Cellier-style; verified with
                           is_directed_acyclic_graph; monolithic fallback always kept
P9  solver assignment      table below
P10 Jacobian coloring      scipy.optimize._numdiff.group_columns per block
P11 emit Plan              immutable typed steps + ASCII BLT dump + relevance
                           bitsets + per-block fallback ladder
```

Compile cost is matching-dominated, effectively linear in incidence nonzeros. The only NP-hard
step (optimal tearing) is approximated greedily and **verified** in linear time.

### Dulmage–Mendelsohn as the diagnostics layer (P6)

| symptom | DM signature | message |
|---|---|---|
| two `con_eq` fighting over one var | overconstrained rows | "constraints X, Y both determine `box.lo`; drop one" |
| var nothing depends on | unmatched/underconstrained col | "`fan.rpm` influences no objective or constraint" |
| square but singular | structural rank < n | "structurally singular: eqs {…} ↔ vars {…}" |
| healthy design problem | square + n−m free cols | "DOF = 3: {…} — the optimizer's search space" |

Port the `RowPartition`/`ColPartition` shape from `pyomo.contrib.incidence_analysis`
(~150 lines of networkx). The underdetermined part *is* the design freedom; the overdetermined
part is where residual-minimization objectives belong. Degeneracy policy in 1.0:
`strict_structure=False` (warn + witnesses; drop zero-incidence trivial rows with a warning
naming the declaration site) — see risk §15.4.

### Per-block solver assignment (P9)

| block | solver |
|---|---|
| 1×1 explicit | direct evaluation (no solve) |
| 1×1 residual | `brentq` if bracketed, else Newton/secant |
| k×k linear (affine rows!) | assemble, `splu`, cache factorization |
| k×k nonlinear, small | dense Newton (+ bounds line search) |
| torn nonlinear | Newton on t ≪ k tears, forward-substitution sweep |
| weak coupling (signal loops) | fixed-point + Aitken/Wegstein |
| objective + inequalities | driver (SLSQP/trust-constr/IPOPT) **at this level**, converged model inside |

Every node implements MAUD's five-method interface (`apply_nonlinear / solve_nonlinear /
apply_linear / solve_linear / linearize`), so solvers compose by recursion and a preconditioner
is just another `solve_linear`. **Default linear solver = assembled sparse LU** — OpenMDAO's
measured 1.5–2 orders of magnitude lesson against matrix-free recursion in Python. Block GS vs
Jacobi is transfer placement (MAUD Algs. 2 vs 3); Jacobi groups mark parallel/vectorization
opportunities (combo sweeps = independent replicas = one seeded solve for all, via coloring).

Solver assignment is **automatic from structure** (MAUD and OpenMDAO both make you choose by
hand — this is where engforge exceeds them), overridable per node:
`Slot.define(..., nonlinear_solver=..., linear_solver=...)`. Per-block MDF-vs-IDF policy:
nested solve vs lifting tear variables to driver equality constraints.

## 7. Derivatives (M6 — revised per adversarial review)

1. **Affine rows** from the tracer — exact *within their guard region* (risk §15.1), free.
2. **Default: colored sparse FD** with per-variable `ref`-derived steps
   (`scipy.optimize._numdiff.approx_derivative(sparsity=(structure, groups))`) — never 0.25.
3. **Complex step: opt-in and hard-gated.** CS silently returns **zero** through `abs()` — and
   the existing fixtures use `abs()` everywhere (`volume = abs(x*y*z)` etc.). Gate at compile:
   probe with a complex seed, assert `Im(out) ≠ 0` per known dep; opcode-blacklist
   `abs/max/min/np.clip/floor/comparisons`. Demoted rows fall back to FD, recorded in
   `explain_graph()`.
4. **Opt-in `jax=True`** per property/component: register Component as a pytree
   (`KeyPath` ≅ dotted Ref keys) → Jacobians keyed by Ref path. OpenMDAO 3.37+/3.40 precedent.
5. Assemble sparse `∂R/∂u`, factorize once in `linearize` (`splu` gives the transpose solve
   free → forward and reverse cost the same), totals via UDE, mode chosen by `n_dv` vs `n_out`.
   Hand the driver an **analytic `jac`** (SLSQP has no `jac_sparsity` — we compute, it consumes).
6. **`check_partials` with an independent oracle**: real central FD with a step-size sweep
   (Richardson) — never CS-verified-by-CS.

## 8. Scaling & units (M7)

Because `R(u)` concatenates metres, pascals, and ratios, `‖r‖` is dimensionally meaningless
without scaling — this is a **correctness requirement for the convergence test**, not a tuning
nicety (the MAUD paper itself is silent on scaling; OpenMDAO's practice layer is the answer).

- Per variable: `ref / ref0 / res_ref` (defaults: numeric bounds → `ref0=lo, ref=hi`; else
  `|default|`; `res_ref = ref`). `scaled_u = (u − ref0)/(ref − ref0)`, `scaled_r = r/res_ref`.
- **Invariant**: user code, tabulation, and the driver's reported values always see *physical*
  values; solver iteration, norms, and transfers always see *scaled* values.
- One affine `(scaler, adder)` pipeline implements **both** scaling and unit conversion
  (OpenMDAO's design). Driver-level scaler/adder stays a separate, non-interacting layer.
- Tolerance ladder defaults: `tol_nl=1e-10 ≪ tol_lin=1e-8 ≪ tol_opt=1e-6`, warned if misordered.

**Units (pint)**: `field(..., metadata={"units": "kg"})` — units are class metadata; instance
values stay plain floats (pint Quantity arithmetic is ~40× float: 8.24 µs vs 214 ns — it must
never enter the hot loop). At compile, pint computes `(factor, offset)` per edge (affine units
like °C need both) and raises `DimensionalityError` **at graph build with the dotted path**.
At boundaries (construction, `run(**kw)`, IO), attrs converters accept `Quantity` or float.
The §5 probe **infers every property's units for free** — automatic correct units in reports,
and dimensional bugs in property code caught at compile. Single
`pint.get_application_registry()`; units serialized as strings; plain-float DataFrames with a
units sidecar (pint-pandas stays opt-in).

## 9. Incremental recompute (M8 — the dynamic-programming layer, revised)

- **Write events**: attrs `on_setattr` (`property_changed`) repointed from the global boolean to
  the graph → eager dirty push to transitive dependents only. **Half of this landed** — see
  Part V: `engforge/events.py` makes writes, slot swaps, and collection mutations *emit*
  (`VALUE_CHANGED` / `SLOT_CHANGED` / `COLLECTION_CHANGED`) on a `pyee` bus, so the dirty-push
  consumer can be added without touching the DSL again. Two delivery caveats recorded there are
  binding on this section: emission is still **gated by `_anything_changed`** outside an active
  session (the very boolean this bullet replaces), and `SLOT_CHANGED` carries an identity guard.
  Both must be lifted when the graph becomes the subscriber, or the dirty push will miss writes.
- **Memoized topological evaluation**: dirty nodes recompute once, in order; clean subtrees never
  run (Excel-recalc semantics).
- **Tolerance backdating** (salsa's early cutoff with float semantics): recomputed value within
  `(atol, rtol)` of cache ⇒ dirtiness does not propagate.
- **Two explicit cache modes** — the critical revision: `NORMAL` (backdating on) vs `PROBE`
  (backdating OFF). Derivative evaluation, `check_partials`, and tracing run in PROBE — otherwise
  an FD step of 1e-6 whose downstream effect is 1e-9 gets backdated and the Jacobian is
  silently biased to zero.
- **Volatile nodes**: `stochastic=True` properties are never memoized and poison memoization
  downstream; row-save dedup is restated over the tabulated node set.
- **Warm starts**: implicit blocks are atomic nodes — previous solution is the initial guess,
  LU factorization reused while valid (Shamanskii).
- **Selection semantics**: pure var/objective selection = relevance pruning
  (`nx.ancestors(selected objectives) ∩ nx.descendants(selected vars)`), **no recompile**.
  But combos that add/remove residual rows, `add_vars` fabrication, activate/deactivate, and
  collection cardinality are **structural** → the Plan cache is keyed by a **structure key**
  (LRU of compiled plans), so structural selection is a cheap cached recompile, never a stale
  plan. Pre-run check: every selected constraint reachable from some selected var.

## 10. Transients (M9)

Phase 1: **integrator-as-component** — `solve_nonlinear` runs `solve_ivp`; the BLT plan compiles
once and is reused every step; for LTI blocks the one-step map Jacobian is exact. `dxdt=0`
becomes Newton on rate residuals with the A-matrix as Jacobian seed — replacing today's
fabricated `1+r²` objective and the constant-`1` "smallness" dummy (`problem_context.py:1330`).
Phase 2 (when per-timestep constraints are needed): time-expanded group — each step a child
system, block-bidiagonal coupled group; no architectural change required.

## 11. Inspector (M10) — budgeted WITH the compiler

OpenMDAO's lesson: the compiled model is unreadable from the script; the N2 diagram exists
because of that, and it shipped ~16 CLI subcommands to make the graph legible. For 1.0:

- `print(problem.plan)` — typed steps + ASCII BLT matrix (à la OpenModelica `bltmatrixdump`).
- `problem.explain_graph()` — per-edge provenance + static/traced disagreements.
- DM diagnosis inside every structural error; **every error attributed to the declaration site**
  (attrs field / `Signal.define` call) — possible because the graph derives from declarations.
- N2-equivalent connection matrix + solver tree (static HTML/SVG; no server).

## 12. Module layout (`engforge/core/`)

```
engforge/core/
  graph.py       # extraction (L0/L1/L2), alias elim, SCC/BLT, DM, structure key
  tracer.py      # value tracer: deps, guards, affine forms, unit inference
  plan.py        # immutable evaluation plan: typed steps, printer, cache
  vectors.py     # flat u/r/du/dr + views; scaled/unscaled contexts; shadow map
  system.py      # the five-method System ABC; Component/Group binding
  nlsolvers.py   # RunOnce, NLBlockGS(+Aitken), NLBlockJacobi, Newton(+LS)
  lnsolvers.py   # DirectLU (default), LinearBlockGS/Jacobi, ScipyKrylov (later)
  derivatives.py # UDE driver: seeds, fwd/rev, totals; coloring
  partials.py    # affine / FD / gated-CS / jax tiers; check_partials
  scaling.py     # ref/ref0/res_ref + (factor, offset) unit edges
  incremental.py # dirty propagation, NORMAL/PROBE modes, backdating, volatile
  diagnose.py    # Dulmage–Mendelsohn, explain_graph, plan dump, N2 export
  driver.py      # optimizer layer over design vars (SLSQP/trust-constr; IPOPT optional)
```

The existing DSL modules (`configuration.py`, `attr_*.py`, `properties.py`) become the
**declaration front-end** emitting a `ProblemSpec` into `graph.py`. Users never see `u`, `r`,
or `∂R/∂u` unless they ask.

### Module disposition (every current module)

| module | 1.0 disposition |
|---|---|
| configuration, attributes, engforge_attributes, attr_slots/signals/solver/dynamics, properties, components, system, component_collections, typing, logging, patterns, common, env_var, locations | **core** (DSL front-end; collections become replicated subsystems §15.3) |
| system_reference (Ref) | **core** — compiled accessor over `u`; public `Ref` export kept |
| solveable, solver, solver_utils, problem_context | **superseded** by core/ — selection algebra (`combo_filter`, `filter_vals`) ports verbatim as a graph query; ProblemExec exit/checkpoint protocol vocabulary kept, re-based on vector+shadow checkpoints |
| tabulation, dataframe | **core**, re-based on u-slices ∪ shadow (declared dtypes preserved) |
| attr_plotting, analysis, reporting | **core** (product feature); Plot/Trace get graph-node handles → compile-time validation |
| datastores/* | **core** (product feature); fix `datastores/reporting.py:3` broken import; delete pip-install-on-import in `datastores/__init__.py` |
| dynamics | **core** — A,B,C,D become exact Jacobian blocks; integrator-as-component |
| eng/costs | → **engforge-econ** |
| eng/fluid_material, thermodynamics, pipes | → **engforge-thermo** |
| eng/structure, structure_beams, geometry, solid_materials | → **engforge-structures** |
| eng/prediction | deferred (`engforge-surrogate` later; zero engforge imports) |
| _testing_components | split: dependency-free `engforge/testing/fixtures.py` + domain fixtures move out (§15.7) |

### Dependencies

Core: `attrs, numpy (≥2 ok after typing.py fixes), scipy, pandas, matplotlib, seaborn, pint,
networkx (now load-bearing), pyee (graph events)`.
**`pyee` is no longer forward-looking** — it is a declared dependency *today*
(`pyproject.toml:41`, `requirements.txt:64`) and the change-event bus ships now
(`engforge/events.py`, Part V). What remains future is the graph that *consumes* the events.
Dropped: `deepdiff` (hash methods reimplemented on hashlib — public API kept), `randomname`
(→ deterministic `f"{cls}-{n}"`), `networkx-query`, `methodtools`, `arrow`, `tabulate`,
`jupyter`, `black` (dev extra), slack-webhook `requests`.
Two one-line unpin blockers in `typing.py`: `numpy.string_` (line 13) and
`pandas.set_option("use_inf_as_na", True)` (line 8).

## 13. The engforge-&lt;domain&gt; repo family

One repo per current segmentation, each a **sibling folder of engforge** at
`…/Ottermatics Dropbox/Ottermatics/modules/`:

| repo | contents | heavy deps | tests |
|---|---|---|---|
| `engforge-econ` | costs.py (CostModel, Economics, cost_property) | — | test_costs.py (21) + Econ/Fan/Motor fixtures |
| `engforge-thermo` | fluid_material, thermodynamics, pipes | CoolProp, fluids | test_pipes.py + thermo fixtures |
| `engforge-structures` | structure, structure_beams, geometry, solid_materials | PyniteFEA, sectionproperties, shapely, sklearn | _pre_test_structures.py + structure tests |

Direction of dependency is absolute: toolboxes depend on engforge, never the reverse.

### Shared repo template (identical across all three)

```
engforge-<domain>/
  pyproject.toml          # depends on engforge>=<core release>; own numpy/dep policy
  README.md
  engforge_<domain>/      # package (import engforge_econ / …)
    __init__.py
    ...modules...
    test/                 # the tests it takes with it
  .github/workflows/
    build.yml             # same shape as engforge's: format + test matrix
    integration.yml       # tests against BOTH released core (pip engforge==X)
                          # AND core@main (pip git+…engforge@main) — catches
                          # core API drift before a core release does
  .pre-commit-config.yaml # same hooks as core
```

- **Same structure, same test/CI conventions in all three** — one template, cloned.
- `integration.yml` is the contract check: a core change that breaks a toolbox surfaces in the
  toolbox's own CI (scheduled + on-dispatch), and core's release workflow can call the three
  integration workflows before publishing (`workflow_call`).
- Per-repo dependency policy: **engforge-structures moves to numpy ≥2 and current
  sectionproperties immediately** (this dissolves the `[all]`-on-py3.12 blocker permanently).
- A meta-extra (`engforge[toolboxes]`) restores one-line install parity with today's
  `engforge[all]`.
- Dropbox caveat applies to all new repos: verify staging after `git add` before committing;
  never create a branch whose upstream tracks `origin/main`.

### Extraction prerequisite (shippable on 0.x — the toolbox track can run in parallel)

The entire core→eng coupling is three isinstance hooks: `attr_slots.py:86-90` (CostModel slot
policy), `solveable.py:154/174-176` (Economics/CostModel update injection),
`component_collections.py:29-31` (CostModel wide-mode requirement). Replace with:
(a) a **slot-policy hook** on the declaration object, (b) a **registrable update-phase node**.
Plus the dependency-free core fixtures (§15.7). Ship as core 0.4; the three repos can then be
scaffolded against 0.4 while the 1.0 engine is built.

## 14. Compatibility

**Kept as spec** (tests ported): selection algebra and its exact selection-set assertions;
dataframe column naming (`comp.aux`, wide/narrow collection shapes); save-dedup rules
(restated over tabulated nodes); SS↔transient consistency (`test_damping`, 0.01); ω accuracy
(`test_sim`); rate residuals ≈ 0 (`test_dynamics_spaces`); the full ProblemExec exit-protocol
matrix (10 tests); deep-scoping component-solver resolution; performance budgets
(1 s SS / 10 s transient — 1.0 targets 10–100× under).

**Deliberate changes** (tests rewritten, migration-noted):

| change | rationale |
|---|---|
| keep-solution default (revert on request); sweeps still evaluate each point from the per-run checkpoint | revert-by-default is already the minority usage in the suite; sweep independence preserved (§15.6) |
| signal ordering declaration → topological | correctness; `pre`/`post` keywords kept as sugar |
| `test_problem` SLSQP constraint-count assertions → plan-summary assertions | internal SLSQP list shape no longer exists |
| `run()` returns a `Problem` (strict Mapping over reserved keys; `["output"][0]` works; dataframe via `.dataframe`) — **LANDED as `engforge.results.Result`** (Part V): same contract, different class name. 1.0's `Problem` supersedes it. Partly outstanding: §15.5's `Problem(ran=False, reason=…)` for the non-run paths — the skipped-run path now returns the (empty) `Result` rather than `None` so callers need no type check (`solveable.py:427-436`, verified), but it carries **no `ran`/`reason` field**, and the already-solved path still raises a bare `Exception` (`:437-438`). | §15.5 — a column named `output` exists in real models |
| context `assertIs` → same-root/distinct-scope; `last_context` always = session root | concurrent sibling scopes require it; keeps `system.dataframe`/`converged`/`run_id` stable |

## 15. Risk register (from the adversarial review — 3 independent critics, 15 confirmed findings)

1. **Affine rows are exact only per guard region.** Compile each recorded guard into a cheap
   float predicate over `u`-slices, evaluated each iteration; a flip invalidates the affine row
   and re-traces. At switching surfaces (Coulomb friction), fall back to FD. Never advertise
   affine rows as globally exact.
2. **Flat float `u` can't carry revert/tabulation alone.** Two-tier checkpoint: numeric vector +
   copy-on-write shadow map for str/bool/object fields and slot bindings. DataFrames assemble
   from u-slices ∪ shadow with declared dtypes.
3. **Collections are runtime-cardinality.** ComponentDict/Iter = replicated subsystems; item
   subgraph compiled once, instantiated N×; cardinality in the structure key; narrow mode =
   iteration frames rebinding the accessor base; mutation emits an invalidation event.
4. **DM strictness would break existing degenerate-but-passing models** (placeholder `return 0`
   constraints exist in the suite). Trivial zero-incidence rows dropped with a warning naming the
   declaration site; `strict_structure=False` in 1.0, fatal in 1.1.
5. **`Problem.__getitem__` must not be column-like** — `output` is a real variable name in
   real models; reserved-key Mapping only; `run()`'s None/raise paths become
   `Problem(ran=False, reason=…)`; `Analysis.run()` returns the same type.
6. **Sweep-state semantics**: per-point independence is sacred; keep-solution applies to the
   final state of a run, never between sweep points.
7. **Fixture split**: core behavioral tests currently pin cost-model objective names
   (`cost_slv`) — a dependency-free `SliderCrankCore` (mass-proxy objective) replaces it; the
   affected assertions are restated; Econ/material fixtures move to engforge-econ.
8. **Callable/state-dependent bounds** (`lambda s,p: s.Ro`): a `bound` node kind with its own
   incidence, re-evaluated per outer iteration; `ref/ref0` from numeric bounds only; probe box =
   numeric bounds ∩ default·[1−δ, 1+δ]; callables are opaque → require `depends=`.
9. **`update()` write sets are unknowable statically**: harvested by a write-recording probe or
   declared `@update(writes=[…])`; undeclared update = conservative barrier node, loudly
   reported in `explain_graph()`.
10. **Probe on a fresh instance from `input_as_dict`, never `deepcopy`** — live systems hold
    figures, weakrefs, emitters, contexts. `verify_graph` runs out-of-band only.
11. **Complex step is a silent-zero trap through `abs()`** — opt-in + hard gate + independent
    FD oracle (§7). The default derivative method is colored FD, not CS.
12. **Backdating must be mode-split** (NORMAL vs PROBE) or FD Jacobians are biased to zero (§9).
13. **Selection can be structural** — structure-key LRU plan cache, not a singleton plan (§9).
14. **Runtime declaration mutation** (`sig.signal.mode = "post"` on a class object) bypasses any
    event channel — freeze declarations after bind; mutations must go through an invalidating API.
15. **Module disposition must be total** — nine previously-unlisted modules are core; public
    exports `EnvVariable`, `Ref`, and the 7 deepdiff-hash methods survive (reimplemented).

## 16. Phasing & gates

| phase | deliverable | gate |
|---|---|---|
| **Doc** (this) | `docs/ENGFORGE_1.0.md` | review/approval |
| **0 — spike** | standalone prototype of extraction + compile + BLT + incremental + Newton core (~1k lines), run on hinge-fit + SpringMass | identical solutions to 0.3.x within tolerance; plan printouts show expected structure (hinge: pure feed-forward + driver; SpringMass: one 2×2 SCC); timing documented |
| **1 — core engine** | `engforge/core/` per §12 | unit tests on graph/solvers/derivatives; `check_partials` green with independent oracle |
| **2 — DSL binding** | forge → ProblemSpec; selection algebra, ProblemExec protocol, tabulation ported | behavioral suite green (kept-as-spec set); accuracy tests; perf budgets beaten 10× |
| **3 — product features** | plotting/reporting/analysis on graph handles; datastores fixes; pint layer; inspector | plot validation at compile; dimensional golden tests; explain_graph/plan dump shipped |
| **T (parallel) — toolboxes** | T1: cut 3 hooks + fixtures → core 0.4. T2: scaffold the three sibling repos from the shared template (code+tests+CI+integration workflows). T3: meta-extra + `engforge.eng.*` deprecation shims | each repo installs + tests green against core 0.4; integration workflows green against core@main |
| **4 — convergence** | re-base toolboxes on core 1.0; remove shims | `pip install engforge` has no eng/db deps; all four repos green |

## 17. References

- Hwang & Martins, *A computational architecture for coupling heterogeneous numerical models and
  computing coupled derivatives*, ACM TOMS 44(4), 2018 — the MAUD paper.
  https://websites.umich.edu/~mdolaboratory/pdf/Hwang2018a.pdf
- Gray, Hwang, Martins, Moore, Naylor, *OpenMDAO: an open-source framework for multidisciplinary
  design, analysis, and optimization*, SMO 59, 2019. https://openmdao.org/pubs/openmdao_overview_2019.pdf
- Martins & Hwang, *Review and Unification of Methods for Computing Derivatives*, AIAA J 2013.
- Martins & Ning, *Engineering Design Optimization*, CUP 2022, chs. 6 & 13. https://mdobook.github.io
- OpenMDAO theory manual: total derivatives, scaling, setup stack, solver API.
  https://openmdao.org/newdocs/versions/latest/theory_manual/
- Ma et al., *ModelingToolkit: A Composable Graph Transformation System*, arXiv:2103.05244.
- Parker, Nicholson, Siirola, Biegler, *Applications of the Dulmage–Mendelsohn decomposition for
  debugging nonlinear optimization problems*, Comput. Chem. Eng. 178, 2023.
- Pyomo `incidence_analysis` (matching/triangularize/DM reference implementation).
- Gebremedhin, Manne, Pothen, *What Color Is Your Jacobian?*, SIAM Review 47(4), 2005.
- Hill & Schäfer, *Sparser, Better, Faster, Stronger* (SparseConnectivityTracer), arXiv:2501.17737.
- salsa (red-green incremental), Adapton, Mokhov et al. *Build Systems à la Carte*, ICFP 2018.
- Macke et al., *Fine-Grained Lineage for Safer Notebook Interactions* (nbsafety), PVLDB 2021.
- pint performance notes: https://pint.readthedocs.io/en/stable/advanced/performance.html
- Cellier & Kofman, *Continuous System Simulation*, Springer 2006 (tearing, causalization).
- Baharev, Schichl, Neumaier, *Tearing systems of nonlinear equations I/II*.

---

# PART X — Worst-case stress tests & the core purpose (critical analysis)

*What is this framework actually FOR, and do the hardest problems validate or break the redesign?
Two worst cases, traced honestly through the architecture — one nested (MPC on a transient with its
own optimal control), one opaque (a component wrapping a CFD/FEA solve, e.g. aeroelasticity). The
point is not to cheerlead the design; it is to find where "everything is a Component, Y=F(X)" strains
and to name the few primitives that would carry it across.*

## X.1 The core purpose, sharpened

engforge's value is **orchestration of coupled solves over Y=F(X) nodes** — composition, selection
(combos), state/episode management, and tabulated results around the solve — NOT being a CFD/FEA/ODE
engine itself. The sharpest statement of the 1.0 north star: *a node may be a closed-form property, a
child Problem with its own plan, or an opaque black box — and the framework schedules, couples,
differentiates, and records them uniformly.* Every hard case below is a test of that sentence. The
recurring discipline: **wrap, don't reimplement**, and keep the escape hatches that make wrapping
possible few, explicit, and bounded — so the common pure-Y=F(X) case never pays for them.

## X.2 Worst case A — MPC on a transient with internal optimal control

The shape is a four-deep cycle-alphabet composition: **MARCH** the real plant in time; at each
control interval **SOLVE** the controller's optimal-control problem; whose every objective evaluation
**MARCH**es an internal prediction model over a horizon; which may itself **SOLVE** per step. So
`MARCH( SOLVE( MARCH( … ) ) )`.

**What the redesign already fits.** The per-root Problem tree is the natural spine: the plant is a
root; the MPC is a child Problem with its *own* objective (tracking cost) and its *own* unknowns
(the control trajectory); the horizon prediction is an isolated **episode** (`SystemState`
capture→evolve→apply) that must not mutate the plant. The ratified routing principle — *a child
branches from the root's state at entry, overlays only its own inputs, derives its own scope* — is
exactly the MPC-from-current-plant-state semantics. The prediction model is usually a *different*
(reduced/linear) system, which `root=True` + a distinct system already supports (two trees,
Part V.N). Combos re-scope the controller vs the plant with no new concept.

**Where it strains — the real gaps.**
- **Horizon as a decision variable.** The input matrix and `Xref` are *point*-based; MPC optimizes a
  control *trajectory* (N steps). This is the time-expanded group of Part IX §10 — not yet built.
  Without it the horizon has to be faked as N replicated scalar vars, which the naming authority can
  express but the solver has no trajectory-aware structure for. **Addition needed: trajectory /
  time-expanded decision vectors as a first-class axis.**
- **Clock isolation is a hard prerequisite, and currently a no-op.** `SystemState.apply(restore_time=
  True)` only restores `sesh._time`; the system-clock write is a placeholder pending Stage 4's
  writable clock. MPC *requires* the inner horizon clock to advance independently and the plant clock
  to restore exactly — so **Stage 4 (ProblemState owns the clock) is a blocking dependency**, not a
  nicety.
- **Cost.** Each controller objective eval is an entire transient. The per-eval context storm
  (Stage 2b) and recompile-per-timestep (Stage 6/M9) would make this quadratically wasteful; the
  **persistent compiled plan + warm starts** (structure-key cache, M4/M8, seeded from the prior
  solve) is what turns "re-solve every control step" from intractable to routine. Derivatives through
  a trajectory by finite difference are O(N·horizon) sims — adjoint/sensitivity (far-future M6) is
  the only real answer at scale; until then MPC is feasible only for small horizons.

**Verdict:** the architecture *shape* is right (nested Problems + episodes + per-tree state), but MPC
exposes two concrete must-haves it does not yet have: a trajectory decision axis and real clock
isolation. Both are already on the roadmap (§10, Stage 4) — MPC is the forcing function that makes
them non-optional rather than nice-to-have.

## X.3 Worst case B — a black-box component wrapping CFD/FEA (aeroelasticity)

A component whose properties (lift, deflection, stress) are produced by an expensive internal/external
field solve, not a closed-form expression; aeroelasticity couples an aero black box and a structural
black box in a fixed point (deflection ↔ load).

**What the redesign already fits.** This *is* Stage 6's "subsystem as an opaque Y=F(X) node": the
component solves internally and exposes only aggregate properties; the parent sees a black box. The
aero↔structure fixed point is a cycle — and the owned graph's SCC detection (Part VII / Stage 3a,
*landed*) is precisely the tool that finds it and flags whether a solver covers it. The graph's
"unknown → depends on everything" fallback already makes an unscannable body *sound* (over-covered,
never under-covered).

**Where it strains — the real gaps, and the single biggest removal.**
- **Dependency declaration for opaque bodies.** A static bytecode scan cannot see inside an external
  CFD call, so the node collapses to "depends on everything" — sound but sparsity-destroying, which
  defeats the whole graph. **Addition needed:** `@system_property(depends=[...], opaque=True)` so a
  black box declares its true inputs (`lift` depends on `{aoa, mach, mesh}`), restoring a sparse,
  correct graph. `@system_property` has no such option today.
- **State that lives outside engforge.** A CFD solver owns a mesh, a solution field, convergence
  history — state not in engforge fields. engforge's revert/episode model assumes it can snapshot and
  restore *all* state through refs; an external black box breaks that assumption outright. **Addition
  needed:** opaque-state checkpoint hooks (`capture_state()` / `apply_state()` the component can
  override) so `SystemState`/`ProblemState` delegate the part they cannot see — or a contract that
  the wrapper is pure-per-eval (re-init from inputs). Without this, episodes and revert silently
  corrupt external solver state.
- **Per-ref dirty invalidation is a prerequisite, not an optimization.** A CFD eval costs minutes;
  the class-coarse `anything_changed` re-runs it on *any* change anywhere in the tree — catastrophic.
  The black box is the case that makes **Stage 4's per-ref dirty-set replacing the tree-walk flag the
  most important *removal* in the plan**: the CFD must re-run only when its own declared inputs are
  dirty.
- **Field outputs.** CFD/FEA produce arrays/meshes; `@system_property` and tabulation assume scalar
  (int/float/str) outputs. Carrying vector/array-valued refs is a genuine gap (the "continuum/cells"
  direction deliberately deferred). Aggregate scalars (total lift, max stress) fit today; full field
  coupling does not.
- **Derivatives through the box.** Shape-optimizing against a CFD objective needs a Jacobian the
  framework cannot autodiff. **Addition needed:** a hook for a node to *provide* its derivative
  (adjoint from the solver, analytic from a reduced model), generalizing M6's `check_partials`; else
  accept finite-difference cost.

**Verdict:** the graph + opaque-node model accommodates the black box *if and only if* four bounded
escape hatches exist — declare-deps, declare-state-capture, provide-derivatives, and (for fields)
vector refs — and the dirty-set removal lands. Reimplementing CFD/FEA as engforge continuum cells is
the wrong path (rejected earlier, correctly); the win is orchestration *around* the solver.

## X.4 Consolidated: additions / removals / reorganizations these demand

**Add (as few, explicit, bounded primitives — escape hatches, not a new ontology):**
1. `@system_property(depends=[...], opaque=True)` — declared dependencies for unscannable bodies
   (both cases). The highest-leverage single addition.
2. Opaque-state checkpoint hooks (`capture_state`/`apply_state`) — delegate the state the framework
   cannot snapshot (black box).
3. Component-provided derivatives — a node supplies its Jacobian (CFD adjoint, reduced-model analytic)
   instead of FD (both cases, at scale).
4. Trajectory / time-expanded decision variables — optimize over a control horizon, not a point (MPC).
5. Persistent compiled plan + warm starts (structure-key cache, M4/M8) — reuse across per-step /
   per-eval re-solves (both cases; the difference between intractable and routine).
6. Array/vector-valued refs — field outputs (black box; the deferred continuum direction).

**Remove (these actively block the hard cases):**
1. The tree-walk `anything_changed` → per-ref dirty-set (Stage 4). *The* blocker for expensive black
   boxes.
2. The per-eval context storm → Ref-direct hot path (Stage 2b). Death when an eval is a CFD run or a
   horizon sim.
3. Recompile-per-timestep in transients → persistent plan (Stage 6/M9). MPC cannot afford it.
4. The scalar-only tabulation assumption → vector refs. Blocks field outputs.

**Reorganize:**
1. **A node may itself be a Problem.** Formalize "opaque node owns a child Problem with its own
   compiled plan" — the MPC controller and the CFD wrapper are the same pattern (a Problem inside a
   node), generalizing Stage 6 beyond mere self-solve.
2. **The graph is the coupling spine, not a report.** Promote 3a (report-only, landed) to the
   execution substrate (3b/M4/M5): SCC → block solver for the aero↔structure fixed point; layers →
   evaluation order.

## X.5 Honest caveats — the purity tension

Every escape hatch above is a crack in the "pure Y=F(X) refs" ideal: external state is not a field,
field outputs are not scalars, an opaque F is not scannable, a trajectory is not a point. The
redesign *can* carry these, but the discipline is existential: keep the hatches **narrow and
declarative** (declare your deps, declare how to checkpoint, provide your Jacobian) so the 95% pure
case never pays, and the black box is a bounded, explicit exception — not a license for every
component to go opaque. The failure mode to avoid is the hatches metastasizing until "Y=F(X)" is a
fiction and the framework is an untyped callback soup. The test for any future addition here: *does it
stay an opt-in boundary primitive, or does it leak complexity into the common case?* If the latter,
it does not belong. These two worst cases are worth building for precisely because, done with that
discipline, a framework that can *schedule and differentiate coupled black-box and self-solving nodes
uniformly* is something essentially no open Python tool offers — that is the high-impact prize, and
also the reason to be ruthless about scope.
