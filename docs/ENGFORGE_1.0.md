# engforge 1.0 — The Compiled Problem Graph

*Design document. Status: approved direction, pre-implementation. This is the authoritative
reference for the 1.0 core rebuild and the engforge-&lt;domain&gt; repo family.*

---

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
  the graph → eager dirty push to transitive dependents only.
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
| `run()` returns a `Problem` (strict Mapping over reserved keys; `["output"][0]` works; dataframe via `.dataframe`) | §15.5 — a column named `output` exists in real models |
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
