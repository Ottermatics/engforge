# engforge 1.0 — next steps (2026-10-18)

> **Transient handoff doc. Delete this file once the four remaining stages below are done.**
> It records where the 1.0 work paused so it can resume cleanly. The authoritative design lives in
> `docs/ENGFORGE_1.0.md` (Part V.N is the running "what landed / what remains" ledger; **Part X is
> the critical analysis that frames everything below**).

## Why — the core purpose, and the worst cases that steer the design

Read `docs/ENGFORGE_1.0.md` **Part X** first. The framework's purpose is **orchestration of coupled
solves over Y=F(X) nodes** (composition, combos re-scoping, state/episodes, results) — not being a
CFD/ODE engine. Two worst-case stress tests validate-or-break every remaining stage and set the
priority order:

- **MPC on a transient with internal optimal control** (`MARCH(SOLVE(MARCH(...)))`): the per-root
  Problem tree + episodes fit the shape, but it **forces** a trajectory/time-expanded decision axis
  (Part IX §10), real clock isolation (Stage 4's writable clock — currently a placeholder), the
  persistent compiled plan + warm starts (Stage 6/M9), and eventually component-provided derivatives.
- **Black-box CFD/FEA components** (aeroelasticity: aero↔structure fixed point): the opaque-node +
  SCC-detection model fits, but it **forces** `@system_property(depends=[...], opaque=True)` (the
  single highest-leverage addition — declared deps for unscannable bodies), opaque-state checkpoint
  hooks (`capture_state`/`apply_state`), per-ref dirty invalidation (Stage 4 — a minutes-long CFD
  must re-run only when ITS inputs change, making the `anything_changed` tree-walk removal the most
  important removal in the plan), and array/vector-valued refs for field outputs.

The discipline (Part X.5): escape hatches stay **narrow and declarative** — declare deps, declare
checkpointing, provide Jacobians — so the pure Y=F(X) case never pays and the hatches don't
metastasize into callback soup. Judge every addition below against that test.

## State at pause — all green

Full suite passing: **166 core + 17 econ + 1 thermo + 7 examples**.

Verify:
```bash
cd <engforge>   && conda run -n pace python -m unittest discover -s engforge/test     -p "test_*.py"
cd <engforge-econo>  && conda run -n pace python -m unittest discover -s engforge_econ  -p "test_*.py"
cd <engforge-thermo> && conda run -n pace python -m unittest discover -s engforge_thermo -p "test_*.py"
cd <engforge> && for f in air_filter spring_mass field_area transient_control \
    design_offdesign balanced_network state_variations; do \
    PYTHONPATH=. conda run -n pace python examples/$f.py >/dev/null && echo "$f OK"; done
```

## Landed (foundation — additive, behavior-preserving)

- **One naming authority.** `SolverInstance.get_alias` returns the declared var string → a dotted
  slot var `declare_var('comp.x')` composes to the canonical dotted ref and actually solves (it used
  to key the leaf, mismatch the writeback, and silently "succeed" at the start value). Leaf
  declarations unchanged. `declare_var(on_missing='warn'|'error'|'skip')` handles empty-slot targets.
- **Run input.** `engforge/run_config.py` `RunConfig(method, tol, options)` (serializable, reaches
  scipy, empty default = no change). Structured `run(inputs=, scope=, config=, sequence=, **shorthand)`
  matching the JSON `{inputs, settings, sequence}` shape; flat shorthand still works.
- **Solver audit cleanup.** Removed `_solver_override`, `SOLVER_OPTIONS`, dead `Yref`/
  `sys_solver_objectives`, unused `norm_x`, ~219 commented root/Jacobian lines; fixed the `con_info`
  double-append (Ycon mislabeling), the dead run-timing branch, the `_prob_levels` leak.
- **Per-root Problem (Stage 2, complete).**
  - 2a: explicit `parent`/`is_root`/`root`; session resolution off the global `class_cache.session`.
  - 2b: multi-root coexistence — LIFO active-root stack, `ProblemExec(..., root=True)` forces an
    independent root (different system allowed), uuid `problems_dict` keys (no more `[True]`
    eviction). Two roots coexist in nested `with` blocks.
  - 2c: per-child re-scoping via nested `root=True` on the same system (its own combo → its own
    `Xref`, outer scope restored on exit).
- **Owned problem graph (Stage 3a).** `engforge/problem_graph.py` `ProblemGraph`: dependency DAG
  keyed by dotted ref name, **update-cycle detection** (SCCs), feed-forward `layers()`,
  `Problem.graph`, `run(cycle_check='warn'|'error')` (default off → report-only). The static
  dependency scan is cached **per class at startup** (`TabulationMixin.cls_property_dependencies`
  in `pre_compile`); per-run assembly does no bytecode work.
- **ProblemState (Stage 4 foundation).** `engforge/state.py` `ProblemState`: per-root start
  checkpoint (a `SystemState`) + event-driven dirty-set of changed `(component, attr)` pairs,
  subscribed for the root's lifetime. Additive — it records changes, does not yet gate caching.

New test files: `test_dotted_vars`, `test_none_slot_var`, `test_run_config`, `test_properties_cache`,
`test_graph_cycles`, `test_problem_roots`, `test_problem_state`, plus `test_cli_inputs`,
`test_events`, `test_state_results`.

## Remaining — behavior-changing stages (land each separately, measured against the full suite)

Source detail: `docs/ENGFORGE_1.0.md` Part V.N "What remains" + Part X (the worst-case forcing
functions), plus `sandbox/NOTES_stage0_1.md` and `sandbox/NOTES_solver_audit.md` (git-ignored
working notes).

**Priority order per Part X:** Stage 4-completion first (per-ref dirty invalidation + the writable
clock — a hard prerequisite for BOTH worst cases), then Stage 6 (opaque/self-solving nodes +
persistent transient plan — the node-is-a-Problem pattern both cases share), then 3b and 5. The
Part X escape hatches (`depends=`/`opaque=` on `@system_property`, `capture_state`/`apply_state`
hooks, component-provided derivatives, trajectory variables, vector refs) slot into Stages 4/6 and
Part IX M6/§10 — add them as the worst-case examples demand, never speculatively.

### Stage 3b — layer-ordered execution (MED-HIGH)
Drive `update_system`/`post_update_system`/signal order from the graph's `layers()` instead of
declaration order; uncovered cycles error by default. **Risk:** changes evaluation order — the
sandbox probes found order-dependent `update()` models, so gate on example-by-example bit-equivalence
before flipping the default. Also fold in the structure-key plan cache (build once per structure, not
per `__enter__`; `full_update` default → False) and stochastic-node cone poisoning.

### Stage 4 (completion) — retire `_anything_changed_` via the dirty-set (MED, cross-repo)
Route `System.anything_changed` and the `solver_cached`/`cached_system_property` invalidation through
the per-root `ProblemState` dirty-set (today `anything_changed` is a full tree walk per read).
**Risk:** load-bearing for the whole cache layer. **Cross-repo lockstep required** — engforge-econ
(`costs.py` writes `self._anything_changed=True`), engforge-structures (`structure.py`),
engforge-thermo (`fluid_material.py` reads it) use the public API; migrate them in the same pass or
keep a shim.

### Stage 5 — hard-drop `data_dict` / `dataframe` object surface (MED)
User decision: hard-drop (no long deprecation). Replace the live-state-inspector role with
`Problem.values(keys=...)` (thin wrapper over `get_ref_values`). Migrate the pinned asserts to the
`Result` path — **econ's ~19 keyed asserts** in `engforge_econ/test/test_costs.py` (keys like
`econ.summary.total_cost` are unchanged, dotted everywhere) + ~11 core `data_dict` asserts + ~23
`system.dataframe` test lines. Econ's post-run lifecycle/LCOE values must stay identical (its
`internal_references` injection moves to graph-registered refs feeding the recorded row). Untangle the
`format_columns` inversion (`results.py` / `problem_context.py` call back into the system — move to a
pure `System.column_metadata()`). Plot/Trace + CSVReporter consume a `Result`.

### Stage 6 — subsystem self-solve + M9 transients (HIGH)
A subsystem marked self-solving becomes an opaque `Y=F(X)` node solved at its layer position — needs
a **new "solve-locally" marker** (the old dead `_solver_override` was removed) + a true subtree prune
in `go_through_configurations` (fixes the grandchild leak) so the parent's solver doesn't reach in. A
solve inside `update()` auto-attaches as a child of the current root (contextvar + `root=True`, now in
place). Transients: make `tr_slvr`/`ss_slvr` persistent child problems created once per `integrate()`
and re-entered per step (reuse the compiled structure) instead of recompiling every timestep.

## Scaling / parallel execution (new track — see `docs/SCALING.md`, 2026-10-19)

Measured groundwork is done (full reports: `sandbox/NOTES_scaling_groundwork.md`): cloudpickle
transports systems/SystemState/Results bit-identically (stdlib pickle never will — uuid-named dynamic
attr classes); ray 2.59 + the `distributed` extra already exist; local parallel pays only above
~300 ms/eval; the serial sweep's X0 chaining means the distribution contract is "every task starts
from a pinned SystemState". Roadmap P0–P3 in the doc: merge contract → `backend="mp"` →
`backend="ray"` → parametric-hash cache + CFD/MPC actors (actors dissolve Part X's opaque-state
problem). Business model (§7.1): **local free forever; flat monthly subscription with rate limits**
(CPU-light workloads = bounded COGS = software margins; no metered-bill anxiety; rides under the
procurement cliff) + **turnkey vertical solutions on validated toolboxes** (structural / fluids /
power+LCOE — the sibling repos are the seeds; the moat is V&V evidence + maintenance, not code).
Realistic ceiling ~$0.5–3M ARR; sequence framework adoption BEFORE cloud, one vertical first.

## Smaller deferred items (audit-backed, not blocking)

- **Combo var/bound fork-merge.** A var's combos (`default,{var}`) vs its `add_var_constraint` bound
  (`default,lim`) can diverge → a var selected under a custom combo drops its bound (unbounded). The
  naive additive fix regressed 4 slider-crank tests (they scope bounds deliberately). Real fix: merge
  the two combo/active forks (`combo_filter` vs the inline ladder in `sys_solver_constraints`) into
  one helper.
- `add_vars` fabricates a throwaway `Solver` subclass per `solver_vars` call — cache by name.
- `add_obj` / `obj` run-kwargs are dead and their solver docstrings misrepresent the contract.
- `engforge/datastores/reporting.py` is unimportable (broken `from engforge.datastores.datastores...`
  import) — dead module.
- Integrator dotted-var naming (`trv_inst`/`filter_vars`) — leaf integrators work; a dotted
  integrator var needs the same naming-authority fix + a test model.

## Standing rules
- No git commits/stages without explicit per-action permission (Dropbox index can drift — re-verify
  after staging).
- `sandbox/` stays git-ignored; it holds working notes/prototypes, not shipped code.
