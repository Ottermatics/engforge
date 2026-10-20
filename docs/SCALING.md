# engforge — Scaling & the Size Problem (2026-10-19)

*Companion to `ENGFORGE_1.0.md`. Parallel execution, Ray leverage, results-at-scale, and the honest
commercial potential of a cloud offering. Everything in §2–§5 is **measured on this repo** (probe
scripts ran against `engforge_1.0`, env `pace`: ray 2.59.0 + cloudpickle 3.1.2 installed; the
`distributed` extra already exists in `pyproject.toml:84` with a stale `~=2.35` pin). §7 is sourced
market research (2026-10-19). Treat unverified revenue estimates as order-of-magnitude.*

---

## 1. What actually needs scale (the taxonomy)

Not everything in engforge is worth parallelizing. In order of value:

| tier | workload | parallel shape | state coupling |
|---|---|---|---|
| T1 | **input-matrix sweeps / multipoint / multi-site (MAP)** | embarrassingly parallel per point | none — each point is (system, point, scope) |
| T2 | **stochastic ensembles** (CRN seeds), multistart | embarrassingly parallel per seed/start | none, seed is an input |
| T3 | **FD Jacobian / constraint evals inside one solve** | parallel per perturbation | needs the Stage-2b Ref-direct hot path first |
| T4 | **heavy nodes**: CFD/FEA black boxes, MPC subproblems | long-lived **stateful** workers | mesh/solution/compiled-plan state must persist between calls |
| T5 | **results at scale**: 10⁵–10⁷ rows from T1–T3 | streaming merge, sparse storage | provenance + index discipline |

T1/T2 are the 90% win and need no solver changes. T3 waits on Stage 2b. T4 is Part X's worst cases —
and distribution *dissolves* rather than complicates them (§4.3).

## 2. Measured feasibility today

| question | measured answer |
|---|---|
| does a System pickle? | **stdlib: NO** — every Solver/Signal declaration is a dynamically-built class with a uuid-randomized name (`attributes.py:186-197`); pickle-by-reference can't resolve it. **cloudpickle: YES** (by-value classes): CubeSystem 43 KB, post-run 85 KB, SystemState 22 KB, Result 89 KB — all round-trip and the clone solves correctly |
| does a task round-trip? | **bit-identical**: (system bytes, point, scope) → spawned worker → rows; 44/44 value keys and `Xans` exactly equal vs in-process. Worker *results* (numeric_rows + scalars) are stdlib-picklable |
| overhead | steady-state (de)serialization ≈ 3 ms/task (**<0.5%** of a 300 ms solve). The real tax is **worker warmup 1.8–4.4 s/process** (engforge import on first loads) — pools must be reused, never per-sweep |
| local speedup | **light evals: parallel LOSES** (serial batch ≈ 73 ms/pt; an independent per-point run ≈ 292 ms — 4.5× fixed overhead from per-run recache/context). **Heavy evals (≥ ~0.5 s/pt): 2.7–3.6×** at 4–8 workers. Rule: parallelize when per-point cost ≳ 300 ms, else the serial batch wins |
| equality caveat | the serial sweep **chains X0 point-to-point** (`solver.py` exits `revert=False` between points), so parallel tasks starting from the snapshot legitimately differ (2/32 convergence flips, ≤1.8e-3 drift on the rest). With the default `revert_every=True` the two coincide — **the distribution contract is: every task starts from a pinned SystemState** |

Merge-layer defects found (fix in P0, §6): non-converged runs record a junk second (reverted) row in
the same Case; `Case.index` restarts per case (32/34 collisions on naive merge); `run_id` is None in
rows (no provenance); `case.output` carries `objectify` closures (unpicklable — ship a slimmed
`{success, Xans, Yobj, Ycon, ans.fun/.x/.nit}`). Also: `datastores/gdocs.py:36` hard-imports ray from
an optional module, and `input_hash` is instance-unstable (the random `name` field) while the
parametric `hash('x','y')` **is** stable — use the parametric form for caching.

## 3. Execution model — one seam, no new user concepts

The per-point loop in `solveable.py:_iterate_input_matrix` (~:392-402) is the only place a matrix
point binds to a Case. It becomes an executor strategy:

- `backend="local"` (default) — **the current loop verbatim**; zero behavior change.
- `backend="mp"` — process pool for one workstation; payloads pre-wrapped in cloudpickle (stdlib
  pickler would fail, §2) or loky; pool reused across runs (warmup).
- `backend="ray"` — cluster scale-out (§4).

`backend` / `max_workers` ride `dflt_parse_kw` exactly like `save_mode`/`run_config` — recognized
context options, stripped from the input matrix, settable from `run()`, the JSON `settings` section,
or the CLI. Core gains `engforge/execution.py` (imports no backend); only `engforge/exec_ray.py`
imports ray (optional extra). `cb`/`post_callback` are driver-side closures: rejected (or explicitly
cloudpickled) on non-local backends; `revert_every=False` + parallel warns (chained-X0 semantics
don't distribute).

## 4. Ray leverage

### 4.1 Tasks + object store (T1/T2)
`ray.put(system)` **once** → one copy per node, zero-copy to workers; each task materializes a
per-task copy — **isolation for free**, and a worker's fresh `class_cache` means the task context is
simply a root (the per-root Problem work is what makes this legal). Driver streams completions
(`ray.wait`) into pre-allocated Cases **indexed by point id, not arrival order** — allocated via
`Result.new_case` (NOT `begin_case`, which deliberately collapses row-less cases).

### 4.2 Content-addressed caching (the "never re-run the same point" layer)
Ray's object store is intra-job; the durable win is a cross-run memo: parametric
`system.hash(*input_keys)` (deepdiff, measured instance-stable) + scope hash → Case rows, persisted
(parquet/sqlite). Hooks at the executor seam: check before submit, write after merge. **Stochastic
caveat (CRN):** `stochastic=True` nodes bypass the cache unless the seed is an explicit input — the
probe-F pattern makes the stochastic point cacheable by keying on `(inputs…, seed)`.

### 4.3 Actors — the Part X worst cases, dissolved
A **CFD/FEA black box as a Ray actor**: the mesh/solution/preconditioner lives *inside* the actor
between calls (warm restarts); only plain data (BCs in, forces out) crosses the boundary. This
*dissolves* the opaque-state problem — state that never crosses the boundary never needs
`capture_state`/`apply_state`. Placement groups give the actor its MPI cores; the engforge component
stays a thin picklable proxy whose property calls `actor.evaluate.remote(...)`. An **MPC controller
actor** holds its compiled plan + last trajectory (warm starts) the same way. "A node may itself be a
Problem" (Part X.4) gets a concrete runtime: *a node may be an actor hosting a Problem.*

## 5. The size problem (T5)
Big sweeps → big frames. Already landed: `save_mode="sparse"` (delta rows, 58% fewer stored values on
transients) + dense reconstruction on read. Needed at scale (P0 fixes): a global `(run_id, point_id,
row_index)` key on merged rows; converged/tag filtering before concat; out-of-core append (parquet
per case, concat lazily). The Result/Case ownership model makes all of this a merge-layer concern —
the solver never touches it.

## 6. Staged roadmap

| stage | scope | acceptance |
|---|---|---|
| **P0 merge contract** | pin per-task start-state; slim task output; global row keys (run_id/point_id); tag non-converged rows; fix `gdocs.py` ray import; refresh the `distributed` pin | parallel merge == serial-with-revert_every merge, bit-equal on converged points |
| **P1 `backend="mp"`** | executor seam + process pool + cloudpickle wrapping + pool reuse | heavy-eval sweep ≥2.5× at 4 workers; light-eval auto-falls-back to local with a hint |
| **P2 `backend="ray"`** | exec_ray.py tasks + object store + as-completed streaming | same equality contract on a multi-node cluster; one `ray.put` per sweep |
| **P3 cache + actors** | parametric-hash memo (CRN-aware); CFD/MPC actor pattern + example | cache hit-rate test; actor warm-restart measured faster than cold |

## 7. Cloud profit potential — the honest read

*Sourced 2026-10-19; estimates flagged. Full comp detail in the research brief (sandbox notes).*

**Comps and their lessons.** Rescale: 14 yrs + $260M raised → ~$33M ARR (est.) selling enterprise HPC
orchestration — a grind. SimScale: ~300k free users → ~$20M (est.) over 12 yrs; freemium conversion
in mechanical engineering is brutally slow. **Anyscale** (hosted Ray): monetization stalled until the
AI wave, then absorbed by a GPU cloud (~$1.65B, 2026) — an exit powered by a compute-hungry buyer
class that systems modeling does not have. **Coiled** (hosted Dask): 50 people → 2, now a *profitable
micro-business* on a ~$0.05/CPU-hr BYOC orchestration tax — the realistic hosted-runtime template.
**Modelon** (closest domain comp): 20 years, open standard, $6k/seat/yr SaaS — **~$6M ARR, still
unprofitable**. Valispace ("GitHub for system models"): ~$20M exit into Altium — the collaboration
layer was worth more as a feature than a company. Flow Engineering ($750M mark, 2026): the capital is
flowing to *AI agents over engineering data, sold top-down to funded hardware primes* — a different
game.

**The ops-aversion thesis, corrected.** True: this audience will never self-host a cluster (data
engineers self-host Ray; mechanical engineers won't — that asymmetry is real and is why Coiled's
remaining customers pay the tax). But the same population mostly doesn't buy hosted compute either —
it stays in Excel. Fifteen years of "democratize simulation for the missing middle" (8% of ≤100-person
manufacturers use HPC vs 62% of >10k) moved the needle only modestly: the aversion is to the **whole
overhead category**, not ops specifically. Conclusion: ops-aversion converts into willingness to pay
for **packaged capability and shareable results**, not core-hours. Revealed price band for this
exact user: Mathcad at ~$800–2,400/seat/yr; the procurement cliff (card/PO-threshold buying) sits
near $500–1,000 — Modelon's $6k/seat is the cautionary other side of it.

**Credible wedges, in order:**
1. **Seats for shared results + model registry** ("Posit Connect for system models"): versioned
   models, run history, shareable trade-study dashboards, sign-off — `Result`/`Case`/`SystemState`
   map onto this directly. $50–200/user/mo, under the procurement cliff. Risk: target companies have
   1–3 modelers; Valispace caps the standalone ceiling.
2. **BYOC burst compute** (the Coiled shape): metered orchestration tax on sweeps running in the
   customer's own cloud account; near-zero COGS. Risk: Y=F(X) sweeps are CPU-light — this is margin
   on wedge 1, not a business.
3. **Packaged vertical apps + VPC/support upmarket**: SMEs buy apps, not frameworks (the
   missing-middle research is explicit); ITAR/IP paranoia makes single-tenant the enterprise tier.
   Risk: each vertical is its own GTM fight; Modelon shows the slope.

### 7.1 Business model v2 — flat subscription with rate limits + turnkey vertical toolboxes

Two refinements that improve on the metered model above:

**Flat monthly subscription with rate limits (the Claude/Cursor shape), not metered core-hours.**
Local (`backend="local"|"mp"`) stays free forever — that's the hook and the trust-builder. The paid
tier is a flat subscription whose cloud bursts are governed by **rate limits (throttling), never
surprise bills**. Why this fits better than metering:
- The CPU-lightness of Y=F(X) sweeps — a *weakness* under metering (dollars of revenue per sweep) —
  becomes a *strength* under caps: COGS per subscriber is tiny and bounded, so margins are
  software-like, not compute-resale-thin (the inverse of Rescale's problem).
- Predictable flat pricing kills metered-bill anxiety (a documented cloud-CAE adoption barrier) and
  rides under the $500–1,000 procurement cliff on a card.
- The pricing shape is now culturally normalized: by 2026 every engineer pays $20–200/mo with rate
  limits for AI tools. The mental model is pre-installed.
- Sketch: **Free** (local + open toolboxes) → **Pro ~$49–99/mo** (cloud bursts under caps, hosted
  results) → **Team ~$199–499/mo** (shared model registry, seats, sign-off) → **Enterprise** (VPC/
  ITAR, validated-toolbox support contracts).
- *Honest stress-test:* Claude-style subs work because usage is daily-habitual; engineering sweeps
  are **bursty/episodic** (heavy one week per quarter) → churn/pause-gaming risk. Mitigations: the
  continuous value must live in the *results layer* (registry, shared dashboards, run history — value
  persists between bursts), annual billing, generous rollover. The subscription is really for the
  workspace; compute caps are the enforcement detail.

**Turnkey vertical solutions on toolboxes — the actual moat.** The runtime is self-hostable and the
framework is open; neither is a moat. What is: **validated, maintained, domain-credible toolboxes and
the turnkey analyses built on them** — structural / fluids-thermal / power+techno-economics — which is
exactly what the missing-middle research says SMEs buy ("pre-packaged apps, not frameworks"), and
exactly where the sibling repos already point (engforge-structures, engforge-thermo, engforge-econ are
the seeds; power/LCOE is the natural third leg given the econ toolbox and the ocean-energy work).
The defensible part is not the toolbox *code* — it's the **V&V evidence** (documented verification
against known solutions, code-check modules: AISC/ASME/IEC-flavored) plus maintenance, which
open-source almost never does and engineers require before trusting a number. Scaling analysis is the
multiplier: "your 10,000-point trade study / Monte Carlo / multipoint optimization finishes over
lunch, from a button" is the turnkey promise only the cloud tier delivers.
- *Honest stress-test:* every vertical is its own GTM and credibility fight (Modelon's 20-year
  slope); V&V documentation is genuinely expensive; code-compliance claims carry liability framing
  that needs care. Pick **one** vertical first — the one with existing consulting credibility — and
  productize from paid client work, not speculation.

**Verdict (v2).** The flat-sub + verticals shape raises the realistic ceiling versus pure metering —
bounded-COGS subscriptions at a few hundred to a few thousand seats is a **~$0.5–3M ARR** business
with software margins, plus vertical/enterprise upside — but it is still a niche-calibrated outcome,
not venture scale, unless the vertical-apps engine compounds (or the play shifts to the
AI-agents-over-engineering-data lane). Sequencing is unchanged and non-negotiable: framework adoption
→ free local/mp backend as hook → hosted results + capped bursts as the subscription → ONE validated
vertical as the moat-building act → others follow proof. Building the cloud before the community is
the known-fatal order of operations.

## 8. Decisions needed before P0
1. Bless cloudpickle as the transport (it already works; stdlib never will without renaming the
   dynamic attr classes — a bigger refactor for little gain).
2. `backend` naming + home (proposed: `dflt_parse_kw`, like `save_mode`).
3. Whether the stable parametric hash should become the default `input_hash` (excluding the random
   `name`) — also fixes cache identity.
