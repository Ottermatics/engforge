"""A tank on the roof, a pump in the basement, and a spec sheet with two pumps on it.

The line is already installed: 25 m of pipe, 6 m of static lift to the tank inlet.
Supply offers two pumps, and the pipe can still be re-spec'd to DN32, DN40 or DN50.
Six real build options -- not twelve, because a pump's shutoff head and its curve
steepness are one row off one spec sheet. They do not get to be swept independently.

That sentence is the whole subject of this example. engforge 1.0 made the run input
layer, the result layer, the state layer and the change bus explicit, and this file
walks them in the order you'd actually hit them on that job:

  1. the input matrix        -- comma keys vary jointly, so the sweep is a
                                PARTIAL factorial (6 points, not 12)
  2. the Result              -- run() hands back a Result; each Case owns its rows
  3. the state tree          -- one context, SystemState.capture at the root,
                                a branch per variation, exact restore, and a diff
                                that names precisely what moved
  4. save_mode="sparse"      -- delta rows, and what forward-filling them back costs
  5. the event bus           -- watch() the value/slot/collection signals, scoped

The physics is a pump/system head balance, which happens to have a closed form:

    H_pump(Q) = h_shutoff - k_curve Q^2          (quadratic curve fit)
    H_sys(Q)  = lift0 + level + R Q^2            (static lift + Darcy-Weisbach)
    R         = 8 f L / (pi^2 g D^5)             (head loss written in flow terms)
    => Q*     = sqrt((h_shutoff - lift0 - level) / (k_curve + R))

So every number engforge's solver produces below can be checked against algebra,
which is what the assertions at the bottom do.

Run it:  python examples/state_variations.py
"""

import contextlib
import io
import math

import numpy
import pandas

from engforge import Component, Slot, Solver, System, Time, field, forge
from engforge import events
from engforge.component_collections import ComponentIterator
from engforge.inputs import count_points, expand_point_factors, split_group_key
from engforge.logging import change_all_log_levels
from engforge.problem_context import ProblemExec
from engforge.properties import system_property
from engforge.results import Case
from engforge.state import SystemState

G = 9.80665  # m/s^2

# ---------------------------------------------------------------------------
# the hardware
# ---------------------------------------------------------------------------


@forge
class Pipe(Component):
    """The discharge line. `resistance` is Darcy-Weisbach rewritten in terms of
    volumetric flow so the head balance stays a plain quadratic in Q:

        h_f = f (L/D) V^2 / 2g,  V = Q/A,  A = pi D^2 / 4  ->  h_f = R Q^2
    """

    L: float = field(default=25.0)  # m, developed length
    D: float = field(default=0.032)  # m, internal diameter
    f: float = field(default=0.022)  # Darcy friction factor

    @system_property
    def flow_area(self) -> float:
        return math.pi * self.D**2 / 4.0

    @system_property
    def resistance(self) -> float:
        """s^2/m^5 -- the R in h_f = R Q^2"""
        return 8.0 * self.f * self.L / (math.pi**2 * G * self.D**5)


@forge
class Pump(Component):
    """A centrifugal pump as its catalogue curve: H = h_shutoff - k_curve Q^2.

    These two numbers are ONE fact about ONE pump. Sweeping them independently
    invents pumps nobody sells -- which is exactly why the matrix below joins
    them with a comma key.
    """

    h_shutoff: float = field(default=18.0)  # m, head at zero flow
    k_curve: float = field(default=0.6e6)  # m / (m^3/s)^2, curve steepness


@forge
class FillLoop(System):
    """Pump -> pipe -> roof tank. The solver finds the operating flow; the
    transient lets the tank fill, which raises the static head and walks the
    operating point back down the pump curve.
    """

    lift0: float = field(default=6.0)  # m, inlet elevation above the pump
    tank_area: float = field(default=1.2)  # m^2, tank plan area
    level: float = field(default=0.0)  # m, water depth in the tank (state)
    Q: float = field(default=3.0e-3)  # m^3/s, the solver's variable

    pipe: Pipe = Slot.define(Pipe)
    pump: Pump = Slot.define(Pump)

    # the steady-state problem: one unknown, one equation.
    var_Q = Solver.declare_var("Q", combos="flow")
    var_Q.add_var_constraint(0.0, kind="min", combos="flow")
    balance = Solver.constraint_equality("head_error", 0.0, combos="flow")

    # the transient: the tank integrates whatever the pump is delivering.
    fill = Time.integrate("level", "fill_rate")

    @system_property
    def static_head(self) -> float:
        return self.lift0 + self.level

    @system_property
    def pump_head(self) -> float:
        return self.pump.h_shutoff - self.pump.k_curve * self.Q**2

    @system_property
    def friction_head(self) -> float:
        return self.pipe.resistance * self.Q**2

    @system_property
    def head_error(self) -> float:
        """The residual the solver drives to zero, DIVIDED BY shutoff head.

        Normalizing matters: in metres the residual is O(10) while Q is O(1e-3),
        and SLSQP walks off a linesearch on that scaling inside the transient
        (observed: "Positive directional derivative for linesearch"). Dimensionless
        residuals are cheap insurance.
        """
        return (
            self.pump_head - self.static_head - self.friction_head
        ) / self.pump.h_shutoff

    @system_property
    def velocity(self) -> float:
        """m/s in the line -- the sanity check on any pipe sizing"""
        return self.Q / self.pipe.flow_area

    @system_property
    def Q_balance(self) -> float:
        """the closed-form operating point, for checking the solver's homework"""
        drive = self.pump.h_shutoff - self.static_head
        if drive <= 0.0:
            return 0.0  # pump cannot lift it; no flow, no imaginary numbers
        return (drive / (self.pump.k_curve + self.pipe.resistance)) ** 0.5

    @system_property
    def fill_rate(self) -> float:
        """m/s of water depth -- the rate `fill` integrates into `level`"""
        return self.Q / self.tank_area


# ---------------------------------------------------------------------------
# the two pumps on the quote, and the matrix they imply
# ---------------------------------------------------------------------------

PUMP_ROWS = [(18.0, 0.6e6), (28.0, 1.4e6)]  # (h_shutoff, k_curve) per pump
DIAMETERS = [0.032, 0.040, 0.050]  # DN32 / DN40 / DN50

#: ONE comma key => `pump.h_shutoff` and `pump.k_curve` step through PUMP_ROWS
#: together (zipped), and that single factor crosses `pipe.D`.
MATRIX = {
    "pump.h_shutoff,pump.k_curve": PUMP_ROWS,
    "pipe.D": DIAMETERS,
}

SOLVE = dict(combos="flow", slv_vars="*", revert_last=False, revert_every=False)


def build() -> FillLoop:
    """Names are explicit on purpose: engforge auto-generates a random readable
    name per instance, and random names make example output non-reproducible."""
    return FillLoop(name="loop", pipe=Pipe(name="line"), pump=Pump(name="pump"))


@contextlib.contextmanager
def quiet():
    """engforge narrates itself on stdout -- a dot per eval, and scipy's own
    integration summary. Mute it around engine calls so the report below is the
    only thing on screen. Exceptions still propagate."""
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        yield buf


def rule(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# 1. the input matrix: a comma key is a partial factorial
# ---------------------------------------------------------------------------


def part1_matrix():
    rule("1. GROUPED MATRIX INPUTS -- jointly varied, not crossed")

    for key, val in MATRIX.items():
        print(f"  {key:>32} : {val}")

    # The same three keys as INDEPENDENT factors -- the full outer product.
    independent = {
        "pump.h_shutoff": [r[0] for r in PUMP_ROWS],
        "pump.k_curve": [r[1] for r in PUMP_ROWS],
        "pipe.D": DIAMETERS,
    }
    n_full = count_points(independent, [])

    # As declared: the comma key is ONE factor of zipped rows.
    groups = [(split_group_key("pump.h_shutoff,pump.k_curve"), PUMP_ROWS)]
    singles = {"pipe.D": DIAMETERS}
    n_group = count_points(singles, groups)

    print()
    print(f"  full outer product (3 independent factors) : {n_full} points")
    print(f"  as declared (1 group x 1 factor)           : {n_group} points")
    print(
        f"  the {n_full - n_group} points we skipped describe pumps nobody sells "
        "(18 m shutoff on a 1.4e6 curve, etc.)"
    )

    print()
    print("  the points engforge will actually visit:")
    print(f"    {'#':>2}  {'h_shutoff':>10} {'k_curve':>10} {'pipe.D':>8}")
    for i, pt in enumerate(expand_point_factors(singles, groups)):
        print(
            f"    {i:>2}  {pt['pump.h_shutoff']:>10.1f} "
            f"{pt['pump.k_curve']:>10.0f} {pt['pipe.D']:>8.3f}"
        )

    # ...and it does. parse_run_kwargs validates the keys, _iterate_input_matrix
    # expands them, and every point becomes one Case.
    loop = build()
    with quiet():
        result = loop.run(**MATRIX, **SOLVE)
    print()
    print(f"  loop.run(**MATRIX) -> {result!r}")
    return n_full, n_group, result


# ---------------------------------------------------------------------------
# 2. the Result: explicit row ownership
# ---------------------------------------------------------------------------


def part2_result(result):
    rule("2. RESULT / CASE -- each point owns its own rows")

    print(f"  type(result)        : {type(result).__name__}")
    print(f"  result.cases        : {len(result.cases)}")
    print(f"  mapping keys        : {list(result)}")

    print()
    print(
        f"    {'case':>4} {'rows':>4} {'h_shut':>7} {'k_curve':>9} {'D':>6} "
        f"{'Q_solved':>10} {'Q_algebra':>10} {'V':>6} {'resid':>10}"
    )
    rows_per_case = []
    worst_resid = 0.0
    worst_vs_algebra = 0.0
    for i, case in enumerate(result.cases):
        row = case.dense_rows[-1]  # this case's own rows, nobody else's
        rows_per_case.append(len(case.rows))
        resid = abs(row["head_error"])
        rel = abs(row["Q"] - row["Q_balance"]) / row["Q_balance"]
        worst_resid = max(worst_resid, resid)
        worst_vs_algebra = max(worst_vs_algebra, rel)
        print(
            f"    {i:>4} {len(case.rows):>4} {case.inputs['pump.h_shutoff']:>7.1f} "
            f"{case.inputs['pump.k_curve']:>9.0f} {case.inputs['pipe.D']:>6.3f} "
            f"{row['Q'] * 1e3:>9.4f}L {row['Q_balance'] * 1e3:>9.4f}L "
            f"{row['velocity']:>6.2f} {resid:>10.2e}"
        )

    df = result.dataframe  # every case's rows, one frame, columns formatted
    print()
    print(f"  result.dataframe    : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  sum(len(case.rows)) : {sum(rows_per_case)}  (same rows, grouped by point)")

    # The pinned legacy contract: a Result IS a Mapping.
    out0 = result["output"][0]
    print()
    print("  legacy Mapping access still works:")
    print(f"    result['input_sets'][0]      : {result['input_sets'][0]}")
    print(f"    result['output'][0]['success'] : {out0['success']}")
    print(f"    result['output'][0]['Xans']    : {out0['Xans']}")
    print(f"    len(result['input_sets'])      : {len(result['input_sets'])}")

    return rows_per_case, worst_resid, worst_vs_algebra, df


# ---------------------------------------------------------------------------
# 3. the state tree: one context, a root state, a branch per variation
# ---------------------------------------------------------------------------

BRANCHES = {
    "bigger line (DN50)": {"pipe.D": 0.050},
    "the other pump": {"pump.h_shutoff": 28.0, "pump.k_curve": 1.4e6},
    "tank half full": {"level": 4.0},
}


def part3_state_tree():
    rule("3. SYSTEM STATE -- a root, a branch per variation, an exact restore")

    loop = build()
    leaf_diffs = {}
    restored_clean = True
    clock_restored = None

    with ProblemExec(loop, {}, level_name="tree") as px:
        root = SystemState.capture(px)
        print(f"  root = SystemState.capture(px)  ->  {root!r}")
        print(f"  it holds every settable ref, {len(root)} of them:")
        print(f"    {sorted(root.values)}")
        print(f"  baseline operating point: Q = {loop.Q_balance * 1e3:.4f} L/s")

        print()
        print("  branching -- root.updated(point).apply(px), then capture the leaf:")
        for label, point in BRANCHES.items():
            leaf_state = root.updated(point)
            leaf_state.apply(px)
            leaf = SystemState.capture(px)

            # diff() names precisely what moved, as {key: (root, leaf)}. Sorted:
            # diff() walks a set union, so its dict order is not stable across
            # processes.
            raw = root.diff(leaf)
            moved = {k: raw[k] for k in sorted(raw)}
            leaf_diffs[label] = moved
            print(f"    {label:<20} Q = {loop.Q_balance * 1e3:7.4f} L/s")
            for key, (was, now) in moved.items():
                print(f"      {key:<22} {was!r:>12}  ->  {now!r}")

            # back to the root, exactly. The leaf stays in hand as a snapshot.
            root.apply(px)
            if root.diff(SystemState.capture(px)):
                restored_clean = False

        print()
        print(f"  after each restore, root.diff(capture) == {{}}  : {restored_clean}")
        print(f"  back at baseline: Q = {loop.Q_balance * 1e3:.4f} L/s")

        # The episode clock lives IN the state -- that is what makes a transient
        # bracket itself. Field restores leave it alone; restore_time=True
        # rewinds it, which is the EPISODE path (see dynamics.simulate).
        print()
        px.sesh._time = 7.5
        drifted = SystemState.capture(px)
        print(f"  clock pushed to t=7.5 -> root.diff(now) = {root.diff(drifted)}")
        root.apply(px)
        print(f"  root.apply(px)                  -> clock {px.sesh._time} (untouched)")
        root.apply(px, restore_time=True)
        clock_restored = px.sesh._time
        print(f"  root.apply(px, restore_time=True) -> clock {clock_restored}")

    return leaf_diffs, restored_clean, clock_restored


# ---------------------------------------------------------------------------
# 4. save_mode="sparse": store the deltas, forward-fill them back
# ---------------------------------------------------------------------------

SIM = dict(dt=0.2, endtime=0.4, run_solver=True, combos="flow", slv_vars="*")


def cells(cases):
    """how many values are actually sitting in storage"""
    return sum(sum(len(row) for row in c.rows.values()) for c in cases)


def part4_sparse():
    rule("4. SAVE_MODE='sparse' -- delta rows over the same matrix")

    # (a) steady state: one row per case. A first row has nothing to diff
    # against, so it is stored whole -- sparse cannot win here, but the
    # reconstruction must still be exact.
    steady = {}
    for mode in ("all", "sparse"):
        loop = build()
        with quiet():
            loop.run(**MATRIX, save_mode=mode, **SOLVE)
        steady[mode] = [c for c in loop.last_result.cases if c.rows]

    s_all, s_sparse = steady["all"], steady["sparse"]
    steady_cols, steady_bad = compare_dense(s_all, s_sparse)
    print("  steady run(), 6 cases x 1 row:")
    print(f"    cells stored   all={cells(s_all):>6}   sparse={cells(s_sparse):>6}")
    print(f"    numeric columns compared : {steady_cols}")
    print(f"    columns that disagree    : {steady_bad}")

    # (b) the transient over the SAME matrix: now each case is a time series,
    # most of whose columns (L, D, f, the pump curve, tank area) never move.
    trans = {}
    for mode in ("all", "sparse"):
        loop = build()
        with quiet():
            loop.sim_matrix(**MATRIX, save_mode=mode, **SIM)
        trans[mode] = [c for c in loop.last_result.cases if c.rows]

    t_all, t_sparse = trans["all"], trans["sparse"]
    c_all, c_sparse = cells(t_all), cells(t_sparse)
    rows_all = sum(len(c.rows) for c in t_all)
    rows_sparse = sum(len(c.rows) for c in t_sparse)
    trans_cols, trans_bad = compare_dense(t_all, t_sparse)

    print()
    print(f"  transient sim_matrix(), {len(t_all)} episodes:")
    print(f"    rows recorded  all={rows_all:>6}   sparse={rows_sparse:>6}")
    print(f"    cells stored   all={c_all:>6}   sparse={c_sparse:>6}")
    print(f"    sparse keeps {100.0 * c_sparse / c_all:.1f}% of the cells")
    print(f"    dense_dataframe reconstructs {len(t_sparse[0].dense_rows)} rows/episode")
    print(f"    numeric columns compared : {trans_cols}")
    print(f"    columns that disagree    : {trans_bad}")
    if trans_bad:
        # Honest about the one wrinkle rather than hiding it: the integrator
        # calls save_data twice at the same clock index, and the second delta
        # at that index stores only what changed -- so the forward-fill carries
        # the previous clock value for those rows.
        dup = sum(1 for r in t_sparse[0].rows.values() if "time" not in r)
        print(
            f"      ({dup} of {len(t_sparse[0].rows)} episode-0 deltas omit 'time';"
            " repeated clock index)"
        )

    # (c) the reconstruction contract itself, replayed on real rows: feed the
    # dense episode's own rows through Case.record_sparse in index order and
    # forward-fill them back. Same rows in, fewer cells stored, identical frame.
    source = t_all[0]
    replay = Case(dict(source.inputs))
    for row in source.numeric_rows:
        replay.record_sparse({k: v for k, v in row.items() if k != "index"})

    want = source.dataframe.drop(columns=["index"])
    got = replay.dense_dataframe.drop(columns=["index"])[want.columns]
    exact = want.equals(got)
    print()
    print("  replaying episode 0's dense rows through Case.record_sparse:")
    print(f"    rows        {len(source.numeric_rows)} -> {len(replay.rows)}")
    print(f"    cells       {cells([source]):>6} -> {cells([replay]):>6}")
    print(f"    dense_dataframe identical to the dense frame : {exact}")

    return dict(
        steady_cells_all=cells(s_all),
        steady_cells_sparse=cells(s_sparse),
        steady_bad=steady_bad,
        trans_cells_all=c_all,
        trans_cells_sparse=c_sparse,
        trans_rows_all=rows_all,
        trans_rows_sparse=rows_sparse,
        trans_bad=trans_bad,
        episodes=len(t_all),
        replay_cells_all=cells([source]),
        replay_cells_sparse=cells([replay]),
        replay_exact=exact,
    )


def compare_dense(dense_cases, sparse_cases):
    """forward-fill the sparse cases and diff them against the dense ones;
    :returns: (numeric columns compared, the ones that disagree)"""
    a = pandas.concat([c.dense_dataframe for c in dense_cases], ignore_index=True)
    b = pandas.concat([c.dense_dataframe for c in sparse_cases], ignore_index=True)
    common = [
        k
        for k in a.columns
        if k in b.columns
        and k != "index"
        and numpy.issubdtype(a[k].dtype, numpy.number)
    ]
    bad = [k for k in common if list(a[k]) != list(b[k])]
    return len(common), bad


# ---------------------------------------------------------------------------
# 5. the event bus: scoped listeners on value / slot / collection changes
# ---------------------------------------------------------------------------


@forge
class Shelf(ComponentIterator):
    """the pumps still under consideration, as a collection. `component_type`
    is an attrs field on ComponentIterator, so it is passed at construction
    rather than overridden on the class (which logs an override warning)."""


def part5_events():
    rule("5. EVENT BUS -- watch() value / slot / collection changes")

    pump_a = Pump(name="pump_a", h_shutoff=PUMP_ROWS[0][0], k_curve=PUMP_ROWS[0][1])
    pump_b = Pump(name="pump_b", h_shutoff=PUMP_ROWS[1][0], k_curve=PUMP_ROWS[1][1])

    values = []
    slots = []
    collections = []

    loop = build()
    with (
        events.watch(events.VALUE_CHANGED, lambda i, n, o, v: values.append(n)),
        events.watch(events.SLOT_CHANGED, lambda i, n, o, v: slots.append((n, v.name))),
        events.watch(
            events.COLLECTION_CHANGED, lambda c, k, it: collections.append((k, it.name))
        ),
    ):
        with quiet():
            shelf = Shelf(name="shelf", component_type=Pump)
            for pump in (pump_a, pump_b):
                shelf.append(pump)  # -> COLLECTION_CHANGED
            # a whole COMPONENT as a matrix value: swapping the pump is a SLOT
            # change, re-spec'ing the pipe is a VALUE change, and the solver
            # writing Q is a VALUE change too.
            loop.run(pump=[pump_a, pump_b], **{"pipe.D": DIAMETERS}, **SOLVE)

    tally = {}
    for name in values:
        tally[name] = tally.get(name, 0) + 1

    # Events carry the live object and its LOCAL field name -- 'D' on the pipe
    # instance, not the dotted 'pipe.D'. Names are resolved through the session
    # refs so there is one naming authority, not two string plumbings.
    print(f"  shelf              : {len(shelf)} pumps under consideration")
    print(f"  COLLECTION_CHANGED : {len(collections)}  {collections}")
    print(f"  SLOT_CHANGED       : {len(slots)}  {slots}")
    print(f"  VALUE_CHANGED      : {len(values)} total, by field:")
    for name, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {name:<16} {n}")

    # Outside the block there are no listeners, so the emitters short-circuit on
    # has_listeners and nothing is collected -- the zero-cost property.
    before = (len(values), len(slots), len(collections))
    listening = (
        events.has_listeners(events.VALUE_CHANGED),
        events.has_listeners(events.SLOT_CHANGED),
        events.has_listeners(events.COLLECTION_CHANGED),
    )
    again = build()
    with quiet():
        Shelf(name="shelf2", component_type=Pump).append(pump_b)
        again.run(pump=[pump_a, pump_b], **{"pipe.D": DIAMETERS}, **SOLVE)
    after = (len(values), len(slots), len(collections))

    print()
    print(f"  listeners after the watch block : {listening}")
    print(f"  an identical run outside it     : {before} -> {after}")

    return dict(
        tally=tally,
        n_values=len(values),
        slots=slots,
        collections=collections,
        before=before,
        after=after,
        listening=listening,
    )


# ---------------------------------------------------------------------------


def main():
    with quiet():
        change_all_log_levels(40)  # warnings and up; engforge is chatty at INFO

    n_full, n_group, result = part1_matrix()
    rows_per_case, worst_resid, worst_rel, df = part2_result(result)
    leaf_diffs, restored_clean, clock = part3_state_tree()
    sp = part4_sparse()
    ev = part5_events()

    rule("WHAT THIS RUN PROVED")
    print(f"  1 matrix     : {n_group} points, not {n_full}; pump specs stayed paired")
    print(
        f"  2 result     : {len(result.cases)} cases x {rows_per_case[0]} row, "
        f"combined frame {df.shape[0]}x{df.shape[1]}"
    )
    print(
        f"               : worst |head_error| {worst_resid:.2e}, "
        f"worst Q vs algebra {worst_rel:.2e}"
    )
    print(
        f"  3 state tree : {len(leaf_diffs)} branches, every restore exact, "
        f"clock rewound to {clock}"
    )
    print(
        f"  4 sparse     : transient cells {sp['trans_cells_all']} -> "
        f"{sp['trans_cells_sparse']} over the same {sp['trans_rows_all']} rows"
    )
    print(
        f"               : replay {sp['replay_cells_all']} -> "
        f"{sp['replay_cells_sparse']} cells, forward-fill exact "
        f"{sp['replay_exact']}"
    )
    print(
        f"  5 events     : {len(ev['collections'])} collection, "
        f"{len(ev['slots'])} slot, {ev['n_values']} value; "
        f"silent outside watch()"
    )

    # ---- regression checks: every number below was observed, not predicted ----

    # 1. a comma key is a partial factorial
    assert n_full == 12, n_full
    assert n_group == 6, n_group
    assert len(result.cases) == 6, len(result.cases)

    # 2. rows are owned per case, and the solver agrees with the algebra
    assert rows_per_case == [1] * 6, rows_per_case
    assert df.shape[0] == 6, df.shape
    assert worst_resid < 1e-6, worst_resid  # dimensionless residual
    assert worst_rel < 1e-6, worst_rel  # relative to the closed form
    assert list(result) == ["output", "input_sets"]
    assert result["input_sets"][0] == {
        "pipe.D": 0.032,
        "pump.h_shutoff": 18.0,
        "pump.k_curve": 600000.0,
    }, result["input_sets"][0]
    assert len(result["input_sets"]) == 6
    # physical sanity: every build lands in a sane pipe velocity band
    v = df["velocity"]
    assert 1.9 < v.min() and v.max() < 3.6, (v.min(), v.max())

    # 3. the state tree: diffs name exactly the keys that moved, restores clean
    assert restored_clean
    assert set(leaf_diffs["bigger line (DN50)"]) == {"pipe.D"}
    assert set(leaf_diffs["the other pump"]) == {"pump.h_shutoff", "pump.k_curve"}
    assert set(leaf_diffs["tank half full"]) == {"level"}
    assert leaf_diffs["bigger line (DN50)"]["pipe.D"] == (0.032, 0.050)
    assert clock == 0.0, clock  # restore_time=True rewound the episode clock

    # 4. sparse storage
    # one row per case: a baseline has nothing to diff against, so no savings
    assert sp["steady_cells_all"] == sp["steady_cells_sparse"] == 144, sp
    assert not sp["steady_bad"], sp["steady_bad"]  # and it reconstructs exactly
    assert sp["episodes"] == 6, sp["episodes"]
    assert sp["trans_rows_all"] == sp["trans_rows_sparse"] == 1302, sp
    # cell counts are asserted as a RATIO, not an exact figure: the row count is
    # fixed by dt/endtime, but how many cells each row stores depends on how
    # many values the integrator moved — a scipy change may shift it.
    assert sp["trans_cells_sparse"] < sp["trans_cells_all"] / 2, (
        sp["trans_cells_sparse"],
        sp["trans_cells_all"],
    )
    # the lone transient wrinkle is the repeated clock index; nothing else
    assert set(sp["trans_bad"]) <= {"time"}, sp["trans_bad"]
    # replayed in index order, the forward-fill is exact and much smaller
    assert sp["replay_exact"]
    assert sp["replay_cells_sparse"] < sp["replay_cells_all"] / 2, (
        sp["replay_cells_sparse"],
        sp["replay_cells_all"],
    )

    # 5. events
    assert ev["collections"] == [(0, "pump_a"), (1, "pump_b")], ev["collections"]
    # two swaps, not six: re-assigning the SAME component is not structural
    assert ev["slots"] == [("pump", "pump_a"), ("pump", "pump_b")], ev["slots"]
    # the VALUE_CHANGED count tracks solver iterations (most writes are the
    # solver probing Q), so assert the SHAPE — which fields moved, and that the
    # swept field moved once per point — not a scipy-version-dependent total.
    assert set(ev["tally"]) == {"Q", "D"}, ev["tally"]
    assert ev["tally"]["D"] == 5, ev["tally"]  # 6 pipe sizes, 5 transitions
    assert ev["tally"]["Q"] > 10, ev["tally"]  # the solver did real work
    assert ev["n_values"] == sum(ev["tally"].values()), (ev["n_values"], ev["tally"])
    assert ev["listening"] == (False, False, False), ev["listening"]
    assert ev["before"] == ev["after"], (ev["before"], ev["after"])  # zero cost

    print()
    print("  all assertions passed.")


if __name__ == "__main__":
    main()
