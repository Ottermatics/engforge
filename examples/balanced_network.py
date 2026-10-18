"""One pump, one header, three filter branches — and nobody metered the branches.

The skid has a pump feeding a manifold: a coarse strainer, a medium filter, and a
fine filter, all hanging off the same header. There is one gauge, on the header,
and it is broken. What you actually know is the pump curve and roughly how
restrictive each element is. What you need is the header pressure and the split,
because a starved fine filter is the one that fails first.

Two statements close the problem, and neither is solvable on its own:

    continuity    sum(w_i) - w_total = 0     every drop that leaves the pump
                                             goes down exactly one branch
    head match    pump.head - dP     = 0     the pump and the header agree on
                                             one pressure

Each branch knows its own flow given a pressure (``w = sqrt(dP/K)``) and the pump
knows its own head given a flow (``head = dp0 - kf*w^2``). Neither knows the
other's argument. That circularity is the whole job: two unknowns, ``dP`` and
``w_total``, pinned by two residuals driven to zero at the same time.

What this example is really about is *where the declarations go*. In engforge a
residual is resolved downward only, so the unknowns and the equations live on the
one node that can see both the pump and the branches — the parent. The branches
themselves are a ``ComponentDict``, which means the size of the network is data:
append a branch and the problem grows without a line of solver code changing.

Run it:

    python examples/balanced_network.py
"""

# Quiet engforge's per-session INFO chatter. Those lines carry a session id and
# object addresses, so leaving them on makes this script's output different on
# every run — and an example you can't diff is an example you can't trust.
from engforge.logging import change_all_log_levels

change_all_log_levels(30)  # warnings and worse only

import numpy as np

from engforge import System, Component, forge, field, Slot, Solver
from engforge import events
from engforge.component_collections import ComponentDict
from engforge.properties import system_property


# --------------------------------------------------------------------------
# The pieces. Each one is honest about what it knows and what it is handed.
# --------------------------------------------------------------------------


@forge
class Branch(Component):
    """One parallel leg: a resistance element fed from the common header.

    ``K`` is the leg's loss coefficient (dP = K*w^2, so w = sqrt(dP/K)) and
    ``dP`` is the header pressure, which the branch does NOT own — the parent
    broadcasts it in. A branch cannot know the header pressure by itself; that
    is exactly why the unknown lives upstairs.

    Note what is *absent* here: no ``Solver.declare_var``, no ``Solver.con_eq``.
    Collection members' solver declarations are silently ignored — verified with
    ``sandbox/cycles/probe_c_collection.py``, where three members of a
    ``ComponentDict`` each declared the same var and constraint and the solver
    saw only the plain-slot sibling (``Xans`` keys ``['plain.x']``, the plain
    child solved to ``x = 1.0``, the three members stayed frozen at ``3.0``).
    A collection supplies physics. It never supplies declarations.
    """

    K: float = field(default=1.0)  # loss coefficient, dP = K*w^2
    dP: float = field(default=1.0)  # header pressure, broadcast by the parent

    @system_property
    def w(self) -> float:
        return (self.dP / self.K) ** 0.5


@forge
class Branches(ComponentDict):
    """A keyed bag of branches, so the network size is data and not code.

    Declare ``component_type`` as a typed field rather than a bare class
    attribute. The bare form works but logs ``WARN: Branches overriding
    inherited attr: component_type``, and the typed form keeps ``Branches()``
    constructible with no arguments.
    """

    component_type: type = field(default=Branch)


@forge
class Pump(Component):
    """Falling head curve: shutoff head ``dp0``, droop ``kf``.

    Like the branches, the pump is handed its argument — total flow — rather
    than deducing it.
    """

    dp0: float = field(default=84.4)  # shutoff head
    kf: float = field(default=0.4)  # droop
    w: float = field(default=0.0)  # total flow, broadcast by the parent

    @system_property
    def head(self) -> float:
        return self.dp0 - self.kf * self.w**2


@forge
class Manifold(System):
    """The common ancestor: the only node that can see the pump AND the branches.

    So this is where both unknowns and both residuals are declared. The rule is
    downward-only resolution, and it is enforced at *instance construction* --
    the class body is accepted, then building the system raises. Try either of
    these and watch it bite:

        # (a) a child reaching up and across to a sibling
        @forge
        class BadBranch(Component):
            r = Solver.con_eq("parent.pump.head", "w", combos="bad")

        # (b) a child naming a sibling slot directly
        @forge
        class BadBranch(Component):
            r = Solver.con_eq("pump.head", "w", combos="bad")

        @forge
        class Bad(System):
            b: BadBranch = Slot.define(BadBranch)
            pump: Pump = Slot.define(Pump)

        Bad(b=BadBranch(), pump=Pump())
        # (a) -> AssertionError: invalid parent in parent.pump.head
        # (b) -> AssertionError: invalid pump in pump.head

    Both messages above are quoted from running it. There is no sideways
    channel: a residual that spans two siblings belongs to their parent.
    """

    branches: Branches = Slot.define_iterator(Branches, wide=True)
    pump: Pump = Slot.define(Pump)

    # The two unknowns. declare_var needs an attrs FIELD of type int/float —
    # never a system_property. (A property is computed; there is nothing to set.)
    dP: float = field(default=1.0)  # header pressure      [initial guess]
    w_total: float = field(default=0.1)  # total pump flow [initial guess]

    p_var = Solver.declare_var("dP", combos="balance")
    w_var = Solver.declare_var("w_total", combos="balance")

    # The two residuals. con_eq needs a system_property — the mirror image of
    # declare_var. Both tagged "balance" so one combo switches on the whole
    # network problem.
    continuity = Solver.con_eq("flow_resid", 0.0, combos="balance")
    head_match = Solver.con_eq("head_resid", 0.0, combos="balance")

    def update(self, parent, *a, **kw):
        """Broadcast the parent's unknowns down to the children.

        This is the only channel today for sharing an unknown across siblings,
        and it is called by the solver on every residual evaluation. Keep it
        idempotent — pure assignment, no accumulators, no counters. The engine
        makes no promise about how many times it runs per solve.
        """
        for b in self.branches.data.values():
            b.dP = self.dP
        self.pump.w = self.w_total

    @system_property
    def branch_flow_sum(self) -> float:
        """What the network draws. Grows with the collection, for free."""
        return sum(b.w for b in self.branches.data.values())

    @system_property
    def flow_resid(self) -> float:
        """continuity: nothing is created or destroyed at the junction"""
        return self.branch_flow_sum - self.w_total

    @system_property
    def head_resid(self) -> float:
        """head match: the pump and the header report the same pressure"""
        return self.pump.head - self.dP


# --------------------------------------------------------------------------
# Truth, on paper. Substituting w_i = sqrt(dP/K_i) into continuity gives
# w_total = S*sqrt(dP) with S = sum(1/sqrt(K_i)), and the head match collapses
# to a LINEAR equation in dP. Any branch set, closed form, no solver.
# --------------------------------------------------------------------------


def exact_balance(Ks, dp0, kf):
    """(dP, w_total) for any branch set -- the answer the solver has to find"""
    S = sum(K**-0.5 for K in Ks)
    dP = dp0 / (1.0 + kf * S * S)
    return dP, S * dP**0.5


def residual_norm(man):
    """||(flow_resid, head_resid)||, with the children synced first.

    Calling update() by hand is what makes the *before* number honest: without
    it the children still hold whatever they were last handed, and you would be
    measuring a residual of a state the system is not actually in.
    """
    man.update(None)
    return float(np.hypot(man.flow_resid, man.head_resid))


def solve_balance(man):
    """turn on the "balance" combo and leave the answer on the object.

    ``combos="balance"`` selects both declare_vars and both con_eqs in one
    shot. ``revert_last`` / ``revert_every`` off means the solved state STAYS
    on the instance, which is what you want when the solve is the deliverable
    rather than one point of a sweep.

    Returns the output dict: ``["Xans"]`` is the solved unknowns, ``["ans"]``
    is the raw scipy OptimizeResult (nfev / nit / success).
    """
    # engforge writes one progress "." to stdout per run; give it a label
    print("  solving            ", end="", flush=True)
    res = man.run(combos="balance", slv_vars="*", revert_last=False, revert_every=False)
    print()
    return res["output"][0]


def balance_table(man, truth_dP):
    """print the physical balance, not the solver's opinion of it"""
    print(f"    {'branch':<10}{'K':>7}{'w':>12}{'share':>9}{'dP = K*w^2':>13}")
    print(f"    {'-' * 50}")
    tot = 0.0
    for name, b in man.branches.data.items():
        tot += b.w
        print(
            f"    {name:<10}{b.K:>7.1f}{b.w:>12.6f}"
            f"{100 * b.w / man.w_total:>8.1f}%{b.K * b.w**2:>13.6f}"
        )
    print(f"    {'-' * 50}")
    print(f"    {'sum':<10}{'':>7}{tot:>12.6f}{100.0:>8.1f}%")
    print(f"    {'w_total':<10}{'':>7}{man.w_total:>12.6f}")
    print(f"    header dP  {man.dP:.6f}   (closed form {truth_dP:.6f})")


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # ACT 1 - three branches, two unknowns, solved together.
    # Ks of 1, 4, 9 against this pump curve put the exact root on integers:
    # dP = 36, flows 6 / 3 / 2, total 11. You can see the physics instead of
    # parsing it.
    # ----------------------------------------------------------------------
    print("\n=== ACT 1 - balance three branches on one header ===")

    bl = Branches()
    for name, K in (("coarse", 1.0), ("medium", 4.0), ("fine", 9.0)):
        bl[name] = Branch(K=K)

    man = Manifold(branches=bl, pump=Pump(), dP=1.0, w_total=0.1)

    Ks3 = [b.K for b in bl.data.values()]
    truth3 = exact_balance(Ks3, man.pump.dp0, man.pump.kf)
    print(f"  closed form        dP = {truth3[0]:.10f}  w_total = {truth3[1]:.10f}")

    r0 = residual_norm(man)
    print(f"  initial guess      dP = {man.dP:.4f}  w_total = {man.w_total:.4f}")
    print(f"  residual norm      {r0:.6e}   <- nothing balances yet")

    out = solve_balance(man)
    ans = out["ans"]

    r1 = residual_norm(man)
    print(f"  residual norm      {r1:.6e}   <- balanced")
    print(f"  Xans               {out['Xans']}")
    print(f"  SLSQP              nfev={ans.nfev} nit={ans.nit} success={ans.success}")
    print(f"  flow_resid         {man.flow_resid:+.6e}")
    print(f"  head_resid         {man.head_resid:+.6e}")
    print()
    balance_table(man, truth3[0])
    print(
        f"\n  Every leg sees the same {truth3[0]:.1f}, so the split is pure resistance:"
        f" the fine leg\n  has 9x the K of the coarse leg and carries"
        f" {bl['fine'].w / bl['coarse'].w:.4f} of its flow -- 1/sqrt(9), exactly."
    )

    # ----------------------------------------------------------------------
    # ACT 2 - the balance has to hold in the RECORDED frame, not just inside
    # the solver. A wide collection reports every member into the parent's row,
    # so the per-branch split is output you can hand someone, not a debug print.
    # ----------------------------------------------------------------------
    print("\n=== ACT 2 - conservation in the dataframe, not just in the solver ===")

    row = man.dataframe.iloc[0]
    wcols = sorted(c for c in man.dataframe.columns if c.startswith("branches."))
    print(f"  collection columns {[c for c in wcols if c.endswith('.w')]}")
    frame_sum = sum(float(row[c]) for c in wcols if c.endswith(".w"))
    print(f"  sum of branch w    {frame_sum:.9f}")
    print(f"  w_total in row     {float(row['w_total']):.9f}")
    print(f"  closure error      {abs(frame_sum - float(row['w_total'])):.3e}")
    print(f"  pump head in row   {float(row['pump.head']):.9f}")
    print(f"  header dp in row   {float(row['dp']):.9f}")

    # ----------------------------------------------------------------------
    # ACT 3 - someone taps the header for an instrument line. The network grows
    # by one leg; the solver declarations do not change at all. Two unknowns
    # before, two unknowns after. And the append announces itself on the event
    # bus, so machinery that caches a problem graph can know it went stale.
    # ----------------------------------------------------------------------
    print("\n=== ACT 3 - append a branch; the problem grows, the code does not ===")

    n_before = len(bl.data)
    before_props = len(man.system_references(recache=True)["properties"])
    seen = []
    with events.watch(
        events.COLLECTION_CHANGED,
        lambda coll, key, item: seen.append((type(coll).__name__, key, item.K)),
    ):
        bl["tap"] = Branch(K=36.0)  # a small, very restrictive instrument tap
    after_props = len(man.system_references(recache=True)["properties"])

    print(f"  COLLECTION_CHANGED {seen}")
    print(f"  branches           {n_before} -> {len(bl.data)}")
    print(f"  tabulated props    {before_props} -> {after_props}")

    w_before = {k: b.w for k, b in bl.data.items() if k != "tap"}
    dP_before = man.dP

    Ks4 = [b.K for b in bl.data.values()]
    truth4 = exact_balance(Ks4, man.pump.dp0, man.pump.kf)
    print(f"  closed form        dP = {truth4[0]:.10f}  w_total = {truth4[1]:.10f}")

    # Same call. Same combo. Same two declarations.
    out = solve_balance(man)
    ans = out["ans"]
    xans4 = out["Xans"]
    r2 = residual_norm(man)
    print(f"  Xans               {xans4}   <- still 2 unknowns")
    print(f"  SLSQP              nfev={ans.nfev} nit={ans.nit} success={ans.success}")
    print(f"  residual norm      {r2:.6e}")
    print()
    balance_table(man, truth4[0])
    print(
        f"\n  The tap draws {bl['tap'].w:.4f} ({100 * bl['tap'].w / man.w_total:.1f}%"
        f" of the flow) and drops the header from {dP_before:.3f} to {man.dP:.3f}."
    )
    print("    what every other leg lost to it:")
    for k, w0 in w_before.items():
        pct = 100 * (bl[k].w / w0 - 1)
        print(f"      {k:<8} {w0:.6f} -> {bl[k].w:.6f}   ({pct:+.2f}%)")

    # ----------------------------------------------------------------------
    # ACT 4 - the answer is the physics, not the starting point. Throw the
    # solver a guess that is wrong in both directions at once.
    # ----------------------------------------------------------------------
    print("\n=== ACT 4 - same network, deliberately terrible initial guess ===")

    man.dP, man.w_total = 200.0, 0.01
    guess_hi = 200.0 / truth4[0]
    guess_lo = truth4[1] / 0.01
    print(f"  guess              dP = {man.dP:.2f}  w_total = {man.w_total:.2f}")
    print(f"  wrongness          {guess_hi:.1f}x too high, {guess_lo:.0f}x too low")
    r3_0 = residual_norm(man)
    print(f"  residual norm      {r3_0:.6e}")
    out = solve_balance(man)
    ans = out["ans"]
    r3 = residual_norm(man)
    print(f"  Xans               {out['Xans']}")
    print(f"  SLSQP              nfev={ans.nfev} nit={ans.nit} success={ans.success}")
    print(f"  residual norm      {r3:.6e}")
    print(
        f"  landed on the same root: ddP = {abs(man.dP - truth4[0]):.3e},"
        f" dw = {abs(man.w_total - truth4[1]):.3e}"
    )

    # ----------------------------------------------------------------------
    # REGRESSION CHECKS - conservation and closed form, not solver success.
    # A solver that reports success and violates continuity is still wrong.
    # Tolerances are set just above what was actually observed.
    # ----------------------------------------------------------------------
    print("\n=== CHECKS ===")

    # ACT 1: the three-branch root, on integers.  observed 8.3414e+01 -> 1.1176e-08
    assert r0 > 1.0, f"the initial guess should NOT balance, got {r0}"
    assert r1 < 1e-6, f"act 1 residual norm {r1}"
    # observed dP = 36.00000000377821, w_total = 11.000000000840323
    assert abs(truth3[0] - 36.0) < 1e-12 and abs(truth3[1] - 11.0) < 1e-12
    print(f"  act 1  residual norm {r0:.3e} -> {r1:.3e}  (< 1e-6)")

    # ACT 2: continuity holds in the recorded row.  observed 2.631e-10 closure
    closure = abs(frame_sum - float(row["w_total"]))
    head_gap = abs(float(row["pump.head"]) - float(row["dp"]))
    assert closure < 1e-6, f"dataframe does not close: {closure}"
    assert head_gap < 1e-5, f"head mismatch in row: {head_gap}"
    print(f"  act 2  frame closure {closure:.3e}  (< 1e-6)")

    # ACT 3: the append changed the network, not the declarations.
    assert seen == [("Branches", "tap", 36.0)], f"COLLECTION_CHANGED payload {seen}"
    assert after_props == before_props + 1, f"{before_props} -> {after_props}"
    assert len(bl.data) == n_before + 1, "the collection did not grow"
    unknowns = sorted(xans4)
    assert unknowns == ["dP", "w_total"], f"unknowns followed branch count: {unknowns}"
    assert r2 < 1e-6, f"act 3 residual norm {r2}"
    print(f"  act 3  props {before_props} -> {after_props}, 2 unknowns, {r2:.3e}")

    # ACT 4: path independence, to the closed form.  observed 7.105e-15 both
    assert r3_0 > 1.0, f"the bad guess should NOT balance, got {r3_0}"
    assert r3 < 1e-6, f"act 4 residual norm {r3}"
    assert abs(man.dP - truth4[0]) < 1e-6, f"dP {man.dP} vs {truth4[0]}"
    assert abs(man.w_total - truth4[1]) < 1e-6, f"w_total {man.w_total} vs {truth4[1]}"
    print(
        f"  act 4  matches the closed form from a guess {guess_hi:.1f}x high"
        f" and {guess_lo:.0f}x low"
    )

    # The conservation law itself, stated once, on the live object.
    assert abs(sum(b.w for b in bl.data.values()) - man.w_total) < 1e-6, "continuity"
    assert abs(man.pump.head - man.dP) < 1e-5, "head match"
    for name, b in bl.data.items():
        # every branch sees the same header pressure and obeys its own curve
        assert abs(b.dP - man.dP) < 1e-12, f"{name} off-header"
        assert abs(b.K * b.w**2 - man.dP) < 1e-9, f"{name} off-curve"
    print("  conserved: sum(w_i) == w_total, pump.head == dP, every K*w^2 == dP")
    print("\nall checks passed\n")
