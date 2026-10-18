"""Design vs off-design: one model, two questions, selected by combos.

A gas-generator spool goes on the test stand next month, and there are two
questions on the whiteboard that look like one question:

  1. "What turbine do I have to buy?"  The compressor is chosen, the duty is
     written in the spec -- 5:1 pressure ratio at rated flow and rated turbine
     inlet temperature. How big a throat, and how much work capacity, does the
     turbine need to sit exactly there?

  2. "What happens when I pull the fuel back?"  The turbine is welded in. On a
     part-power morning the turbine inlet temperature is 10% low. What speed,
     flow and pressure ratio does the spool actually settle at?

Those are the SAME two equations. Mass has to get through the turbine throat,
and the turbine has to pay the compressor's shaft bill. Nothing about the
physics changes between question 1 and question 2 -- the only thing that
changes is which symbols you are allowed to move.

That is the whole idea here. In engforge a residual is a statement about the
world; *which symbol is unknown* is a separate, run-time choice. You tag the
unknowns with `combos=` and pick the tag at `run()` time:

    size_A  = Solver.declare_var("A_t", combos="design")      # hardware
    run_N   = Solver.declare_var("N",   combos="offdesign")   # operating point
    r_flow  = Solver.con_eq("flow_net", 0.0, combos="design,offdesign")

    spool.run(combos="design")      -> unknowns are (A_t, S_t), duty is held
    spool.run(combos="offdesign")   -> unknowns are (N, W),  hardware is held

One class. One residual set. Two unknown partitions. That is the turbomachinery
cycle-deck workflow, and it works today.

The physics (nondimensional, inlet conditions normalized to 1):

    compressor head    PR  = 1 + p1*N^2 - p2*W^2          (head ~ N^2, loss ~ W^2)
    compressor shaft   Pc  = W*(PR^m - 1)/eta_c + kw*N^3  (ideal work / eff + windage)
    turbine swallow    Wc  = A_t*PR/sqrt(theta)           (choked: W*sqrt(T4)/P4 = const)
    turbine shaft      Pt  = S_t*W*theta*(1 - PR^-m)      (work ~ T4 * expansion)

    residual 1 (flow)   W  - Wc = 0
    residual 2 (shaft)  Pt - Pc = 0

with m = (gamma-1)/gamma = 2/7. Those two exponents are why this problem is
well behaved: the compressor's bill grows like PR^m while the turbine's income
saturates like (1 - PR^-m), so the two curves cross once on the operating line
rather than running nearly parallel. Verified -- exactly one crossing in
PR = [1.05, 11] at every temperature swept below. (Drop the bracket to 1.0001
and a second appears right at PR = 1, where windage alone balances a spool that
is barely turning. It is a numerical artifact of the N^3 term, not a machine.)

Single-shaft, no bleed, no fuel-flow addition -- the first-pass matching
assumption. The duty is normalized so design is N = W = theta = 1.

Run:  conda run -n pace python examples/design_offdesign.py
"""

# Quiet the per-solve engforge chatter BEFORE any logging units exist, so the
# report below is the only thing on stdout (the PROB| session ids are uuid4 --
# left on, this example's output would differ run to run).
import engforge.logging as _elog

_elog.LOG_LEVEL = 50
_elog.LoggingMixin.log_level = 50

import math

from scipy.optimize import brentq

from engforge import System, Component, forge, field, Slot, Solver
from engforge.properties import system_property

M = 2.0 / 7.0  # (gamma-1)/gamma for air, gamma = 1.4


# --- the compressor: a fixed machine, a characteristic, a shaft bill ---------
@forge
class Compressor(Component):
    """Head-flow characteristic + shaft power absorbed, both at (N, W)."""

    N: float = field(default=1.0)  # spool speed, fraction of design
    W: float = field(default=1.0)  # mass flow, fraction of design

    p1: float = field(default=4.5)  # head coefficient    (rise ~ N^2)
    p2: float = field(default=0.5)  # throttling coefficient (loss ~ W^2)
    eta: float = field(default=0.85)  # polytropic-ish efficiency
    k_wind: float = field(default=0.05)  # windage / bearing drag ~ N^3

    @system_property
    def PR(self) -> float:
        return 1.0 + self.p1 * self.N**2 - self.p2 * self.W**2

    @system_property
    def pwr_absorbed(self) -> float:
        return self.W * (self.PR**M - 1.0) / self.eta + self.k_wind * self.N**3


# --- the turbine: the part being SIZED. A_t and S_t are its two sizing knobs -
@forge
class Turbine(Component):
    """A choked turbine. Flow capacity is geometry; work capacity is staging.

    A_t -- effective throat area. A choked nozzle passes a fixed CORRECTED
           flow, so W*sqrt(T4)/P4 is a constant of the hardware. Nondimensional:
           W = A_t * PR / sqrt(theta).
    S_t -- work scalar (stage count x efficiency x cp ratio), lumped.
    """

    W: float = field(default=1.0)
    PR: float = field(default=5.0)
    theta: float = field(default=1.0)  # T4 / T4_design -- the throttle

    A_t: float = field(default=0.2)  # SIZING unknown
    S_t: float = field(default=2.0)  # SIZING unknown

    @system_property
    def flow_capacity(self) -> float:
        return self.A_t * self.PR / math.sqrt(self.theta)

    @system_property
    def pwr_delivered(self) -> float:
        return self.S_t * self.W * self.theta * (1.0 - self.PR**-M)


# --- the spool: where every unknown and every residual has to live -----------
@forge
class Spool(System):
    """Compressor + turbine on one shaft.

    All four unknowns are fields HERE, not on the children, because engforge
    resolves solver declarations downward only: the node that declares a
    residual must be able to see every term in it, and only the parent can see
    both the compressor and the turbine. (Declaring `A_t` through a dotted path
    -- Solver.declare_var("turb.A_t") -- does not work: it logs
    `WARN: state| key A_t not in refs` and the "solved" value comes back equal
    to the start value. Verified. Keep unknowns local and broadcast them.)
    """

    comp: Compressor = Slot.define(Compressor)
    turb: Turbine = Slot.define(Turbine)

    # --- operating point -----------------------------------------------------
    N: float = field(default=1.0)
    W: float = field(default=1.0)
    theta: float = field(default=1.0)
    # --- hardware ------------------------------------------------------------
    A_t: float = field(default=0.2)
    S_t: float = field(default=2.0)

    # THE MECHANISM. Four unknowns, two disjoint tags. Passing `combos=`
    # explicitly REPLACES declare_var's default of f"default,{var}" -- so these
    # vars are reachable ONLY through the tag written here.
    size_A = Solver.declare_var("A_t", combos="design")
    size_S = Solver.declare_var("S_t", combos="design")
    run_N = Solver.declare_var("N", combos="offdesign")
    run_W = Solver.declare_var("W", combos="offdesign")

    # ...and ONE residual set, true in both modes, so it carries both tags.
    # There is no second class and no second model. This is the entire trick.
    r_flow = Solver.con_eq("flow_net", 0.0, combos="design,offdesign")
    r_shaft = Solver.con_eq("shaft_net", 0.0, combos="design,offdesign")

    def update(self, parent, *a, **kw):
        """Push the parent's unknowns down into the children.

        `update()` is the only channel for sharing an unknown across siblings
        today. It must stay idempotent: engforge calls it once per residual
        evaluation, not once per run -- measured on the runs below at 58 calls
        for section 1's design solve (nfev 6) and 136 for section 2's
        off-design solve (nfev 17); a single off-design point started on the
        answer costs 36 (nfev 3). The count tracks solver iterations, so do not
        depend on any particular value -- depend on it being many. Assignments
        only; put an accumulator in here and it will be wrong by a factor
        nobody can predict.
        """
        self.comp.N = self.N
        self.comp.W = self.W
        self.turb.W = self.W
        self.turb.theta = self.theta
        self.turb.A_t = self.A_t
        self.turb.S_t = self.S_t
        self.turb.PR = self.comp.PR

    @system_property
    def flow_net(self) -> float:
        """Continuity: what the compressor pumps minus what the throat passes."""
        return self.W - self.turb.flow_capacity

    @system_property
    def shaft_net(self) -> float:
        """Shaft: turbine income minus compressor bill."""
        return self.turb.pwr_delivered - self.comp.pwr_absorbed


# --- helpers -----------------------------------------------------------------
def spool(**kw):
    """A fresh, fully-populated Spool."""
    return Spool(comp=Compressor(), turb=Turbine(), **kw)


def solve(system, mode, **matrix):
    """Run one partition and hand back (Result, first scipy OptimizeResult)."""
    res = system.run(
        combos=mode,
        slv_vars="*",
        revert_last=False,  # keep the answer ON the object
        revert_every=False,  # and warm-start each sweep point from the last
        **matrix,
    )
    return res, res["output"][0]["ans"]


# The three functions below are the ANSWER KEY. They read their coefficients
# off the Compressor instance rather than repeating them, so they cannot drift
# out of step with the model if you retune it.
def pressure_ratio(c, N, W):
    return 1.0 + c.p1 * N**2 - c.p2 * W**2


def hand_sized(c, PR_d, N_d=1.0, W_d=1.0, theta_d=1.0):
    """Design mode in closed form -- the residuals are LINEAR in (A_t, S_t).

    flow:  W = A_t*PR/sqrt(theta)       ->  A_t = W*sqrt(theta)/PR
    shaft: S_t*W*theta*(1-PR^-m) = Pc   ->  S_t = Pc / (W*theta*(1-PR^-m))

    That linearity is why section 1 lands in nfev = 6. Sizing is the easy
    direction; the physics is only nonlinear in the operating point.
    """
    A_t = W_d * math.sqrt(theta_d) / PR_d
    Pc = W_d * (PR_d**M - 1.0) / c.eta + c.k_wind * N_d**3
    S_t = Pc / (W_d * theta_d * (1.0 - PR_d**-M))
    return A_t, S_t


def truth_offdesign(c, theta, A_t, S_t):
    """Independent truth: eliminate down to one equation in PR, bracket-solve.

    Pick PR  ->  the flow residual gives W  ->  the characteristic gives N  ->
    the shaft residual is then a scalar function of PR alone. brentq, not
    SLSQP, and not engforge -- so agreement means something.

    Bracketed from PR = 1.05 deliberately. The windage term leaves a second,
    near-PR=1 crossing (a numerical idle point, not a machine anyone runs);
    the upper root is the operating line.
    """

    def shaft_of_PR(PR):
        W = A_t * PR / math.sqrt(theta)
        N = math.sqrt((PR - 1.0 + c.p2 * W * W) / c.p1)
        income = S_t * W * theta * (1.0 - PR**-M)
        bill = W * (PR**M - 1.0) / c.eta + c.k_wind * N**3
        return income - bill

    PR = brentq(shaft_of_PR, 1.05, 11.0, xtol=1e-15, rtol=8.9e-16)
    W = A_t * PR / math.sqrt(theta)
    N = math.sqrt((PR - 1.0 + c.p2 * W * W) / c.p1)
    return N, W, PR


if __name__ == "__main__":
    COEFF = Compressor()  # the characteristic, for the answer key
    PR_DESIGN = pressure_ratio(COEFF, 1.0, 1.0)  # = 1 + p1 - p2 = 5.0

    # ======================================================================
    # 0. The segmentation table, read straight off the class.
    #    This is the whole design/off-design split, and it is four lines of
    #    declaration plus two shared residuals.
    # ======================================================================
    print("\n=== declarations: who is unknown, and when ===")
    probe = spool()
    print(f"  {'declaration':<11} {'kind':<5} {'symbol':<16} combos")
    for name, slv in probe.solvers.items():
        # on an INSTANCE, .var / .lhs are resolved Refs -- .key is the plain
        # dotted name (the Ref's own repr carries a memory address).
        if slv.slvtype == "var":
            sym = slv.var.key
        else:
            sym = f"{slv.lhs.key} = {slv.rhs}"
        print(f"  {name:<11} {slv.slvtype:<5} {sym:<16} {','.join(slv.combos)}")

    # ======================================================================
    # 1. DESIGN -- the duty is given, the hardware is unknown.
    #    N, W and theta are plain fields that nothing tags under "design", so
    #    the solver cannot touch them. They ARE the spec.
    # ======================================================================
    print("\n=== 1. design: hold the duty, solve the hardware ===")
    sized = spool(N=1.0, W=1.0, theta=1.0, A_t=0.30, S_t=1.00)  # bad guesses
    print(f"  spec:  N = {sized.N}, W = {sized.W}, theta = {sized.theta}")
    print(f"  start: A_t = {sized.A_t}, S_t = {sized.S_t}")

    res_d, ans_d = solve(sized, "design")
    A_t, S_t = sized.A_t, sized.S_t
    hA, hS = hand_sized(COEFF, PR_DESIGN)

    print(f"\n  Xans:  {res_d['output'][0]['Xans']}")
    print(f"  A_t  = {A_t:.12f}   (closed form {hA:.12f})")
    print(f"  S_t  = {S_t:.12f}   (closed form {hS:.12f})")
    print(f"  duty untouched: N = {sized.N}, W = {sized.W}, PR = {sized.comp.PR}")
    print(f"  residuals: flow = {sized.flow_net:+.3e}, shaft = {sized.shaft_net:+.3e}")
    print(f"  SLSQP: nfev = {ans_d.nfev}, nit = {ans_d.nit}, success = {ans_d.success}")

    # ======================================================================
    # 2. The consistency check every cycle deck has to pass.
    #    SAME OBJECT. The hardware it just solved for stays put; now flip the
    #    partition and make the operating point the unknown. Knock N and W
    #    off the answer first, so the solver has to actually come back.
    # ======================================================================
    print("\n=== 2. off-design at design conditions (the consistency check) ===")
    sized.N, sized.W = 0.80, 0.80
    print(f"  kicked to: N = {sized.N}, W = {sized.W}  (hardware held at the sized values)")

    res_c, ans_c = solve(sized, "offdesign")
    print(f"\n  Xans:  {res_c['output'][0]['Xans']}")
    print(f"  N    = {sized.N:.12f}   (design point 1.0)")
    print(f"  W    = {sized.W:.12f}   (design point 1.0)")
    print(f"  PR   = {sized.comp.PR:.12f}   (design point {PR_DESIGN})")
    print(f"  hardware untouched: A_t = {sized.A_t:.12f}, S_t = {sized.S_t:.12f}")
    print(f"  residuals: flow = {sized.flow_net:+.3e}, shaft = {sized.shaft_net:+.3e}")
    print(f"  SLSQP: nfev = {ans_c.nfev}, nit = {ans_c.nit}, success = {ans_c.success}")
    consistency_err = max(abs(sized.N - 1.0), abs(sized.W - 1.0))
    print(f"  worst deviation from the design point: {consistency_err:.3e}")

    # ======================================================================
    # 3. OFF-DESIGN IS A SWEEP. Design is one point -- that is what "design
    #    point" means. Everywhere else is the input matrix: theta is a plain
    #    field, so a list of thetas is just four (seven) runs of the same
    #    partition, warm-started off each other.
    # ======================================================================
    print("\n=== 3. off-design: pull the fuel back ===")
    THETAS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    line = spool(A_t=A_t, S_t=S_t, N=1.0, W=1.0)  # the sized machine
    res_o, _ = solve(line, "offdesign", theta=THETAS)

    print(
        f"\n  {'T4/T4d':>7} {'N':>10} {'W':>10} {'PR':>8}"
        f" {'dN':>8} {'dW':>8} {'|err| vs brentq':>16} {'nfev':>5}"
    )
    swept = {}
    worst_truth_err = 0.0
    for i in sorted(res_o["output"]):
        out = res_o["output"][i]
        th = res_o["input_sets"][i]["theta"]
        Nq, Wq = out["Xans"]["N"], out["Xans"]["W"]
        PRq = pressure_ratio(COEFF, Nq, Wq)
        Nt, Wt, PRt = truth_offdesign(COEFF, th, A_t, S_t)
        err = max(abs(Nq - Nt), abs(Wq - Wt), abs(PRq - PRt))
        worst_truth_err = max(worst_truth_err, err)
        swept[th] = (Nq, Wq, PRq)
        print(
            f"  {th:>7.2f} {Nq:>10.6f} {Wq:>10.6f} {PRq:>8.4f}"
            f" {Nq - 1:>+7.1%} {Wq - 1:>+7.1%} {err:>16.3e}"
            f" {out['ans'].nfev:>5}"
        )

    # the recorded frame, not just the solver's scratch state (columns are
    # lower-cased on the way into the dataframe: N -> "n", comp.PR -> "comp.pr")
    df = res_o.dataframe
    worst_row_resid = max(
        max(abs(r), abs(s)) for r, s in zip(df["flow_net"], df["shaft_net"])
    )
    print(f"\n  rows recorded: {len(df)}   worst |residual| in the frame: {worst_row_resid:.3e}")

    n_loss = 1.0 - swept[0.70][0]
    print(
        f"  Field note: a 5% cut in turbine inlet temperature costs"
        f" {1 - swept[0.95][0]:.1%} of the speed."
    )
    print(
        f"              a 30% cut costs {n_loss:.1%}, and pressure ratio falls"
        f" {PR_DESIGN:.1f} -> {swept[0.70][2]:.2f}."
    )
    print("              The spool amplifies temperature. Nothing in the model says so;")
    print("              it falls out of two exponents that disagree.")

    # ======================================================================
    # 4. The trap, run rather than narrated. Tag both combos at once and you
    #    have FOUR unknowns for TWO equations. The solver does not complain.
    #    It returns a point where both residuals are satisfied -- and it is
    #    not the machine you designed, nor an operating point of it.
    # ======================================================================
    print("\n=== 4. the trap: combos='design,offdesign' ===")
    both = spool(A_t=A_t, S_t=S_t, theta=1.0, N=0.80, W=0.80)
    res_b, ans_b = solve(both, "design,offdesign")
    print(f"\n  Xans: {res_b['output'][0]['Xans']}")
    print(f"  residuals: flow = {both.flow_net:+.3e}, shaft = {both.shaft_net:+.3e}  <- satisfied")
    print(f"  but  A_t = {both.A_t:.6f} (sized {A_t:.6f}),  S_t = {both.S_t:.6f} (sized {S_t:.6f})")
    print(f"       N   = {both.N:.6f} (design 1.0),    W   = {both.W:.6f} (design 1.0)")
    print(f"  SLSQP reported success = {ans_b.success}, nfev = {ans_b.nfev}")
    print("  Four unknowns, two equations: a satisfied residual is not a determined problem.")

    # ======================================================================
    # ASSERTIONS -- every number below was observed, and this file doubles as
    # a regression check on the combo partition machinery.
    # ======================================================================

    # 1. design mode moved the hardware and ONLY the hardware
    assert set(res_d["output"][0]["Xans"]) == {"A_t", "S_t"}, res_d["output"][0]["Xans"]
    assert abs(A_t - 0.2) < 1e-12, A_t  # observed 0.19999999999999998
    assert abs(S_t - 1.998960079296221) < 1e-8, S_t  # observed 1.9989600799843958
    # vs the closed form: A_t to 2.8e-17, S_t to 6.9e-10
    assert abs(A_t - hA) < 1e-12 and abs(S_t - hS) < 1e-8
    assert sized.A_t == A_t and sized.S_t == S_t

    # 2. the consistency check: the sized machine, asked where it runs at
    #    design conditions, comes back to the design point. Kicked to
    #    (0.80, 0.80), it returns to (1.0, 1.0) -- observed worst deviation
    #    1.171e-09, and PR back to 5.0 within 5.903e-09.
    assert consistency_err < 1e-8, consistency_err
    assert abs(sized.comp.PR - PR_DESIGN) < 1e-7, sized.comp.PR
    assert set(res_c["output"][0]["Xans"]) == {"N", "W"}  # partition really swapped
    assert sized.A_t == A_t and sized.S_t == S_t  # hardware bit-for-bit untouched

    # 3. the sweep: 7 thetas -> 7 cases, each agreeing with an independent
    #    brentq on the reduced 1-D residual. Worst observed error 7.425e-06,
    #    at theta = 0.95 -- the first warm-start step off the design point.
    #    Why 1e-5 and not 1e-10: the flat path drives the residual SUM OF
    #    SQUARES, so near a root the objective is quadratically flat. SLSQP's
    #    default tol=1e-10 on that objective is only worth ~1e-6 in N and W,
    #    and PR = 1 + 4.5*N^2 - 0.5*W^2 amplifies it by ~9x. The residuals
    #    themselves are much better than the unknowns -- see the frame check.
    assert len(res_o["output"]) == len(THETAS) == 7
    assert worst_truth_err < 1e-5, worst_truth_err
    assert worst_row_resid < 1e-6, worst_row_resid  # observed 3.725e-07
    assert len(df) == 7, len(df)
    # physics: monotone wind-down in speed, flow and pressure ratio
    Ns = [swept[t][0] for t in THETAS]
    Ws = [swept[t][1] for t in THETAS]
    PRs = [swept[t][2] for t in THETAS]
    assert Ns == sorted(Ns, reverse=True), Ns
    assert Ws == sorted(Ws, reverse=True), Ws
    assert PRs == sorted(PRs, reverse=True), PRs
    # amplification: the spool loses more speed than the temperature it lost.
    # observed 10.4% speed for a 5% temperature cut -> a factor of 2.08.
    amplification = (1 - swept[0.95][0]) / 0.05
    assert amplification > 2.0, amplification
    assert 0.62 < n_loss < 0.63, n_loss  # observed 0.620653 for a 30% cut
    assert line.A_t == A_t and line.S_t == S_t  # the sweep never touched hardware

    # 4. under-determined: residuals driven to zero at a WRONG point
    assert len(res_b["output"][0]["Xans"]) == 4
    assert abs(both.flow_net) < 1e-6 and abs(both.shaft_net) < 1e-6
    assert abs(both.A_t - A_t) > 0.01, both.A_t  # observed 0.1819 vs 0.2000
    assert abs(both.N - 1.0) > 0.05, both.N  # observed 0.8831 vs 1.0

    print("\nOK: 4 unknowns, 2 residuals, 2 partitions.")
    print(f"    design sized (A_t, S_t) = ({A_t:.6f}, {S_t:.6f}) to closed form;")
    print(f"    off-design returned to the design point within {consistency_err:.2e};")
    print(f"    7 swept operating points matched brentq within {worst_truth_err:.2e}.")

    # ======================================================================
    # WHAT TO NOTICE -- combos as the segmentation tool
    # ======================================================================
    # * The residuals never moved. `flow_net` and `shaft_net` are statements
    #   about a shaft and a throat; they do not know, and must not know, which
    #   of their symbols you happen to be solving for this afternoon.
    #
    # * The partition is not in the model, it is in the CALL. Section 1 and
    #   section 2 are the same object, the same class, the same two equations,
    #   one `run()` apart. All that differs is the string: "design" selects
    #   {A_t, S_t}, "offdesign" selects {N, W}. engforge re-selects the unknown
    #   set at problem-build time, so the partition is a run-time argument,
    #   never a second subclass with the arrows reversed.
    #
    # * Untagged fields are the specification. In design mode, N, W and theta
    #   are not "fixed values" in any special sense -- they are simply fields
    #   that no active `declare_var` claims, so they sit there and define the
    #   duty. Flip the tag and the same field becomes the answer. One field,
    #   two roles, decided by a word.
    #
    # * This is what makes `combos` more than bookkeeping. The tag is the thing
    #   that makes the question answerable: section 4 is the same model with
    #   both tags live, and it happily reports success on a four-unknown,
    #   two-equation problem. The solver checks that the residuals are zero.
    #   Only you can check that the question had one answer.
    #
    # * Scaling up, the same two lines grow with you. Add `combos="design"` to
    #   a bleed fraction and it joins the sizing partition; add an off-design
    #   limit with `Solver.con_ineq(..., combos="offdesign")` and it is only
    #   enforced where it is physical. Real decks also carry "windmill",
    #   "starting", "max-climb" -- each one a tag over the SAME residuals.
    #
    # * `run(combos=...)` takes fnmatch wildcards, which is how you select a
    #   family of operating modes at once -- and it is sharp in both
    #   directions. Verified on this model:
    #       combos="offdes*"   -> unknowns ['N', 'W']            (what you meant)
    #       combos="*design"   -> unknowns ['A_t','N','S_t','W'] (section 4's trap,
    #                             because "offdesign" also ends in "design")
    #   Name your combos so a glob cannot accidentally union two partitions.
    #   "size" / "operate" would have been the safer pair of words here.
