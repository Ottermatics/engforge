"""Transient control showcase — grouped matrix inputs + explicit Case results.

Demonstrates, with zero new declaration concepts:
  * a PID controller living WITH the plant as a Component (gains = fields);
  * transient response cases (sine / Heaviside step) as Profile COMPONENTS
    swept into a Slot — "everything is a component";
  * tuned gain sets varying JOINTLY via a comma-key GROUP, crossed with the
    profiles: 2 gain-sets x 2 profiles = 4 transients, NOT 8;
  * each case's rows owned by its own Case (engforge/results.py) — no shared
    row store, deterministic across reruns.

Run:  conda run -n pace python examples/transient_control.py
"""

import numpy as np
from engforge import System, Component, forge, field, Slot
from engforge.properties import system_property


# --- response-case profiles: components, so cases are swept like any input ---
@forge
class Profile(Component):
    """base setpoint profile r(t)"""

    def r(self, t) -> float:
        return 0.0


@forge
class SineProfile(Profile):
    amp: float = field(default=1.0)
    freq: float = field(default=0.5)  # Hz

    def r(self, t) -> float:
        return self.amp * np.sin(2 * np.pi * self.freq * t)


@forge
class StepProfile(Profile):
    """Heaviside step"""

    u0: float = field(default=1.0)
    t_on: float = field(default=0.5)

    def r(self, t) -> float:
        return self.u0 if t >= self.t_on else 0.0


# --- PID gains as a component (a tuned SET of gains = one group row) ----------
@forge
class PID(Component):
    kp: float = field(default=1.0)
    ki: float = field(default=0.0)


# --- first-order plant + PI control, states on the loop system ----------------
@forge
class ControlLoop(System):
    """dx/dt = (u_cmd - x)/tau ;  u_cmd = kp*e + ki*I ;  dI/dt = e = r(t) - x"""

    dynamic_state_vars: list = ["x", "I"]
    nonlinear: bool = True  # rebuild matrices each step (profile + gains live)

    tau: float = field(default=0.35)
    x: float = field(default=0.0)
    I: float = field(default=0.0)

    pid: PID = Slot.define(PID)
    profile: Profile = Slot.define(Profile)

    @system_property
    def setpoint_now(self) -> float:
        return float(self.profile.r(self.time))

    @system_property
    def error_now(self) -> float:
        return self.setpoint_now - self.x

    # state-space: Xd = A@X + F   with X = [x, I]
    def create_state_matrix(self, *a, **kw):
        kp, ki = self.pid.kp, self.pid.ki
        return np.array([[-(kp + 1.0) / self.tau, ki / self.tau], [-1.0, 0.0]])

    def create_state_constants(self, **kw):
        return np.array([0.0, 0.0])

    def update_state(self, t, A, X):
        return self.create_state_matrix()

    def update_state_constants(self, t, F, X):
        r = float(self.profile.r(t))
        return np.array([self.pid.kp * r / self.tau, r])


def settling_metrics(df, profile, dt):
    t = df["time"].values if "time" in df.columns else np.arange(len(df)) * dt
    r = np.array([profile.r(ti) for ti in t])
    x = df["x"].values
    err = x - r
    return dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        final_err=float(abs(err[-1])),
    )


if __name__ == "__main__":
    DT, END = 0.01, 5.0

    loop = ControlLoop(pid=PID(), profile=StepProfile())

    # the grouped sweep: gain SETS zipped (2 rows) x profile cases (2) = 4 runs
    res = loop.sim_matrix(
        dt=DT,
        endtime=END,
        **{
            "pid.kp,pid.ki": [(2.0, 1.0), (8.0, 4.0)],      # tuned sets — JOINT
            "profile": [StepProfile(), SineProfile()],       # response cases
        },
    )

    pts = res["input_sets"]
    outs = res["output"]
    print(f"\nran {len(pts)} transients (2 gain-sets x 2 profiles — not 8):\n")
    print(f"{'case':>4} {'kp':>5} {'ki':>5} {'profile':>14} {'rmse':>9} {'final_err':>10}")
    metrics = {}
    for i in sorted(pts):
        p, df = pts[i], outs[i]
        prof = p["profile"]
        m = settling_metrics(df, prof, DT)
        metrics[i] = (p, m)
        print(
            f"{i:>4} {p['pid.kp']:>5} {p['pid.ki']:>5} "
            f"{type(prof).__name__:>14} {m['rmse']:>9.4f} {m['final_err']:>10.4f}"
        )

    # sanity: on the STEP, the stiffer gain set must track better at the end
    # (final error — rmse penalizes the overshoot aggressive gains trade for speed)
    fe = {p["pid.kp"]: m["final_err"]
          for p, m in metrics.values() if isinstance(p["profile"], StepProfile)}
    assert fe[8.0] < fe[2.0], f"stiffer gains should settle closer: {fe}"
    print("\nOK: stiffer gain-set settles closer on the step "
          f"(final_err {fe[8.0]:.4f} < {fe[2.0]:.4f}); gain sets never cross-combined.")
