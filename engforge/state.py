"""SystemState — the explicit, typed snapshot of a system's mutable state.

Promotes the implicit dicts that previously lived scattered across the problem
context (`record_state`, `x_start`, `temp_state`) into one structure with four
operations: capture / apply / updated / copy. Episodes (a transient simulation,
a sweep point, a solver attempt) bracket themselves with a SystemState instead
of mutate-then-carefully-restore logic spread across the context.

This is the staging of the 1.0 engine's flat state vector (docs/ENGFORGE_1.0.md
Part VI vectors.py): the storage later becomes a flat array without changing
callers.
"""

from engforge.system_reference import Ref


class SystemState:
    """A copyable snapshot of every settable reference in a problem context,
    plus the episode clock. Values may be numeric, strings, or component
    instances (slot bindings)."""

    __slots__ = ("values", "time")

    def __init__(self, values: dict = None, time: float = 0.0):
        self.values = dict(values) if values else {}
        self.time = time

    # --- construction --------------------------------------------------------
    @classmethod
    def capture(cls, context) -> "SystemState":
        """snapshot the current system state through the context's refs
        (the typed successor of ``ProblemExec.record_state``)"""
        sesh = context.sesh
        refs = {k: v for k, v in sesh.all_comps_and_vars.items()}
        values = Ref.refset_get(refs, sys=sesh.system, prob=context)
        return cls(values=values, time=getattr(sesh, "_time", 0.0))

    # --- application ---------------------------------------------------------
    def apply(self, context, restore_time: bool = False):
        """write this state back through the context's refs (the typed
        successor of ``revert_to_start``/``activate_temp_state``).

        :param restore_time: also restore the session/system clock — used by
            EPISODE brackets (transients); solver revert paths must NOT touch
            time (integration advances it between nested context exits).
        """
        sesh = context.sesh
        if self.values:
            Ref.refset_input(
                sesh.all_comps_and_vars, self.values, fail=False, scope="state"
            )
        if restore_time:
            sesh._time = self.time
            try:
                sesh.system.time = self.time
            except Exception:
                pass
        return self

    # --- derivation ----------------------------------------------------------
    def updated(self, point: dict) -> "SystemState":
        """a NEW state = this state overlaid with an input-matrix point
        (plain / dotted / component values)"""
        new = self.copy()
        if point:
            new.values.update(point)
        return new

    def copy(self) -> "SystemState":
        return SystemState(values=self.values, time=self.time)

    def diff(self, other: "SystemState") -> dict:
        """{key: (self, other)} for values that differ (inspection/debug)"""
        out = {}
        keys = set(self.values) | set(other.values)
        for k in keys:
            a, b = self.values.get(k), other.values.get(k)
            try:
                same = a == b
            except Exception:
                same = a is b
            if not same:
                out[k] = (a, b)
        if self.time != other.time:
            out["<time>"] = (self.time, other.time)
        return out

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"SystemState({len(self.values)} refs, t={self.time})"
