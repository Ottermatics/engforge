"""SystemState — a typed, copyable snapshot of a system's settable refs plus
the episode clock. Four operations: capture / apply / updated / copy. Episodes
(a transient, a sweep point, a solver attempt) bracket themselves with one.
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
        """write these values back through the context's refs.

        :param restore_time: also restore the clock (episode brackets); solver
            revert paths leave time alone.

        Keys absent from the ref set are tolerated (warned) rather than fatal:
        a slot-swap sweep can change which refs exist between capture and apply.
        """
        sesh = context.sesh
        if self.values:
            Ref.refset_input(
                sesh.all_comps_and_vars, self.values, fail=False, scope="state"
            )
        if restore_time:
            sesh._time = self.time
        return self

    # --- derivation ----------------------------------------------------------
    def updated(self, point: dict, refs: dict = None) -> "SystemState":
        """a NEW state = this state overlaid with an input point.

        :param refs: optional name->ref mapping; when given, keys in ``point``
            absent from it raise with close-match suggestions.
        """
        new = self.copy()
        if point:
            if refs is not None:
                import difflib

                unknown = [k for k in point if k not in refs]
                if unknown:
                    sugg = {
                        k: difflib.get_close_matches(k, list(refs), n=3)
                        for k in unknown
                    }
                    raise KeyError(
                        f"state| unknown point key(s) {unknown}; did you mean {sugg}?"
                    )
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
