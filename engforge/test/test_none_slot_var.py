"""A solver variable declared into an optional slot that is left empty (None)
must be handled by its `on_missing` policy at problem establishment — "skip"
drops it silently, "warn" logs once per problem, "error" raises — instead of
being silently dropped with no feedback.
"""

import unittest

from engforge import System, Component, forge, field, Slot
from engforge.attr_solver import Solver
from engforge.problem_context import ProblemExec, IllegalArgument
from engforge.properties import system_property


@forge
class Leg(Component):
    x: float = field(default=1.0)

    @system_property
    def twox(self) -> float:
        return 2 * self.x


def make(policy):
    @forge
    class Host(System):
        base: float = field(default=1.0)
        leg: Leg = Slot.define(Leg, none_ok=True)

        v = Solver.declare_var("leg.x", combos="s", on_missing=policy)
        v.add_var_constraint(0.0, "min", combos="s")

        @system_property
        def resid(self) -> float:
            return self.base

    return Host


class TestNoneSlotVar(unittest.TestCase):
    def test_skip_is_silent_and_runs(self):
        Host = make("skip")
        h = Host(leg=None)  # empty slot
        h.run(combos="s", revert_last=False)  # must not raise
        # var dropped: not in the problem
        with ProblemExec(h, {"combos": "s"}) as px:
            self.assertNotIn("leg.x", px.sesh.all_problem_vars)

    def test_error_raises_at_establishment(self):
        Host = make("error")
        h = Host(leg=None)
        with self.assertRaises(IllegalArgument):
            h.run(combos="s", revert_last=False)

    def test_filled_slot_solves_normally(self):
        Host = make("warn")
        h = Host(leg=Leg())  # slot present → var is live, no missing warning
        with ProblemExec(h, {"combos": "s"}) as px:
            self.assertIn("leg.x", px.sesh.all_problem_vars)
            self.assertEqual(px.sesh._sys_refs.get("missing", {}), {})


if __name__ == "__main__":
    unittest.main()
