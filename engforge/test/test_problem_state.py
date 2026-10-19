"""The per-root ProblemState records which component attributes changed while
its root is active (event-driven), and holds the start checkpoint. This is the
explicit, per-tree successor to the tree-walking anything_changed flag.
"""

import unittest

from engforge import System, forge, field
from engforge.properties import system_property
from engforge.problem_context import ProblemExec
from engforge.state import ProblemState, SystemState
from engforge import events


@forge
class StSys(System):
    a: float = field(default=1.0)
    b: float = field(default=2.0)

    @system_property
    def s(self) -> float:
        return self.a + self.b


class TestProblemState(unittest.TestCase):
    def test_records_value_changes(self):
        s = StSys()
        with ProblemExec(s, {}) as px:
            px.state.clear()
            s.a = 9.0
            self.assertTrue(px.state.is_dirty(s, "a"))
            self.assertFalse(px.state.is_dirty(s, "b"))

    def test_checkpoint_is_system_state(self):
        s = StSys()
        with ProblemExec(s, {}) as px:
            self.assertIsInstance(px.state, ProblemState)
            # the start checkpoint is the root's captured SystemState
            self.assertIsInstance(px.state.checkpoint, SystemState)

    def test_listener_removed_after_root_exits(self):
        s = StSys()
        with ProblemExec(s, {}):
            self.assertTrue(events.has_listeners(events.VALUE_CHANGED))
        # the root's ProblemState unsubscribed on exit
        self.assertFalse(events.has_listeners(events.VALUE_CHANGED))

    def test_direct_problemstate_dirty(self):
        ps = ProblemState()
        ps.subscribe()
        try:
            s = StSys()
            s._anything_changed = False
            s.a = 5.0
            self.assertTrue(ps.is_dirty(s, "a"))
            self.assertTrue(ps.is_dirty(s))  # any attr of s
        finally:
            ps.unsubscribe()


if __name__ == "__main__":
    unittest.main()
