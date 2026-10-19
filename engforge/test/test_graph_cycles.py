"""The owned reference graph detects update cycles. A feed-forward system has
none and yields topological layers; a system whose properties read each other
is flagged as a strongly-connected component, and run(cycle_check='error')
raises on it. Report-only otherwise (no solve behavior change)."""

import unittest

from engforge import System, Component, forge, field, Slot
from engforge.properties import system_property
from engforge.problem_context import ProblemExec, IllegalArgument
from engforge.problem_graph import ProblemGraph


@forge
class Leaf(Component):
    y: float = field(default=1.0)

    @system_property
    def twoy(self) -> float:
        return 2 * self.y


@forge
class FeedForward(System):
    a: float = field(default=1.0)
    comp: Leaf = Slot.define(Leaf)

    @system_property
    def s(self) -> float:
        return self.a + self.comp.twoy


@forge
class Cyclic(System):
    k: float = field(default=1.0)

    @system_property
    def p(self) -> float:
        return self.q + self.k

    @system_property
    def q(self) -> float:
        return self.p - 1.0


class TestProblemGraph(unittest.TestCase):
    def test_feed_forward_no_cycles(self):
        g = ProblemGraph(FeedForward(comp=Leaf()))
        self.assertEqual(g.cycles(), [])
        # dependency ordering: comp.twoy before s
        layers = g.layers()
        flat = [n for layer in layers for n in layer]
        self.assertLess(flat.index("comp.twoy"), flat.index("s"))

    def test_cycle_detected(self):
        g = ProblemGraph(Cyclic())
        cyc = g.cycles()
        self.assertEqual(len(cyc), 1)
        self.assertEqual(set(cyc[0]), {"p", "q"})
        self.assertIsNone(g.layers())  # no valid order through a cycle

    def test_problem_graph_accessor(self):
        s = FeedForward(comp=Leaf())
        with ProblemExec(s, {}) as px:
            self.assertEqual(px.graph.cycles(), [])
            self.assertIn("nodes", px.graph.report())

    def test_cycle_check_error_raises(self):
        with self.assertRaises(IllegalArgument):
            Cyclic().run(cycle_check="error", revert_last=False)

    def test_cycle_check_off_by_default(self):
        # default: report-only, a cyclic model still runs (no raise)
        Cyclic().run(revert_last=False)


if __name__ == "__main__":
    unittest.main()
