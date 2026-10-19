"""Per-root Problem scaffolding: a context carries an explicit parent chain
(parent / is_root / root), and session resolution for the shared root_possible
names follows that chain to the context's own root rather than the process
global. This is the structural step toward multi-root coexistence.
"""

import unittest

from engforge import System, forge, field
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.problem_context import ProblemExec


@forge
class RootSys(System):
    x: float = field(default=1.0)

    @system_property
    def y(self) -> float:
        return 2 * self.x


class TestPerRootResolution(unittest.TestCase):
    def test_root_identity(self):
        s = RootSys()
        with ProblemExec(s, {}) as px:
            self.assertTrue(px.is_root)
            self.assertIsNone(px.parent)
            self.assertIs(px.root, px)

    def test_interior_parent_chain(self):
        s = RootSys()
        with ProblemExec(s, {}) as root:
            interior = ProblemExec(s, {}, level_name="inner")
            self.assertFalse(interior.is_root)
            self.assertIs(interior.parent, root)
            self.assertIs(interior.root, root)

    def test_interior_resolves_shared_state_from_its_root(self):
        """an interior context resolves the shared root_possible names (e.g.
        _slv_kw) through its OWN root via the parent chain, not a global."""
        s = RootSys()
        with ProblemExec(s, {}) as root:
            interior = ProblemExec(s, {}, level_name="inner")
            # sesh resolves to the owning root
            self.assertIs(interior.sesh, root.sesh)
            # a shared private name routes to the root's value
            self.assertIs(interior._slv_kw, root._slv_kw)


class TestMultiRoot(unittest.TestCase):
    def test_two_independent_roots_coexist(self):
        """root=True starts an independent root even while another is active;
        they nest LIFO and each resolves to itself."""
        a, b = RootSys(name="A", x=1.0), RootSys(name="B", x=5.0)
        with ProblemExec(a, {}, root=True) as pa:
            self.assertTrue(pa.is_root)
            self.assertIs(pa.system, a)
            with ProblemExec(b, {}, root=True) as pb:
                self.assertTrue(pb.is_root)
                self.assertIs(pb.system, b)  # different system allowed
                self.assertIsNot(pa, pb)
                self.assertIs(pa.root, pa)
                self.assertIs(pb.root, pb)
                self.assertIn(pa, ProblemExec._root_stack)  # A suspended under B
                self.assertIs(ProblemExec.session, pb)  # B is active
            # B released -> A restored (LIFO)
            self.assertIs(ProblemExec.session, pa)
        self.assertFalse(hasattr(ProblemExec, "session"))  # all released

    def test_root_problems_dict_no_eviction(self):
        """two roots register under distinct ids (no problems_dict[True] clobber)."""
        a, b = RootSys(name="A"), RootSys(name="B")
        with ProblemExec(a, {}, root=True) as pa:
            with ProblemExec(b, {}, root=True) as pb:
                self.assertNotEqual(pa._problem_id, pb._problem_id)


@forge
class DualScope(System):
    A: float = field(default=1.0)
    N: float = field(default=1.0)

    hw = Solver.declare_var("A", combos="design")
    op = Solver.declare_var("N", combos="offdesign")
    bal = Solver.constraint_equality("resid", 0, combos="design,offdesign")

    @system_property
    def resid(self) -> float:
        return self.A * self.N - 2.0


class TestChildRescope(unittest.TestCase):
    def test_nested_root_rescopes_same_system(self):
        """a nested root=True on the same system with a different combo derives
        its OWN scope (per-child re-scoping), and the outer scope is restored
        when it exits."""
        s = DualScope()
        with ProblemExec(s, {"combos": "design"}) as outer:
            self.assertEqual(set(outer.sesh.Xref), {"A"})
            with ProblemExec(s, {"combos": "offdesign"}, root=True) as inner:
                self.assertTrue(inner.is_root)
                self.assertEqual(set(inner.sesh.Xref), {"N"})  # different unknown
            self.assertEqual(set(outer.sesh.Xref), {"A"})  # restored


if __name__ == "__main__":
    unittest.main()
