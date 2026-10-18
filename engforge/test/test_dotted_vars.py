"""A solver variable declared through a dotted slot path
(Solver.declare_var('filt.w')) drives that slot's field during a solve, is
keyed by its dotted path, and leaf declarations behave identically.
"""

import unittest

import numpy as np

from engforge import System, Component, forge, field, Slot
from engforge.attr_solver import Solver
from engforge.properties import system_property


@forge
class _Filt(Component):
    w: float = field(default=1.0)
    k_loss: float = field(default=50.0)

    @system_property
    def dP_filter(self) -> float:
        return self.k_loss * self.w


@forge
class _DottedLinear(System):
    base: float = field(default=400.0)
    filt: _Filt = Slot.define(_Filt)

    # THE unification: the slot var IS the solver unknown, declared by its
    # dotted path on the system.
    flow = Solver.declare_var("filt.w", combos="flow")
    flow.add_var_constraint(0.01, "min", combos="flow")
    eq = Solver.constraint_equality("resid", 0, combos="flow")

    @system_property
    def resid(self) -> float:
        return self.base - self.filt.dP_filter


@forge
class _DottedQuadratic(System):
    filt: _Filt = Slot.define(_Filt)

    flow = Solver.declare_var("filt.w", combos="flow")
    flow.add_var_constraint(0.01, "min", combos="flow")
    eq = Solver.constraint_equality("sum_dP", 0, combos="flow")

    @system_property
    def sum_dP(self) -> float:
        return 400.0 - 0.1 * self.filt.w**2 - self.filt.dP_filter


class TestDottedSolverVar(unittest.TestCase):
    def test_linear_dotted_var_solves(self):
        s = _DottedLinear(filt=_Filt())
        out = s.run(combos="flow", revert_last=False, revert_every=False)
        self.assertAlmostEqual(s.filt.w, 8.0, places=4)  # 50*w == 400
        self.assertAlmostEqual(s.resid, 0.0, places=5)
        o = out["output"][0]
        self.assertTrue(o["success"])
        self.assertIn("filt.w", o["Xans"])  # keyed by the DOTTED path, not 'w'
        self.assertAlmostEqual(o["Xans"]["filt.w"], 8.0, places=4)

    def test_quadratic_dotted_var_solves(self):
        s = _DottedQuadratic(filt=_Filt())
        s.run(combos="flow", revert_last=False, revert_every=False)
        root = [r for r in np.roots([0.1, 50, -400]) if r > 0][0]
        self.assertAlmostEqual(s.filt.w, root, places=3)
        self.assertAlmostEqual(s.sum_dP, 0.0, places=4)

    def test_leaf_declaration_unchanged(self):
        """a leaf declaration on the OWNING component keys identically ('filt.w'
        either way) — the fix must not change this."""

        @forge
        class _LeafFilt(_Filt):
            wvar = Solver.declare_var("w", combos="flow")
            wvar.add_var_constraint(0.01, "min", combos="flow")

        @forge
        class _Sys(System):
            base: float = field(default=400.0)
            filt: _LeafFilt = Slot.define(_LeafFilt)
            eq = Solver.constraint_equality("resid", 0, combos="flow")

            @system_property
            def resid(self) -> float:
                return self.base - self.filt.dP_filter

        s = _Sys(filt=_LeafFilt())
        out = s.run(combos="flow", revert_last=False, revert_every=False)
        self.assertAlmostEqual(s.filt.w, 8.0, places=4)
        self.assertIn("filt.w", out["output"][0]["Xans"])  # composed from prefix


if __name__ == "__main__":
    unittest.main()
