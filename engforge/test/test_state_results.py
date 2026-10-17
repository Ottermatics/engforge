"""Contract tests for the explicit state/result structures:
SystemState (engforge/state.py) and Result/Case (engforge/results.py).
"""

import unittest

from engforge import System, Component, forge, field, Slot
from engforge.properties import system_property
from engforge.problem_context import ProblemExec
from engforge.state import SystemState
from engforge.results import Result, Case


@forge
class Inner(Component):
    y: float = field(default=1.0)


@forge
class Sys(System):
    a: float = field(default=2.0)
    comp: Inner = Slot.define(Inner)

    @system_property
    def s(self) -> float:
        return self.a + 10 * self.comp.y


def mk():
    return Sys(comp=Inner())


class TestSystemState(unittest.TestCase):
    def test_capture_apply_roundtrip(self):
        s = mk()
        with ProblemExec(s, {}, level_name="t") as px:
            st = SystemState.capture(px)
            s.a = 99.0
            s.comp.y = 42.0
            st.apply(px)
            self.assertEqual(s.a, 2.0)
            self.assertEqual(s.comp.y, 1.0)

    def test_updated_point_applies_dotted(self):
        s = mk()
        with ProblemExec(s, {}, level_name="t") as px:
            st = SystemState.capture(px).updated({"a": 5.0, "comp.y": 7.0})
            st.apply(px)
            self.assertEqual(s.a, 5.0)
            self.assertEqual(s.comp.y, 7.0)

    def test_time_restore_is_optional(self):
        s = mk()
        with ProblemExec(s, {}, level_name="t") as px:
            st = SystemState.capture(px)
            px.sesh._time = 3.3
            st.apply(px)  # fields only
            self.assertEqual(px.sesh._time, 3.3)
            st.apply(px, restore_time=True)  # episode restore
            self.assertEqual(px.sesh._time, st.time)

    def test_diff(self):
        a = SystemState({"x": 1.0, "y": 2.0}, time=0.0)
        b = SystemState({"x": 1.0, "y": 3.0}, time=1.0)
        d = a.diff(b)
        self.assertIn("y", d)
        self.assertIn("<time>", d)
        self.assertNotIn("x", d)


class TestResultCases(unittest.TestCase):
    def test_run_returns_cases_with_disjoint_rows(self):
        s = mk()
        s.run(a=[1, 2, 3], revert_last=False, revert_every=False)
        res = s.last_context.result
        cases = [c for c in res.cases if c.rows]
        self.assertEqual(len(cases), 3)
        for c in cases:
            self.assertEqual(len(c.rows), 1)  # one row per point, owned
        self.assertEqual(len(s.dataframe), 3)  # combined frame unchanged

    def test_mapping_compat(self):
        s = mk()
        out = s.run(a=[1, 2], revert_last=False, revert_every=False)
        self.assertIn(0, out["output"])
        self.assertEqual(out["input_sets"][0]["a"], 1)
        self.assertEqual(len(out["input_sets"]), 2)

    def test_case_record_dedup(self):
        c = Case({"a": 1})
        self.assertTrue(c.record({"v": 1}))                    # first row
        self.assertFalse(c.record({"v": 1}, changed=False))     # unchanged -> skip
        self.assertTrue(c.record({"v": 2}, changed=True))       # changed
        self.assertTrue(c.record({"v": 2}, changed=False, force=True))  # forced
        self.assertEqual(len(c.rows), 3)


if __name__ == "__main__":
    unittest.main()
