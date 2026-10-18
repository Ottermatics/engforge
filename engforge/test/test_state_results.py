"""Contract tests for the explicit state/result structures:
SystemState (engforge/state.py) and Result/Case (engforge/results.py).
"""

import unittest

from engforge import System, Component, forge, field, Slot, Solver
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


class TestSparseRecording(unittest.TestCase):
    def test_record_sparse_deltas(self):
        c = Case()
        self.assertTrue(c.record_sparse({"a": 1.0, "b": 2.0}))  # baseline = full
        self.assertTrue(c.record_sparse({"a": 1.0, "b": 3.0}))  # only b changed
        self.assertFalse(c.record_sparse({"a": 1.0, "b": 3.0}))  # nothing changed
        rows = c.numeric_rows
        self.assertEqual(len(rows), 2)
        self.assertIn("a", rows[0])
        self.assertNotIn("a", rows[1])  # unchanged value omitted from delta
        self.assertEqual(rows[1]["b"], 3.0)

    def test_dense_reconstruction(self):
        c = Case()
        c.record_sparse({"a": 1.0, "b": 2.0})
        c.record_sparse({"a": 1.0, "b": 3.0})
        c.record_sparse({"a": 4.0, "b": 3.0})
        dense = c.dense_rows
        self.assertEqual(len(dense), 3)
        self.assertEqual(dense[1]["a"], 1.0)  # forward-filled
        self.assertEqual(dense[1]["b"], 3.0)
        self.assertEqual(dense[2]["a"], 4.0)
        self.assertEqual(dense[2]["b"], 3.0)

    def test_sparse_run_matches_dense_run(self):
        """a sparse run's reconstructed frame equals the 'all' frame on the
        same input matrix, and its stored rows carry strictly fewer values"""
        import pandas

        s1 = mk()
        s1.run(a=[1.0, 2.0], revert_last=False, revert_every=False)
        df_all = s1.last_context.result.dataframe

        s2 = mk()
        s2.run(a=[1.0, 2.0], save_mode="sparse", revert_last=False, revert_every=False)
        res = s2.last_context.result
        cases = [c for c in res.cases if c.rows]
        dense = pandas.concat(
            [c.dense_dataframe for c in cases], ignore_index=True
        )

        import numpy

        # compare value columns; 'name' is per-instance identity, not state
        common = [
            k
            for k in df_all.columns
            if k in dense.columns
            and k != "index"
            and numpy.issubdtype(df_all[k].dtype, numpy.number)
        ]
        self.assertTrue(common)  # same ref set — 'sparse' records the 'all' refs
        for k in common:
            self.assertEqual(list(df_all[k]), list(dense[k]), msg=k)

    def test_sparse_rows_shrink_in_simulation(self):
        """within one case, later sparse rows store fewer cells than the
        baseline row when most values are static"""
        c = Case()
        c.record_sparse({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        c.record_sparse({"a": 1.0, "b": 2.0, "c": 3.0, "d": 5.0})
        rows = sorted(c.rows.items())
        base = rows[0][1]
        delta = rows[1][1]
        self.assertGreater(len(base), len(delta))


class TestSimulateResult(unittest.TestCase):
    """simulate() evaluates on a COPY, so its context never attaches to the
    original system — the Result is handed back explicitly instead"""

    def test_last_result_reaches_episode_cases(self):
        from engforge._testing_components import SpringMass

        s = SpringMass()
        df = s.simulate(dt=0.01, endtime=0.2)
        res = s.last_result
        self.assertIsNotNone(res)  # was None before: rows were unreachable
        cases = [c for c in res.cases if c.rows]
        self.assertEqual(len(cases), 1)  # one episode, one case
        self.assertEqual(len(cases[0].rows), len(df))

    def test_sparse_episode_stores_fewer_cells(self):
        from engforge._testing_components import SpringMass

        counts = {}
        shapes = {}
        for mode in ("all", "sparse"):
            s = SpringMass()
            df = s.simulate(dt=0.01, endtime=0.5, save_mode=mode)
            case = [c for c in s.last_result.cases if c.rows][0]
            counts[mode] = sum(len(r) for r in case.rows.values())
            shapes[mode] = df.shape
            if mode == "sparse":
                # deltas reconstruct to the same dense shape
                self.assertEqual(case.dense_dataframe.shape[0], len(case.rows))
        self.assertLess(counts["sparse"], counts["all"])
        self.assertEqual(shapes["sparse"][0], shapes["all"][0])


@forge
class SolvedSys(System):
    """has a real solver variable + objective, so save_mode='prob' has
    something to narrow down to"""

    a: float = field(default=2.0)
    w: float = field(default=1.0)
    comp: Inner = Slot.define(Inner)

    var_w = Solver.declare_var("w", combos="w")
    goal = Solver.objective("err", combos="w", kind="min")

    @system_property
    def err(self) -> float:
        return (self.w - self.a) ** 2

    @system_property
    def s(self) -> float:
        return self.a + 10 * self.comp.y


class TestSaveModes(unittest.TestCase):
    """all five modes must WORK and differ in what they store.
    ('nums' passed the nested system_references dict to refset_get and raised
    AttributeError for every system — it was dead until fixed.)"""

    def run_mode(self, mode, solve=True):
        s = SolvedSys(comp=Inner())
        kw = dict(combos="w", slv_vars="*") if solve else {}
        s.run(
            a=[1.0, 2.0],
            save_mode=mode,
            revert_last=False,
            revert_every=False,
            **kw,
        )
        return s.last_result.dataframe

    def test_every_mode_runs(self):
        for mode in ("vars", "nums", "all", "prob", "sparse"):
            with self.subTest(mode=mode):
                df = self.run_mode(mode)
                self.assertEqual(len(df), 2, msg=mode)

    def test_prob_records_nothing_without_an_active_solver(self):
        """'prob' is scoped to the variables the SOLVER RUN uses: with no combo
        selected there are no problem vars, so there is nothing to record"""
        df = self.run_mode("prob", solve=False)
        self.assertEqual(list(df.columns), ["index"])

    def test_nums_excludes_non_numeric(self):
        df = self.run_mode("nums")
        self.assertNotIn("name", df.columns)  # str field dropped
        self.assertIn("a", df.columns)

    def test_prob_narrower_than_all(self):
        df_prob = self.run_mode("prob")
        df_all = self.run_mode("all")
        self.assertLess(len(df_prob.columns), len(df_all.columns))
        # the solver variable and its objective are what 'prob' is for
        self.assertIn("w", df_prob.columns)

    def test_all_includes_properties(self):
        df = self.run_mode("all")
        for col in ("a", "s", "err"):
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
