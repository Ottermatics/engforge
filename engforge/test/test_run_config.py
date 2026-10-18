"""RunConfig: the serializable optimizer-settings object, its dict round-trip,
and that its settings actually reach the solver on a steady-state run."""

import unittest

from engforge import System, forge, field
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.run_config import RunConfig


class TestRunConfig(unittest.TestCase):
    def test_dict_roundtrip(self):
        rc = RunConfig(method="SLSQP", tol=1e-9, options={"maxiter": 500})
        d = rc.to_dict()
        self.assertEqual(d, {"method": "SLSQP", "tol": 1e-9, "options": {"maxiter": 500}})
        self.assertEqual(RunConfig.from_dict(d), rc)

    def test_empty_is_noop(self):
        self.assertEqual(RunConfig().minimizer_kwargs(), {})
        self.assertEqual(RunConfig().to_dict(), {})

    def test_from_dict_passthrough_and_none(self):
        rc = RunConfig(method="SLSQP")
        self.assertIs(RunConfig.from_dict(rc), rc)
        self.assertEqual(RunConfig.from_dict(None), RunConfig())

    def test_unknown_key_rejected(self):
        with self.assertRaises(AssertionError):
            RunConfig.from_dict({"methd": "SLSQP"})  # typo

    def test_reaches_solver(self):
        captured = {}

        @forge
        class S(System):
            x: float = field(default=1.0)
            v = Solver.declare_var("x")
            v.add_var_constraint(0.0, "min")
            eq = Solver.constraint_equality("resid", 0)

            @system_property
            def resid(self) -> float:
                return self.x - 2.0

        import engforge.solver_utils as su

        orig = su.sciopt.minimize

        def spy(fun, x0, **kw):
            captured.update(kw)
            return orig(fun, x0, **kw)

        su.sciopt.minimize = spy
        try:
            S().run(run_config={"method": "SLSQP", "options": {"maxiter": 123}},
                    revert_last=False)
        finally:
            su.sciopt.minimize = orig

        self.assertEqual(captured.get("method"), "SLSQP")
        self.assertEqual(captured.get("options", {}).get("maxiter"), 123)


@forge
class SweepSys(System):
    x: float = field(default=1.0)

    @system_property
    def twox(self) -> float:
        return 2 * self.x


class TestStructuredRunInput(unittest.TestCase):
    def test_structured_inputs_bucket(self):
        res = SweepSys().run(inputs={"x": [1.0, 2.0, 3.0]})
        self.assertEqual(len(res.cases), 3)
        self.assertEqual(list(res.dataframe["x"]), [1.0, 2.0, 3.0])

    def test_shorthand_still_works(self):
        res = SweepSys().run(x=[1.0, 2.0])
        self.assertEqual(len(res.cases), 2)

    def test_structured_equals_shorthand(self):
        a = SweepSys().run(inputs={"x": [1.0, 2.0]}, config={"method": "SLSQP"})
        b = SweepSys().run(x=[1.0, 2.0], run_config={"method": "SLSQP"})
        self.assertEqual(list(a.dataframe["x"]), list(b.dataframe["x"]))

    def test_sequence_bucket(self):
        res = SweepSys().run(inputs={"x": [1.0, 2.0]}, sequence=[{"x": 9.0}])
        self.assertTrue((res.dataframe["x"] == 9.0).all())


if __name__ == "__main__":
    unittest.main()
