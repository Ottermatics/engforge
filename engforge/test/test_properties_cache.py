"""Contract tests for the property/caching layer — pins four defects found by
the determinism classification pass (all verified by execution before fixing):

1. `stochastic=True` was dead: Configuration.pre_compile (a no-op) shadowed
   TabulationMixin.pre_compile in the MRO, so `_always_save_data` never set.
2. `cached_system_property` crashed on first access (`gname` never set by the
   system_property family's __init__/__call__).
3. The `cached_sys_prop`/`cached_system_prop` aliases pointed at the UNCACHED
   system_property class.
4. `solver_cached` (and friends) crashed on class-level access (no
   `instance is None` guard).
"""

import unittest

from engforge import System, forge, field
from engforge.properties import (
    system_property,
    cached_system_property,
    cached_sys_prop,
    cached_system_prop,
    solver_cached,
)


@forge
class StochasticSys(System):
    a: float = field(default=1.0)

    @system_property(stochastic=True)
    def noisy(self) -> float:
        return self.a  # determinism is irrelevant; the FLAG is under test


@forge
class CachedSys(System):
    a: float = field(default=1.0)

    @cached_system_property
    def heavy(self) -> float:
        CachedSys._evals = getattr(CachedSys, "_evals", 0) + 1
        return self.a * 2

    @solver_cached
    def solver_val(self) -> float:
        return self.a + 1


class TestStochasticFlag(unittest.TestCase):
    def test_pre_compile_reaches_tabulation_hook(self):
        """a stochastic property must set class-wide always-save (the flag's
        one consumer, TabulationMixin.pre_compile, must actually run)"""
        self.assertTrue(StochasticSys._always_save_data)
        # and the instance surface agrees
        s = StochasticSys()
        self.assertTrue(s.always_save_data)

    def test_non_stochastic_class_unaffected(self):
        self.assertFalse(CachedSys._always_save_data)


class TestCachedSystemProperty(unittest.TestCase):
    def test_aliases_point_at_the_cached_class(self):
        self.assertIs(cached_sys_prop, cached_system_property)
        self.assertIs(cached_system_prop, cached_system_property)

    def test_first_access_does_not_crash(self):
        s = CachedSys()
        self.assertEqual(s.heavy, 2.0)  # raised AttributeError: 'gname' before

    def test_cache_hits_until_input_changes(self):
        s = CachedSys()
        CachedSys._evals = 0
        _ = s.heavy
        n_first = CachedSys._evals
        s._anything_changed = False
        _ = s.heavy
        _ = s.heavy
        self.assertEqual(CachedSys._evals, n_first)  # served from cache
        s.a = 5.0  # attrs on_setattr flags the change
        self.assertEqual(s.heavy, 10.0)
        self.assertGreater(CachedSys._evals, n_first)  # recomputed

    def test_class_level_access_returns_descriptor(self):
        self.assertIsInstance(CachedSys.heavy, cached_system_property)
        self.assertIsInstance(CachedSys.solver_val, solver_cached)

    def test_gname_set_by_both_decorator_forms(self):
        self.assertEqual(CachedSys.heavy.gname, "heavy")
        self.assertEqual(StochasticSys.noisy.gname, "noisy")


if __name__ == "__main__":
    unittest.main()
