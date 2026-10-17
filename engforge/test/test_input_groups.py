"""Contract tests for the run input layer (engforge 1.0 candidate):
default runs, custom inputs, and GROUPED matrix inputs (comma-keys) that vary
jointly — a partial factorial instead of the complete outer product.
See docs/ENGFORGE_1.0.md Part IV.
"""

import unittest

from engforge import System, Component, forge, field, Slot
from engforge.properties import system_property


@forge
class Inner(Component):
    y: float = field(default=0.0)


@forge
class Toy(System):
    a: float = field(default=0.0)
    b: float = field(default=0.0)
    c: float = field(default=0.0)
    comp: Inner = Slot.define(Inner)

    @system_property
    def s(self) -> float:
        return self.a + 10 * self.b + 100 * self.c + 1000 * self.comp.y


def mk():
    return Toy(comp=Inner())


RUNKW = dict(revert_last=False, revert_every=False)


class TestDefaults(unittest.TestCase):
    def test_default_run_one_point(self):
        t = mk()
        r = t.run(**RUNKW)
        self.assertEqual(list(r["output"].keys()), [0])
        self.assertEqual(r["input_sets"][0], {})
        self.assertEqual(len(t.dataframe), 1)

    def test_custom_scalar_and_dotted(self):
        t = mk()
        r = t.run(a=2, **{"comp.y": 3}, **RUNKW)
        self.assertEqual(len(r["input_sets"]), 1)
        df = t.dataframe
        self.assertEqual(len(df), 1)
        self.assertEqual(float(df["s"].iloc[0]), 2 + 3000)

    def test_backcompat_product(self):
        t = mk()
        r = t.run(a=[1, 2], b=[1, 2, 3], **RUNKW)
        self.assertEqual(len(r["input_sets"]), 6)
        self.assertEqual(len(t.dataframe), 6)
        pts = {(p["a"], p["b"]) for p in r["input_sets"].values()}
        self.assertEqual(pts, {(i, j) for i in (1, 2) for j in (1, 2, 3)})


class TestGroups(unittest.TestCase):
    def test_group_zips(self):
        t = mk()
        r = t.run(**{"a,b": [(1, 2), (3, 4)]}, **RUNKW)
        pts = sorted((p["a"], p["b"]) for p in r["input_sets"].values())
        self.assertEqual(pts, [(1, 2), (3, 4)])  # 2 rows, NOT 4

    def test_group_crosses_list(self):
        t = mk()
        r = t.run(**{"a,b": [(1, 2), (3, 4)], "c": [10, 20]}, **RUNKW)
        pts = sorted((p["a"], p["b"], p["c"]) for p in r["input_sets"].values())
        self.assertEqual(
            pts, [(1, 2, 10), (1, 2, 20), (3, 4, 10), (3, 4, 20)]
        )  # 4, not 8

    def test_group_times_group(self):
        t = mk()
        r = t.run(
            **{"a,b": [(1, 2), (3, 4)], "c,comp.y": [(1, 0), (2, 1), (3, 2)]},
            **RUNKW,
        )
        self.assertEqual(len(r["input_sets"]), 6)  # 2 x 3
        pts = {(p["a"], p["b"], p["c"], p["comp.y"]) for p in r["input_sets"].values()}
        self.assertIn((1, 2, 2, 1), pts)
        self.assertNotIn((1, 4, 1, 0), pts)  # never cross within a group

    def test_group_dotted_subkeys(self):
        t = mk()
        r = t.run(**{"a,comp.y": [(1, 5), (2, 6)]}, **RUNKW)
        df = t.dataframe
        self.assertEqual(sorted(df["comp.y"].tolist()), [5, 6])
        self.assertEqual(sorted(df["a"].tolist()), [1, 2])

    def test_single_bare_row_promoted(self):
        t = mk()
        r = t.run(**{"a,b": (7, 8)}, **RUNKW)
        self.assertEqual(len(r["input_sets"]), 1)
        self.assertEqual((r["input_sets"][0]["a"], r["input_sets"][0]["b"]), (7, 8))


class TestGroupErrors(unittest.TestCase):
    def test_unequal_row_length(self):
        with self.assertRaises(AssertionError):
            mk().run(**{"a,b": [(1, 2), (3,)]}, **RUNKW)

    def test_duplicate_subkey_across_inputs(self):
        with self.assertRaises(AssertionError):
            mk().run(**{"a,b": [(1, 2)], "a": [1]}, **RUNKW)

    def test_bad_subkey_name(self):
        with self.assertRaises(AssertionError):
            mk().run(**{"a,nope": [(1, 2)]}, **RUNKW)

    def test_bad_dotted_subkey(self):
        with self.assertRaises(AssertionError):
            mk().run(**{"a,wrong.y": [(1, 2)]}, **RUNKW)


class TestSequence(unittest.TestCase):
    def test_sequence_rows_cross_and_override(self):
        t = mk()
        r = t.run(
            a=[1, 2], sequence=[{"b": 1}, {"b": 2}, {"a": 9, "b": 3}], **RUNKW
        )
        pts = sorted((p["a"], p["b"]) for p in r["input_sets"].values())
        # 2 (a) x 3 (sequence rows); third row OVERRIDES a
        self.assertEqual(pts, [(1, 1), (1, 2), (2, 1), (2, 2), (9, 3), (9, 3)])

    def test_sequence_now_validated(self):
        with self.assertRaises(AssertionError):
            mk().run(a=[1], sequence=[{"nope": 1}], **RUNKW)
        with self.assertRaises(AssertionError):
            mk().run(a=[1], sequence=[{"b": object()}], **RUNKW)


class TestComponentValues(unittest.TestCase):
    def test_component_in_group(self):
        t = mk()
        r = t.run(**{"comp,a": [(Inner(y=1.0), 10.0), (Inner(y=2.0), 20.0)]}, **RUNKW)
        self.assertEqual(len(r["input_sets"]), 2)
        df = t.dataframe
        self.assertEqual(sorted(df["comp.y"].tolist()), [1.0, 2.0])
        self.assertEqual(sorted(df["a"].tolist()), [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
