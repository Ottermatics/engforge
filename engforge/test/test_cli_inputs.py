"""Contract tests for JSON input sets (engforge/inputs.py loaders) and the
Analysis.cli() entry point.
"""

import json
import os
import tempfile
import unittest

from engforge import System, Component, forge, field, Slot
from engforge.analysis import Analysis
from engforge.inputs import (
    load_input_sets,
    parse_cli_value,
    parse_set_overrides,
)
from engforge.properties import system_property


@forge
class Inner(Component):
    y: float = field(default=1.0)


@forge
class CliSys(System):
    a: float = field(default=2.0)
    comp: Inner = Slot.define(Inner)

    @system_property
    def s(self) -> float:
        return self.a + 10 * self.comp.y


def mk_analysis():
    return Analysis(system=CliSys(comp=Inner()), show_plots=False)


def write_json(doc) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fp:
        json.dump(doc, fp)
    return path


class TestLoadInputSets(unittest.TestCase):
    def test_flat_document(self):
        path = write_json({"a": [1, 2], "comp.y": 3.0})
        sets = load_input_sets(path)
        self.assertEqual(sets["inputs"], {"a": [1, 2], "comp.y": 3.0})
        self.assertEqual(sets["settings"], {})
        self.assertIsNone(sets["sequence"])

    def test_sectioned_document(self):
        path = write_json(
            {
                "inputs": {"a": [1, 2]},
                "settings": {"save_mode": "sparse"},
                "sequence": [{"comp.y": 5.0}],
            }
        )
        sets = load_input_sets(path)
        self.assertEqual(sets["inputs"], {"a": [1, 2]})
        self.assertEqual(sets["settings"], {"save_mode": "sparse"})
        self.assertEqual(sets["sequence"], [{"comp.y": 5.0}])

    def test_unknown_section_fails_loudly(self):
        path = write_json({"inputs": {"a": 1}, "setings": {}})  # typo'd section
        with self.assertRaises(AssertionError):
            load_input_sets(path)

    def test_group_rows_ride_as_lists(self):
        path = write_json({"a,comp.y": [[1, 2], [3, 4]]})
        sets = load_input_sets(path)
        self.assertEqual(sets["inputs"]["a,comp.y"], [[1, 2], [3, 4]])


class TestCliValueParsing(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_cli_value("2"), 2)
        self.assertEqual(parse_cli_value("2.5"), 2.5)
        self.assertEqual(parse_cli_value("[1,2]"), [1, 2])
        self.assertEqual(parse_cli_value("true"), True)
        self.assertEqual(parse_cli_value("hello"), "hello")  # bare string

    def test_set_overrides(self):
        out = parse_set_overrides(["a=5", "comp.y=[1,2]", "mode=fast"])
        self.assertEqual(out, {"a": 5, "comp.y": [1, 2], "mode": "fast"})

    def test_set_requires_equals(self):
        with self.assertRaises(AssertionError):
            parse_set_overrides(["a"])


class TestAnalysisCli(unittest.TestCase):
    def test_json_run(self):
        an = mk_analysis()
        path = write_json({"a": [1.0, 2.0]})
        out = an.cli(["--input-json", path, "--no-plots"])
        self.assertEqual(len(out["input_sets"]), 2)
        self.assertEqual(out["input_sets"][0]["a"], 1.0)

    def test_set_overrides_file(self):
        an = mk_analysis()
        path = write_json({"a": [1.0, 2.0]})
        out = an.cli(["--input-json", path, "--set", "a=[3.0]", "--no-plots"])
        self.assertEqual(len(out["input_sets"]), 1)
        self.assertEqual(out["input_sets"][0]["a"], 3.0)

    def test_set_only_no_file(self):
        an = mk_analysis()
        out = an.cli(["--set", "a=7.0", "--no-plots"])
        self.assertEqual(out["input_sets"][0]["a"], 7.0)
        # the run RECORDS a=7 then reverts the live system (default revert_last)
        df = an.system.last_context.result.dataframe
        self.assertTrue((df["a"] == 7.0).all())
        self.assertEqual(an.system.a, 2.0)

    def test_group_key_via_json(self):
        an = mk_analysis()
        path = write_json({"a,comp.y": [[1.0, 2.0], [3.0, 4.0]]})
        out = an.cli(["--input-json", path, "--no-plots"])
        ins = out["input_sets"]
        self.assertEqual(len(ins), 2)  # zipped rows, not a 2x2 product
        self.assertEqual(ins[0]["a"], 1.0)
        self.assertEqual(ins[0]["comp.y"], 2.0)
        self.assertEqual(ins[1]["a"], 3.0)

    def test_sequence_section(self):
        an = mk_analysis()
        path = write_json(
            {"inputs": {"a": [1.0, 2.0]}, "sequence": [{"comp.y": 5.0}]}
        )
        out = an.cli(["--input-json", path, "--no-plots"])
        self.assertEqual(len(out["input_sets"]), 2)
        df = an.system.last_context.result.dataframe
        self.assertTrue((df["comp.y"] == 5.0).all())

    def test_save_mode_flag(self):
        an = mk_analysis()
        out = an.cli(["--set", "a=[1.0,2.0]", "--save-mode", "sparse", "--no-plots"])
        res = an.system.last_context.result
        cases = [c for c in res.cases if c.rows]
        self.assertEqual(len(cases), 2)

    def test_list_inputs(self):
        an = mk_analysis()
        out = an.cli(["--list-inputs"])
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
