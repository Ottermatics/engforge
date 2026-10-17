"""Result / Case — explicit ownership of run results and tabulation rows.

Tabulation moves OUT of the object interface: rows are recorded by an explicit
`Case.record(...)` call targeting the active case, instead of accumulating as a
side effect in a dict on the context singleton (where transient cases keyed by
TIME collided/overwrote each other). A `Result` owns the cases of one run and
renders the combined dataframe; each `Case` renders its own.

Compatibility: `Result` is a Mapping so `result["output"][i]` /
`result["input_sets"][i]` keep working (the pinned run() contract).
"""

from collections.abc import Mapping

import pandas


class Case:
    """one point of the input matrix: its inputs, rows, and solver output"""

    __slots__ = ("inputs", "rows", "output", "_index")

    def __init__(self, inputs: dict = None):
        self.inputs = dict(inputs) if inputs else {}
        self.rows = {}
        self.output = None
        self._index = 0

    def record(self, row: dict, index=None, force=False, changed=True) -> bool:
        """record one output-state row. The dedup policy (moved verbatim from
        the context's save_data): save when forced, when first, or when the
        system reports change. Returns True if the row was stored."""
        if not (force or not self.rows or changed):
            return False
        if index is None:
            index = self._index
            self._index += 1
        row = dict(row)
        row["index"] = index
        self.rows[index] = row
        return True

    @property
    def numeric_rows(self) -> list:
        keep = lambda kv: not isinstance(kv[1], (list, dict, tuple))
        return [
            dict(filter(keep, r.items()))
            for _, r in sorted(self.rows.items(), key=lambda kv: kv[0])
        ]

    @property
    def dataframe(self) -> "pandas.DataFrame":
        return pandas.DataFrame(self.numeric_rows)

    def __repr__(self):
        return f"Case({len(self.rows)} rows, inputs={list(self.inputs)})"


class Result(Mapping):
    """the cases of one run; Mapping-compatible with the legacy
    {"output": {i: ...}, "input_sets": {i: ...}} return shape"""

    def __init__(self, system=None):
        self.system = system
        self.cases: list = []

    def new_case(self, inputs: dict = None) -> Case:
        case = Case(inputs)
        self.cases.append(case)
        return case

    # --- dataframe over all cases --------------------------------------------
    @property
    def numeric_rows(self) -> list:
        out = []
        for c in self.cases:
            out.extend(c.numeric_rows)
        return out

    @property
    def dataframe(self) -> "pandas.DataFrame":
        df = pandas.DataFrame(self.numeric_rows)
        if self.system is not None:
            try:
                self.system.format_columns(df)
            except Exception:
                pass
        return df

    # --- legacy mapping contract ----------------------------------------------
    _keys = ("output", "input_sets")

    def __getitem__(self, key):
        if key == "output":
            return {i: c.output for i, c in enumerate(self.cases)}
        if key == "input_sets":
            return {i: c.inputs for i, c in enumerate(self.cases)}
        raise KeyError(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"Result({len(self.cases)} cases)"
