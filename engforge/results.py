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

_MISSING = object()  # sentinel for sparse diffs


class Case:
    """one point of the input matrix: its inputs, rows, and solver output"""

    __slots__ = ("inputs", "rows", "output", "_index", "_last_dense")

    def __init__(self, inputs: dict = None):
        self.inputs = dict(inputs) if inputs else {}
        self.rows = {}
        self.output = None
        self._index = 0
        self._last_dense = None

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

    def record_sparse(self, row: dict, index=None) -> bool:
        """record only the DELTA vs this case's prior recorded state — the
        sparse-storage mode: unchanged values are omitted from the stored row
        and reconstructed (forward-filled) when the dataframe is rendered."""
        prior = getattr(self, "_last_dense", None)
        if prior is None:
            delta = dict(row)  # first row is the full baseline
        else:
            delta = {}
            for k, v in row.items():
                pv = prior.get(k, _MISSING)
                try:
                    same = v == pv
                except Exception:
                    same = v is pv
                if not same:
                    delta[k] = v
        self._last_dense = dict(row)
        if not delta:
            return False
        return self.record(delta, index=index, force=True)

    @property
    def numeric_rows(self) -> list:
        keep = lambda kv: not isinstance(kv[1], (list, dict, tuple))
        return [
            dict(filter(keep, r.items()))
            for _, r in sorted(self.rows.items(), key=lambda kv: kv[0])
        ]

    @property
    def dense_rows(self) -> list:
        """numeric rows with sparse deltas forward-filled to dense rows"""
        out = []
        acc = {}
        for r in self.numeric_rows:
            acc.update(r)
            out.append(dict(acc))
        return out

    @property
    def dense_dataframe(self) -> "pandas.DataFrame":
        return pandas.DataFrame(self.dense_rows)

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
        fmt = getattr(self.system, "format_columns", None)
        if callable(fmt) and not df.empty:
            # no bare `except: pass` here — a broken formatter must surface
            # rather than silently yielding unformatted columns
            fmt(df)
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
