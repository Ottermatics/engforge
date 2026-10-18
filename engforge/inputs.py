"""Pure input-matrix expansion for `run()`/`sim_matrix` — the engforge 1.0 run
input layer (the MAP cycle element; see docs/ENGFORGE_1.0.md Part IV).

Three input forms ride the one kwargs channel:

  * plain keys       ``x=[1,2]``                      -> product factors (as always)
  * GROUP keys       ``"a,b": [(1,2),(3,4)]``         -> comma-joined subkeys whose
                     rows vary JOINTLY (zipped) and cross the other factors —
                     a partial factorial, not a complete outer product
  * ``sequence=[{...},{...}]``                        -> row-dict group, crossed
                     innermost, overriding on key collision (legacy semantics)

This module is deliberately pure (no engforge imports, stdlib only): the
engforge-specific validation (field/slot existence, value types) stays in
``SolveableMixin.parse_run_kwargs``; the CLI/JSON input-set loaders below are
skins over these functions.

JSON input-set files (``load_input_sets``) come in two shapes:

  * flat        ``{"a": [1,2], "comp.y": 3.0, "kp,ki": [[2,1],[8,4]]}``
                — the whole document is the inputs dict
  * sectioned   ``{"inputs": {...}, "settings": {...}, "sequence": [{...}]}``
                — settings are run/solver kwargs (save_mode, combos, ...)
                  and sequence is the innermost override-row list

JSON has no tuples, so group rows arrive as lists-of-lists — exactly what
``normalize_group_rows`` already accepts.
"""

import itertools
import json

__all__ = [
    "is_group_key",
    "split_group_key",
    "normalize_group_rows",
    "expand_point_factors",
    "count_points",
    "load_input_sets",
    "parse_cli_value",
    "parse_set_overrides",
]


def is_group_key(key: str) -> bool:
    """a run-kwarg key containing a comma declares a jointly-varying group"""
    return isinstance(key, str) and "," in key


def split_group_key(key: str) -> list:
    """split ``"a, comp.b"`` into subkeys (each may be dotted); validates shape"""
    subkeys = [k.strip() for k in key.split(",")]
    assert all(subkeys), f"empty subkey in group {key!r}"
    assert len(subkeys) == len(set(subkeys)), f"duplicate subkey in group {key!r}"
    return subkeys


def normalize_group_rows(key: str, subkeys: list, rows) -> list:
    """coerce & check a group's value into a list of tuples of len(subkeys).

    Accepts a list of rows (tuple/list each); a single tuple row is promoted to
    a one-row list for symmetry with scalar->``[scalar]`` coercion.
    """
    if isinstance(rows, tuple) and not any(isinstance(r, (list, tuple)) for r in rows):
        rows = [rows]  # single row given bare
    assert isinstance(rows, list), (
        f"group {key!r} requires a list of rows, got {type(rows).__name__}"
    )
    out = []
    for i, row in enumerate(rows):
        assert isinstance(row, (list, tuple)), (
            f"group {key!r} row {i} must be a tuple/list, got {row!r}"
        )
        assert len(row) == len(subkeys), (
            f"group {key!r} row {i} has {len(row)} values for "
            f"{len(subkeys)} subkeys {subkeys}: {row!r}"
        )
        out.append(tuple(row))
    assert out, f"group {key!r} has no rows"
    return out


def expand_point_factors(singles: dict, groups: list, sequence: list = None):
    """yield merged point dicts: product over factors — singles first (kwargs
    insertion order, matching legacy ordering exactly when no groups exist),
    then groups in declaration order; the last factor varies fastest and
    ``sequence`` rows are the innermost factor, OVERRIDING colliding keys
    (legacy ``_iterate_input_matrix`` semantics).

    :param singles: {key: [values]} — each key one factor (insertion order)
    :param groups:  [(subkeys, rows)] — each group ONE factor of zipped rows
    :param sequence: list of override dicts (or None/[] -> one empty row)
    """
    factors = []
    for k, vals in singles.items():
        factors.append([{k: v} for v in vals])
    for subkeys, rows in groups:
        factors.append([dict(zip(subkeys, row)) for row in rows])

    seq = sequence if sequence else [{}]

    for combo in itertools.product(*factors):
        base = {}
        for d in combo:
            base.update(d)
        for row in seq:
            icur = base.copy()
            if row:
                icur.update(**row)
            yield icur


def count_points(singles: dict, groups: list, sequence: list = None) -> int:
    """number of points :func:`expand_point_factors` will yield"""
    n = 1
    for vals in singles.values():
        n *= len(vals)
    for _, rows in groups:
        n *= len(rows)
    n *= max(len(sequence), 1) if sequence else 1
    return n


# --- JSON / CLI input-set loading --------------------------------------------

#: the recognized top-level sections of a sectioned input-set document
RESERVED_SECTIONS = ("inputs", "settings", "sequence")


def load_input_sets(path: str) -> dict:
    """load a JSON input-set file into ``{"inputs", "settings", "sequence"}``.

    A flat document is promoted to the inputs dict; a sectioned document may
    ONLY contain the reserved sections (an unknown top-level key next to
    ``inputs`` is almost certainly a typo'd section — fail loudly).

    Keys are NOT validated here — field/slot existence and value types are
    checked by ``parse_run_kwargs`` when the point matrix is parsed, so errors
    carry the system's context.
    """
    with open(path) as fp:
        doc = json.load(fp)
    assert isinstance(doc, dict), f"input-set file must be a JSON object: {path}"

    if any(k in doc for k in RESERVED_SECTIONS):
        unknown = [k for k in doc if k not in RESERVED_SECTIONS]
        assert not unknown, (
            f"unknown top-level section(s) {unknown} in {path}; a sectioned "
            f"input-set may only contain {RESERVED_SECTIONS}"
        )
        inputs = doc.get("inputs", {}) or {}
        settings = doc.get("settings", {}) or {}
        sequence = doc.get("sequence", None)
    else:
        inputs, settings, sequence = doc, {}, None

    assert isinstance(inputs, dict), f"'inputs' must be an object in {path}"
    assert isinstance(settings, dict), f"'settings' must be an object in {path}"
    assert sequence is None or (
        isinstance(sequence, list) and all(isinstance(r, dict) for r in sequence)
    ), f"'sequence' must be a list of objects in {path}"

    return {"inputs": inputs, "settings": settings, "sequence": sequence}


def parse_cli_value(text: str):
    """parse one CLI value: JSON first (numbers, lists, booleans, null,
    quoted strings), bare string as the fallback"""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def parse_set_overrides(pairs: list) -> dict:
    """parse repeatable ``--set KEY=VALUE`` arguments into an inputs dict;
    values go through :func:`parse_cli_value` (``--set a=[1,2]`` sweeps)"""
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        key = key.strip()
        assert sep and key, f"--set expects KEY=VALUE, got {pair!r}"
        out[key] = parse_cli_value(value)
    return out
