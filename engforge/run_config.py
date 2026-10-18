"""RunConfig — a small serializable container for the settings of a solve.

Carries the optimizer controls that otherwise have no explicit home: the
``scipy.optimize.minimize`` method, tolerance, and options. All fields are
optional and default to empty, so a run with no config uses the solver's
built-in behavior; set a field to opt into specific control.

The object round-trips through plain dicts (``to_dict`` / ``from_dict``), so it
is YAML- and JSON-compatible without any extra dependency. Extend by adding
fields here and surfacing them in ``minimizer_kwargs`` / ``to_dict`` as needed.

    RunConfig(method="SLSQP", tol=1e-9, options={"maxiter": 500})

    # from a YAML/JSON settings block:
    RunConfig.from_dict({"method": "SLSQP", "options": {"maxiter": 500}})
"""

import attrs


@attrs.define
class RunConfig:
    #: scipy.optimize.minimize method (None -> scipy chooses from the problem)
    method: str = None
    #: solver tolerance (None -> scipy default)
    tol: float = None
    #: scipy `options` dict (maxiter, ftol, finite_diff_rel_step, ...)
    options: dict = attrs.field(factory=dict)

    def minimizer_kwargs(self) -> dict:
        """the subset to hand scipy.optimize.minimize; empty keys are omitted
        so unset fields never override scipy's defaults."""
        out = {}
        if self.method is not None:
            out["method"] = self.method
        if self.tol is not None:
            out["tol"] = self.tol
        if self.options:
            out["options"] = dict(self.options)
        return out

    def to_dict(self) -> dict:
        """plain-dict form (YAML/JSON-safe); unset fields are dropped."""
        d = {}
        if self.method is not None:
            d["method"] = self.method
        if self.tol is not None:
            d["tol"] = self.tol
        if self.options:
            d["options"] = dict(self.options)
        return d

    @classmethod
    def from_dict(cls, data) -> "RunConfig":
        """build from a plain dict (or pass through an existing RunConfig /
        None). Unknown keys are rejected so a typo'd setting fails loudly."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        assert isinstance(data, dict), f"run_config must be a dict, got {data!r}"
        known = {f.name for f in attrs.fields(cls)}
        unknown = set(data) - known
        assert not unknown, f"unknown run_config key(s) {unknown}; allowed: {known}"
        return cls(
            method=data.get("method"),
            tol=data.get("tol"),
            options=dict(data.get("options", {})),
        )
