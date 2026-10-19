"""The problem graph — the owned dependency DAG over a system's fields and
properties, used to detect update cycles (the core 1.0 convergence concern) and
to order evaluation.

Node identity is the dotted ref name (the single naming authority). Edges run in
dependency direction (``a -> b`` means *b reads a*), so a strongly-connected
component of size > 1 is an algebraic update cycle.

Two layers, split by WHEN they can be known:

* **Static, per class, computed at class-compile (startup):** the dependency
  chains of each ``@system_property`` getter — the ``self.<attr>`` /
  ``self.<comp>.<attr>`` reads extracted by a bytecode scan. These depend only
  on the class's code, so they are scanned once and cached on the class
  (``cls_property_dependencies`` / ``TabulationMixin.pre_compile``), never per
  run. The scan over-approximates safely (it cannot miss a real ``self`` read),
  so a reported "no cycle" is trustworthy.
* **Per problem (runtime):** stitching those per-class chains along the actual
  component/slot instance tree into dotted node names, and (later) marking roles
  from the active combo selection. This is cheap — no bytecode work.

This is the seed of Part IX's compiled graph (M2/M4) and its cycle diagnostics
(M10).
"""

import dis

import networkx as nx


def _self_names(code):
    """local names that mean ``self`` in this code object."""
    out = set()
    if code.co_argcount:
        out.add(code.co_varnames[0])
    for n in (code.co_freevars or ()) + (code.co_cellvars or ()):
        if n == "self":
            out.add(n)
    return out


def _is_method_form(ins):
    """3.12 merged LOAD_METHOD into LOAD_ATTR; bit0 of arg flags a method load
    (a call, which terminates a value-read chain)."""
    if ins.opname != "LOAD_ATTR":
        return False
    if ins.arg is not None and (ins.arg & 1):
        return True
    return "NULL|self" in (ins.argrepr or "")


def read_chains(func):
    """the set of ``self``-rooted attribute read chains in ``func``.

    ``self.a`` -> ``('a',)``; ``self.comp.b`` -> ``('comp', 'b')``. Method-form
    loads terminate a chain (they are calls, not value reads).
    """
    code = getattr(func, "__code__", None)
    if code is None:
        return set()
    selfn = _self_names(code)
    ins = list(dis.get_instructions(code))
    n = len(ins)
    chains = set()
    i = 0
    while i < n:
        op = ins[i]
        if op.opname in ("LOAD_FAST", "LOAD_DEREF", "LOAD_CLASSDEREF"):
            j, chain = i + 1, []
            while j < n and ins[j].opname == "LOAD_ATTR":
                if _is_method_form(ins[j]):
                    break
                chain.append(ins[j].argval)
                j += 1
            if op.argval in selfn and chain:
                chains.add(tuple(chain))
            i = max(j, i + 1)
            continue
        i += 1
    return chains


class ProblemGraph:
    """a dependency DAG over one system's fields and properties.

    Assembled per problem from the per-class dependency chains (scanned once at
    startup via ``cls_property_dependencies``) stitched along the instance tree.
    """

    def __init__(self, system):
        self.system = system
        self.G = nx.DiGraph()
        self._build()

    # --- construction --------------------------------------------------------
    def _walk(self):
        """yield (dotted_prefix, component) over the system and its children."""
        seen = set()

        def rec(comp, prefix):
            if id(comp) in seen:
                return
            seen.add(id(comp))
            yield prefix, comp
            internals = getattr(comp, "internal_configurations", None)
            if not callable(internals):
                return
            try:
                children = internals(False)
            except Exception:
                children = {}
            for key, child in (children or {}).items():
                if child is None:
                    continue
                yield from rec(child, f"{prefix}{key}." if prefix else f"{key}.")

        yield from rec(self.system, "")

    def _build(self):
        G = self.G
        for prefix, comp in self._walk():
            props = set(comp.system_properties_classdef().keys())
            try:
                fields = set(comp.input_fields().keys())
            except Exception:
                fields = set()
            for nm in props | fields:
                node = f"{prefix}{nm}"
                G.add_node(node, kind="property" if nm in props else "field")

            # per-class dependency chains (scanned once at class-compile)
            deps = comp.cls_property_dependencies()
            for pname, chains in deps.items():
                target = f"{prefix}{pname}"
                if target not in G:
                    continue
                for chain in chains:
                    dep = prefix + ".".join(chain)
                    if dep in G and dep != target:
                        G.add_edge(dep, target, kind="read")

    # --- queries -------------------------------------------------------------
    def cycles(self):
        """algebraic update cycles: SCCs with more than one node, each a sorted
        list of member node ids."""
        return [
            sorted(c)
            for c in nx.strongly_connected_components(self.G)
            if len(c) > 1
        ]

    def layers(self):
        """feed-forward evaluation layers (topological generations). Only valid
        when there are no cycles."""
        if self.cycles():
            return None
        return [sorted(gen) for gen in nx.topological_generations(self.G)]

    def report(self) -> str:
        """a readable summary: node/edge counts and any update cycles."""
        lines = [
            f"ProblemGraph[{getattr(self.system, 'classname', self.system)}]: "
            f"{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges"
        ]
        cyc = self.cycles()
        if not cyc:
            lines.append("  no update cycles (feed-forward)")
        else:
            lines.append(f"  {len(cyc)} update cycle(s):")
            for c in cyc:
                lines.append(f"    cycle: {' -> '.join(c)} -> {c[0]}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"ProblemGraph({self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges, {len(self.cycles())} cycles)"
        )
