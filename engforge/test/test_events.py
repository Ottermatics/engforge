"""Contract tests for the engforge event bus (engforge/events.py):
value / slot / collection change signals and the watch() scoping rule.
"""

import unittest

from engforge import System, Component, forge, field, Slot
from engforge import events
from engforge.component_collections import ComponentDict, ComponentIterator
from engforge.problem_context import ProblemExec
from engforge.properties import system_property


@forge
class Inner(Component):
    y: float = field(default=1.0)


@forge
class EvtSys(System):
    a: float = field(default=2.0)
    comp: Inner = Slot.define(Inner)

    @system_property
    def s(self) -> float:
        return self.a + 10 * self.comp.y


@forge
class InnerDict(ComponentDict):
    component_type = Inner


@forge
class InnerList(ComponentIterator):
    component_type = Inner


def mk():
    return EvtSys(comp=Inner())


class TestValueEvents(unittest.TestCase):
    def test_value_changed_emits(self):
        s = mk()
        seen = []
        with ProblemExec(s, {}, level_name="t"):
            with events.watch(events.VALUE_CHANGED, lambda *a: seen.append(a)):
                s.a = 5.0
        self.assertEqual(len(seen), 1)
        inst, name, old, new = seen[0]
        self.assertIs(inst, s)
        self.assertEqual(name, "a")
        self.assertEqual(old, 2.0)
        self.assertEqual(new, 5.0)

    def test_same_value_no_emit(self):
        s = mk()
        seen = []
        with ProblemExec(s, {}, level_name="t"):
            with events.watch(events.VALUE_CHANGED, lambda *a: seen.append(a)):
                s.a = 2.0  # unchanged scalar — is_different is False
        self.assertEqual(seen, [])

    def test_slot_changed_emits(self):
        s = mk()
        old_comp = s.comp
        new_comp = Inner(y=9.0)
        seen = []
        with ProblemExec(s, {}, level_name="t"):
            with events.watch(events.SLOT_CHANGED, lambda *a: seen.append(a)):
                s.comp = new_comp
        self.assertEqual(len(seen), 1)
        inst, name, old, new = seen[0]
        self.assertIs(inst, s)
        self.assertEqual(name, "comp")
        self.assertIs(old, old_comp)
        self.assertIs(new, new_comp)


class TestCollectionEvents(unittest.TestCase):
    def test_dict_setitem_emits(self):
        dc = InnerDict(component_type=Inner)
        item = Inner()
        seen = []
        with events.watch(events.COLLECTION_CHANGED, lambda *a: seen.append(a)):
            dc["k"] = item
        self.assertEqual(len(seen), 1)
        coll, key, it = seen[0]
        self.assertIs(coll, dc)
        self.assertEqual(key, "k")
        self.assertIs(it, item)

    def test_list_append_emits_and_resets(self):
        lc = InnerList(component_type=Inner)
        item = Inner()
        seen = []
        with events.watch(events.COLLECTION_CHANGED, lambda *a: seen.append(a)):
            lc.append(item)
        self.assertEqual(len(seen), 1)
        coll, key, it = seen[0]
        self.assertIs(coll, lc)
        self.assertEqual(key, 0)
        self.assertIs(it, item)
        self.assertIs(lc.current_item, None)  # reset() ran

    def test_append_type_enforced(self):
        lc = InnerList(component_type=Inner)
        with self.assertRaises(AssertionError):
            lc.append("not a component")

    def test_setitem_type_enforced(self):
        dc = InnerDict(component_type=Inner)
        with self.assertRaises(AssertionError):
            dc["k"] = "not a component"


class TestEventDrivenRefresh(unittest.TestCase):
    """the session SUBSCRIBES to structural events: a slot swap or collection
    mutation flags `comp_changed`, and the next refresh rebinds the reference
    graph. This is what makes `full_update=False` correct instead of fast-and-wrong.
    """

    def test_slot_swap_invalidates_refs(self):
        s = mk()
        with ProblemExec(s, {}, level_name="t") as px:
            px.full_update = False  # event-driven refresh only
            self.assertFalse(px.comp_changed)

            new = Inner(y=5.0)
            s.comp = new
            self.assertTrue(px.comp_changed)  # the event fired

            px.min_refresh()
            self.assertFalse(px.comp_changed)  # consumed by the refresh
            ref = px.sesh.all_comps_and_vars["comp.y"]
            self.assertIs(ref.comp, new)  # refs rebound to the new component
            self.assertEqual(ref.value(), 5.0)

    def test_same_component_reassignment_is_not_structural(self):
        """re-assigning the SAME component must not flag — this case dominates
        the assignment volume during a solve (4263 -> 4 events on the suite)"""
        s = mk()
        same = s.comp
        with ProblemExec(s, {}, level_name="t") as px:
            px.full_update = False
            s.comp = same
            self.assertFalse(px.comp_changed)

    def test_collection_change_invalidates_refs(self):
        s = mk()
        lc = InnerList(component_type=Inner)
        with ProblemExec(s, {}, level_name="t") as px:
            px.full_update = False
            self.assertFalse(px.comp_changed)
            lc.append(Inner(y=3.0))
            self.assertTrue(px.comp_changed)

    def test_subscription_removed_on_exit(self):
        from engforge import events as ev

        s = mk()
        with ProblemExec(s, {}, level_name="t"):
            self.assertTrue(ev.has_listeners(ev.SLOT_CHANGED))
        self.assertFalse(ev.has_listeners(ev.SLOT_CHANGED))
        self.assertFalse(ev.has_listeners(ev.COLLECTION_CHANGED))


class TestWatchScoping(unittest.TestCase):
    def test_listener_removed_after_block(self):
        seen = []
        with events.watch(events.VALUE_CHANGED, lambda *a: seen.append(a)):
            self.assertTrue(events.has_listeners(events.VALUE_CHANGED))
        self.assertFalse(events.has_listeners(events.VALUE_CHANGED))
        s = mk()
        with ProblemExec(s, {}, level_name="t"):
            s.a = 7.0  # no listener → zero-cost guard, nothing recorded
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
