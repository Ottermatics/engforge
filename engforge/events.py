"""The engforge event bus — the explicit change-signal layer of the 1.0 vision
(docs/ENGFORGE_1.0.md M8): attribute writes, slot swaps, and collection
mutations EMIT events that downstream machinery subscribes to (sparse
recording now; dependency-graph invalidation next).

Design rules:
  * pyee (already a dependency) process-wide bus; zero cost when nobody listens
    (emitters guard on ``has_listeners``).
  * Events carry the live objects — subscribers resolve names through the
    session refs (one naming authority, no duplicate string plumbing).
  * ``watch(event, fn)`` is a context manager so listeners are scoped —
    sessions subscribe for their lifetime and unhook on exit.

Events:
  VALUE_CHANGED(instance, name, old, new)      an attrs field changed value
  SLOT_CHANGED(instance, name, old, new)       a slot's COMPONENT was swapped
  COLLECTION_CHANGED(collection, key, item)    a collection member set/added
"""

from contextlib import contextmanager

from pyee import EventEmitter

VALUE_CHANGED = "value_changed"
SLOT_CHANGED = "slot_changed"
COLLECTION_CHANGED = "collection_changed"

#: the process-wide change bus
bus = EventEmitter()


def has_listeners(event: str) -> bool:
    return bool(bus.listeners(event))


def emit_value_changed(instance, name, old, new):
    if bus.listeners(VALUE_CHANGED):
        bus.emit(VALUE_CHANGED, instance, name, old, new)


def emit_slot_changed(instance, name, old, new):
    if bus.listeners(SLOT_CHANGED):
        bus.emit(SLOT_CHANGED, instance, name, old, new)


def emit_collection_changed(collection, key, item):
    if bus.listeners(COLLECTION_CHANGED):
        bus.emit(COLLECTION_CHANGED, collection, key, item)


@contextmanager
def watch(event: str, fn):
    """subscribe ``fn`` to ``event`` for the duration of the block"""
    bus.on(event, fn)
    try:
        yield fn
    finally:
        try:
            bus.remove_listener(event, fn)
        except Exception:
            pass
