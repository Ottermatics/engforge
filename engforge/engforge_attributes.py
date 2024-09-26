from engforge.attributes import ATTR_BASE, AttributeInstance
from engforge.attr_dynamics import Time
from engforge.attr_solver import Solver
from engforge.attr_signals import Signal
from engforge.attr_slots import Slot
from engforge.attr_plotting import Plot, Trace
from engforge.logging import LoggingMixin, log
from engforge.typing import *

from contextlib import contextmanager
import deepdiff
import typing
import datetime


import attr, attrs

from attrs import Attribute


class EngAttr(LoggingMixin):
    pass


log = EngAttr()


def get_attributes_of(cls, subclass_of: type = None, exclude=False):
    choose = issubclass
    if exclude:
        choose = lambda ty, type_set: not issubclass(ty, type_set)

    if subclass_of is None:
        subclass_of = ATTR_BASE

    # This handles the attrs class before or after compilation
    attrval = {}
    if "__attrs_attrs__" in cls.__dict__:  # Handle Attrs Class
        for k, v in attrs.fields_dict(cls).items():
            if isinstance(v.type, type) and choose(v.type, subclass_of):
                attrval[k] = v

    # else:  # Handle Pre-Attrs Class
    # FIXME: should this run first?
    for k, v in cls.__dict__.items():
        if isinstance(v, type) and choose(v, subclass_of):
            attrval[k] = v

    return attrval


class AttributedBaseMixin(LoggingMixin):
    """A mixin that adds the ability to configure all engforge.core attributes of a class"""

    # Auto Configuration Methods
    @classmethod
    def collect_all_attributes(cls):
        """collects all the attributes for a system"""
        out = {}
        for base_class in ATTR_BASE.subclasses():
            nm = base_class.__name__.lower()
            out[nm] = base_class.collect_cls(cls)
        return out

    def collect_inst_attributes(self, **kw):
        """collects all the attributes for a system"""
        out = {}
        for base_class in ATTR_BASE.subclasses():
            nm = base_class.__name__.lower()
            out[nm] = base_class.collect_attr_inst(self, **kw)
        return out

    @classmethod
    def _get_init_attrs_data(cls, subclass_of: type, exclude=False):
        choose = issubclass
        if exclude:
            choose = lambda ty, type_set: not issubclass(ty, type_set)

        attrval = {}
        if "__attrs_attrs__" in cls.__dict__:  # Handle Attrs Class
            for k, v in attrs.fields_dict(cls).items():
                if isinstance(v.type, type) and choose(v.type, subclass_of):
                    attrval[k] = v

        # else:  # Handle Pre-Attrs Class
        # FIXME: should this run first?
        for k, v in cls.__dict__.items():
            if isinstance(v, type) and choose(v, subclass_of):
                attrval[k] = v

        return attrval

    # Attribute Methods
    @property
    def attrs_fields(self) -> set:
        return set(attr.fields(self.__class__))

    @classmethod
    def _extract_type(cls, typ):
        """gathers valid types for an attribute.type"""
        from engforge.attr_slots import Slot
        from engforge.configuration import Configuration

        if not isinstance(typ, type) or typ is None:
            return list()

        if issubclass(typ, Slot):
            accept = typ.accepted
            if isinstance(accept, (tuple, list)):
                return list(accept)
            return [accept]

        elif issubclass(typ, Configuration):
            return [typ]

        elif issubclass(typ, TABLE_TYPES):
            return [typ]

    @classmethod
    def check_ref_slot_type(cls, sys_key: str) -> list:
        """recursively checks class slots for the key, and returns the slot type"""

        from engforge.configuration import Configuration

        slot_refs = cls.slot_refs()
        if sys_key in slot_refs:
            return slot_refs[sys_key]

        slts = cls.input_attrs()
        key_segs = sys_key.split(".")
        out = []
        # print(slts.keys(),sys_key)
        if "." not in sys_key and sys_key not in slts:
            pass

        elif sys_key in slts:
            # print(f'slt find {sys_key}')
            return cls._extract_type(slts[sys_key].type)
        else:
            fst = key_segs[0]
            rem = key_segs[1:]
            if fst in slts:
                sub_clss = cls._extract_type(slts[fst].type)
                out = []
                for acpt in sub_clss:
                    if isinstance(acpt, type) and issubclass(acpt, Configuration):
                        vals = acpt.check_ref_slot_type(".".join(rem))
                        # print(f'recursive find {acpt}.{rem} = {vals}')
                        if vals:
                            out.extend(vals)

                    elif isinstance(acpt, type):
                        out.append(acpt)

        slot_refs[sys_key] = out

        return out

    @classmethod
    def slot_refs(cls, recache=False):
        """returns all slot references in this configuration"""
        key = f"{cls.__name__}_prv_slot_sys_refs"
        if recache == False and hasattr(cls, key):
            return getattr(cls, key)
        o = {}
        setattr(cls, key, o)
        return o

    @classmethod
    def slots_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all slots attributes for class"""
        return cls._get_init_attrs_data(Slot)

    @classmethod
    def signals_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all signals attributes for class"""
        return cls._get_init_attrs_data(Signal)

    @classmethod
    def solvers_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all signals attributes for class"""
        return cls._get_init_attrs_data(Solver)

    @classmethod
    def transients_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all signals attributes for class"""
        return cls._get_init_attrs_data(Time)

    @classmethod
    def trace_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all trace attributes for class"""
        return cls._get_init_attrs_data(Trace)

    @classmethod
    def plot_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all plot attributes for class"""
        return cls._get_init_attrs_data(Plot)

    @classmethod
    def input_attrs(cls):
        return attr.fields_dict(cls)

    @classmethod
    def input_fields(cls, add_ign_types: list = None):
        """no attr base types, no tuples, no lists, no dicts"""
        ignore_types = [
            ATTR_BASE,
            # tuple,
            # list,
            # dict,
        ]
        if add_ign_types:
            ignore_types.extend(add_ign_types)
        return cls._get_init_attrs_data(tuple(ignore_types), exclude=True)

    @classmethod
    def numeric_fields(cls):
        ignore_types = (
            ATTR_BASE,
            str,
            tuple,
            list,
            dict,
        )
        typ = cls._get_init_attrs_data(ignore_types, exclude=True)
        return {k: v for k, v in typ.items() if v.type in (int, float)}

    @classmethod
    def table_fields(cls):
        keeps = (str, float, int)  # TODO: add numpy fields
        typ = cls._get_init_attrs_data(keeps)
        return {k: v for k, v in typ.items()}

    # Dictonaries
    @property
    def as_dict(self):
        """returns values as they are in the class instance"""
        from engforge.configuration import Configuration

        inputs = self.input_attrs()
        # TODO: add signals?
        properties = getattr(self, "system_properties_classdef", None)
        if properties:
            inputs.update(properties())

        o = {k: getattr(self, k, None) for k, v in inputs.items()}
        return o

    #TODO: refactor this, allowing a nesting return option for sub components, by default True (later to be reverted to False, as a breaking change). this messes up hashing and we can just the other object hash
    @property
    def input_as_dict(self):
        """returns values as they are in the class instance, but converts classes inputs to their input_as_dict"""
        from engforge.configuration import Configuration

        o = {k: getattr(self, k, None) for k in self.input_fields()}
        o = {
            k: v if not isinstance(v, Configuration) else v.input_as_dict
            for k, v in o.items()
        }
        return o

    @property
    def table_row_dict(self):
        """returns values as they would be put in a table row from this instance ignoring any sub components"""
        from engforge.configuration import Configuration
        o = {k: getattr(self, k, None) for k in self.table_fields()}
        return o        

    @property
    def numeric_as_dict(self):
        """recursively gets internal components numeric_as_dict as well as its own numeric values"""
        from engforge.configuration import Configuration

        o = {k: getattr(self, k, None) for k in self.numeric_fields()}
        o = {
            k: v if not isinstance(v, Configuration) else v.numeric_as_dict
            for k, v in o.items()
        }
        return o

    # Hashes
    # TODO: issue with logging sub-items
    def hash(self,*args, **input_kw):
        """hash by parm or by input_kw, only input can be hashed by lookup as system properties can create a recursive loop and should be deterministic from input"""
        d = {k:v for k,v in self.input_as_dict.items() if k in args}
        d.update(input_kw) #override with input_kw
        return d,deepdiff.DeepHash(d, ignore_encoding_errors=True,significant_digits = 6)
        
    def hash_with(self, **input_kw):
        """
        Generates a hash for the object's dictionary representation, updated with additional keyword arguments.
        Args:
            **input_kw: Arbitrary keyword arguments to update the object's dictionary representation.
        Returns:
            The hash value of the updated dictionary.
        Raises:
            Any exceptions raised by deepdiff.DeepHash if hashing fails.
        """
        d = self.as_dict
        d.update(input_kw)
        return deepdiff.DeepHash(d, ignore_encoding_errors=True)[d]

    def hash_numeric_with(self, **input_kw):
        d = self.numeric_as_dict
        d.update(input_kw)
        return deepdiff.DeepHash(d, ignore_encoding_errors=True)[d]

    @property
    def unique_hash(self):
        d = self.as_dict
        return deepdiff.DeepHash(d, ignore_encoding_errors=True)[d]

    @property
    def input_hash(self):
        d = self.input_as_dict
        return deepdiff.DeepHash(d, ignore_encoding_errors=True)[d]

    @property
    def table_hash(self):
        d,hsh = self.hash(**self.table_row_dict)
        return hsh[d]
    
    @property
    def numeric_hash(self):
        d = self.numeric_as_dict
        return deepdiff.DeepHash(d, ignore_encoding_errors=True)[d]

    # Configuration Push/Pop methods
    def setattrs(self, dict):
        """sets attributes from a dictionary"""
        msg = f"invalid keys {set(dict.keys()) - set(self.input_attrs())}"
        assert set(dict.keys()).issubset(set(self.input_attrs())), msg
        for k, v in dict.items():
            setattr(self, k, v)

    @contextmanager
    def difference(self, **kwargs):
        """a context manager that will allow you to dynamically change any information, then will change it back in a fail safe way.

        with self.difference(name='new_name', value = new_value) as new_config:
            #do stuff with config, ok to fail

        you may not access any "private" variable that starts with an `_` as in _whatever

        difference is useful for saving slight differences in configuration in conjunction with solve
        you might create wrappers for eval, or implement a strategy pattern.

        only attributes may be changed.

        #TODO: allow recursive operation with sub comps or systems.
        #TODO: make a full system copy so the system can be reverted later
        """
        _temp_vars = {}

        _temp_vars.update(
            {
                arg: getattr(self, arg)
                for arg in kwargs.keys()
                if hasattr(self, arg)
                if not arg.startswith("_")
            }
        )

        bad_vars = set.difference(set(kwargs.keys()), set(_temp_vars.keys()))
        if bad_vars:
            self.warning("Could Not Change {}".format(",".join(list(bad_vars))))

        try:  # Change Variables To Input
            self.setattrs(kwargs)
            yield self
        finally:
            rstdict = {k: _temp_vars[k] for k, v in kwargs.items()}
            self.setattrs(rstdict)
