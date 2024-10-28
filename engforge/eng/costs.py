"""Defines a CostModel & Economics Component that define & orchestrate cost accounting respectively.

CostModels can have a `cost_per_item` and additionally calculate a `cumulative_cost` from internally defined `CostModel`s.

CostModel's can have cost_property's which detail how and when a cost should be applied & grouped. By default each CostModel has a `cost_per_item` which is reflected in `item_cost` cost_property set on the `initial` term as a `unit` category. Multiple categories of cost are also able to be set on cost_properties as follows

Cost Models can represent multiple instances of a component, and can be set to have a `num_items` multiplier to account for multiple instances of a component. CostModels can have a `term_length` which will apply costs over the term, using the `cost_property.mode` to determine at which terms a cost should be applied. 

```
@forge
class Widget(Component,CostModel):
    
    #use num_items as a multiplier for costs, `cost_properties` can have their own custom num_items value.
    num_items:float = 100 

    @cost_property(mode='initial',category='capex,manufacturing',num_items=1)
    def cost_of_XYZ(self) -> float:
        return cost

```

Economics models sum CostModel.cost_properties recursively on the parent they are defined. Economics computes the grouped category costs for each item recursively as well as summary properties like annualized values and levelized cost. Economic output is determined by a `fixed_output` or overriding `calculate_production(self,parent)` to dynamically calculate changing economics based on factors in the parent.

Default costs can be set on any CostModel.Slot attribute, by using default_cost(<slot_name>,<cost>) on the class, this will provide a default cost for the slot if no cost is set on the instance. Custom costs can be set on the instance with custom_cost(<slot_name>,<cost>). If cost is a CostModel, it will be assigned to the slot if it is not already assigned.

The economics term_length applies costs over the term, using the `cost_property.mode` to determine at which terms a cost should be applied.

@forge
class Parent(System,CostModel)

    econ = Slot.define(Economics) #will calculate parent costs as well
    cost = Slot.define(Widget) #slots automatically set to none if no input provided

Parent(econ=Economics(term_length=25,discount_rate=0.05,fixed_output=1000))


"""

from engforge.components import Component
from engforge.configuration import forge, Configuration
from engforge.tabulation import TabulationMixin, system_property
from engforge.system_reference import Ref
from engforge.properties import (
    instance_cached,
    solver_cached,
    cached_system_property,
)
from engforge.logging import LoggingMixin
from engforge.component_collections import ComponentIter
import typing
import attrs
import uuid
import numpy
import collections
import pandas
import collections
import re


class CostLog(LoggingMixin):
    pass


log = CostLog()

# Cost Term Modes are a quick lookup for cost term support
global COST_TERM_MODES, COST_CATEGORIES
COST_TERM_MODES = {
    "initial": lambda inst, term, econ: True if term < 1 else False,
    "maintenance": lambda inst, term, econ: True if term >= 1 else False,
    "always": lambda inst, term, econ: True,
    "end": lambda inst, term, econ: (
        True if hasattr(econ, "term_length") and term == econ.term_length - 1 else False
    ),
}

category_type = typing.Union[str, list]
COST_CATEGORIES = set(("misc",))


def get_num_from_cost_prop(ref):
    """analyzes the reference and returns the number of items"""
    if isinstance(ref, (float, int)):
        return ref
    co = getattr(ref.comp.__class__, ref.key, None)
    return co.get_num_items(ref.comp)


class cost_property(system_property):
    """A thin wrapper over `system_property` that will be accounted by `Economics` Components and apply term & categorization

    `cost_property` should return a float/int always and will raise an error if the return annotation is different, although annotations are not required and will default to float.

    #Terms:
    Terms start counting at 0 and can be evald by the Economic.term_length
    cost_properties will return their value as system_properties do without regard for the term state, however a CostModel's costs at a term can be retrived by `costs_at_term`. The default mode is for `initial` cost

    #Categories:
    Categories are a way to report cost categories and multiple can be applied to a cost. Categories are grouped by the Economics system at reported in bulk by term and over the term_length

    """

    valild_types = (int, float)
    cost_categories: list = None
    term_mode: str = None
    num_items: int = None

    _all_modes: dict = COST_TERM_MODES
    _all_categories: set = COST_CATEGORIES

    def __init__(
        self,
        fget=None,
        fset=None,
        fdel=None,
        doc=None,
        desc=None,
        label=None,
        stochastic=False,
        mode: str = "initial",
        category: category_type = None,
        num_items: int = None,
    ):
        """extends system_property interface with mode & category keywords
        :param mode: can be one of `initial`,`maintenance`,`always` or a function with signature f(inst,term,econ) as an integer and returning a boolean True if it is to be applied durring that term.
        """
        super().__init__(fget, fset, fdel, doc, desc, label, stochastic)

        self.valild_types = (int, str)  # only numerics
        if isinstance(mode, str):
            mode = mode.lower()
            assert (
                mode in COST_TERM_MODES
            ), f"mode: {mode} is not in {set(COST_TERM_MODES.keys())}"
            self.term_mode = mode
        elif callable(mode):
            fid = str(uuid.uuid4())
            self.__class__._all_modes[fid] = mode
            self.term_mode = fid
        else:
            raise ValueError(f"mode: {mode} must be cost term str or callable")

        # cost categories
        if category is not None:
            if isinstance(category, str):
                self.cost_categories = category.split(",")
            elif isinstance(category, list):
                self.cost_categories = category
            else:
                raise ValueError(f"categories: {category} not string or list")
            for cc in self.cost_categories:
                self.__class__._all_categories.add(cc)
        else:
            self.cost_categories = ["misc"]

        # number of items override
        if num_items is not None:
            self.num_items = num_items

    def apply_at_term(self, inst, term, econ=None):
        if term < 0:
            raise ValueError(f"negative term!")
        if self.__class__._all_modes[self.term_mode](inst, term, econ):
            return True
        return False

    def get_func_return(self, func):
        """ensures that the function has a return annotation, and that return annotation is in valid sort types"""
        anno = func.__annotations__
        typ = anno.get("return", None)
        if typ is not None and not typ in (int, float):
            raise Exception(
                f"system_property input: function {func.__name__} must have valid return annotation of type: {(int,float)}"
            )
        else:
            self.return_type = float

    def get_num_items(self, obj):
        """applies the num_items override or the costmodel default if not set"""
        if self.num_items is not None:
            k = self.num_items
        else:
            k = obj.num_items if isinstance(obj, CostModel) else 1
        return k

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # class support
        if self.fget is None:
            raise AttributeError("unreadable attribute")

        # apply the costmodel with the item multiplier
        k = self.get_num_items(obj)
        return k * self.fget(obj)


@forge
class CostModel(Configuration, TabulationMixin):
    """CostModel is a mixin for components or systems that reports its costs through the `cost` system property, which by default sums the `item_cost` and `sub_items_cost`.

    `item_cost` is determined by `calculate_item_cost()` which by default uses: `cost_per_item` field to return the item cost, which defaults to `numpy.nan` if not set. Nan values are ignored and replaced with 0.

    `sub_items_cost` system_property summarizes the costs of any component in a Slot that has a `CostModel` or for SlotS which CostModel.declare_cost(`slot`,default=numeric|CostModelInst|dict[str,float])
    """

    # TODO: remove "default costs" concept and just use cost_properties since thats declarative and doesn't create a "phantom" representation to maintain
    # TODO: it might be a good idea to add a "castable" namespace for components so they can all reference a common dictionary. Maybe all problem variables are merged into a single namespace to solve issues of "duplication"
    _slot_costs: dict  # TODO: insantiate per class

    # basic attribute interface returns the item cost as `N x cost_per_item``
    cost_per_item: float = attrs.field(default=numpy.nan)
    num_items: int = attrs.field(default=1)  # set to 0 to disable the item cost

    def __on_init__(self):
        self.set_default_costs()
        self.debug(f"setting default costs {self._slot_costs}")

    def update_dflt_costs(self):
        """updates internal default slot costs if the current component doesn't exist or isn't a cost model, this is really a component method but we will use it never the less.

        This should be called from Component.update() if default costs are used
        """
        if self._slot_costs:
            current_comps = self.internal_components()
            for k, v in self._slot_costs.items():
                # Check if the cost model will  be accessed
                no_comp = k not in current_comps
                is_cost = not no_comp and isinstance(current_comps[k], CostModel)
                dflt_is_cost_comp = all(
                    [isinstance(v, CostModel), isinstance(v, Component)]
                )
                if no_comp and not is_cost and dflt_is_cost_comp:
                    self.debug("Updating default {k}")
                    v.update(self)

    def set_default_costs(self):
        """set default costs if no costs are set"""
        inter_config = self.internal_configurations()
        for k, dflt in self._slot_costs.items():
            if k not in inter_config and isinstance(dflt, CostModel):
                setattr(self, k, attrs.evolve(dflt, parent=self))
            elif (
                k not in inter_config
                and isinstance(dflt, type)
                and issubclass(dflt, CostModel)
            ):
                self.warning(
                    f"setting default cost {k} from costmodel class, provide a default instance instead!"
                )
                setattr(self, k, dflt())

        # Reset cache
        self.internal_components(True)

    @classmethod
    def subcls_compile(cls):
        assert not issubclass(cls, ComponentIter), "component iter not supported"
        log.debug(f"compiling costs {cls}")
        cls.reset_cls_costs()

    @classmethod
    def reset_cls_costs(cls):
        cls._slot_costs = {}

    @classmethod
    def default_cost(
        cls,
        slot_name: str,
        cost: typing.Union[float, "CostModel"],
        warn_on_non_costmodel=True,
    ):
        """Provide a default cost for Slot items that are not CostModel's. Cost is applied class wide, but can be overriden with custom_cost per instance"""
        assert not isinstance(
            cost, type
        ), f"insantiate classes before adding as a cost!"
        assert slot_name in cls.slots_attributes(), f"slot {slot_name} doesnt exist"
        assert isinstance(cost, (float, int, dict)) or isinstance(
            cost, CostModel
        ), "only numeric types or CostModel instances supported"

        atrb = cls.slots_attributes(attr_type=True)[slot_name]
        atypes = atrb.accepted
        if warn_on_non_costmodel and not any(
            [issubclass(at, CostModel) for at in atypes]
        ):
            log.warning(f"assigning cost to non CostModel based slot {slot_name}")

        cls._slot_costs[slot_name] = cost

        # IDEA: create slot if one doesn't exist, for dictionaries and assign a ComponentDict+CostModel in wide mode?

    def custom_cost(
        self,
        slot_name: str,
        cost: typing.Union[float, "CostModel"],
        warn_on_non_costmodel=True,
    ):
        """Takes class costs set, and creates a copy of the class costs, then applies the cost numeric or CostMethod in the same way but only for that instance of"""
        assert not isinstance(
            cost, type
        ), f"insantiate classes before adding as a cost!"
        assert slot_name in self.slots_attributes(), f"slot {slot_name} doesnt exist"
        assert isinstance(cost, (float, int, dict)) or isinstance(
            cost, CostModel
        ), "only numeric types or CostModel instances supported"

        atrb = self.__class__.slots_attributes(attr_type=True)[slot_name]
        atypes = atrb.accepted
        if warn_on_non_costmodel and not any(
            [issubclass(at, CostModel) for at in atypes]
        ):
            self.warning(f"assigning cost to non CostModel based slot {slot_name}")

        # convert from classinfo
        if self._slot_costs is self.__class__._slot_costs:
            self._slot_costs = self.__class__._slot_costs.copy()
        self._slot_costs[slot_name] = cost
        self.set_default_costs()

        # if the cost is a cost model, and there's nothing assigned to the slot, assign it
        # if assign_when_missing and isinstance(cost,CostModel):
        #     if hasattr(self,slot_name) and getattr(self,slot_name) is None:
        #         self.info(f'assigning custom cost {slot_name} with {cost}')
        #         setattr(self,slot_name,cost)
        #     elif hasattr(self,slot_name):
        #         self.warning(f'could not assign custom cost to {slot_name} with {cost}, already assigned to {getattr(self,slot_name)}')

    def calculate_item_cost(self) -> float:
        """override this with a parametric model related to this systems attributes and properties"""
        return self.num_items * self.cost_per_item

    @system_property
    def sub_items_cost(self) -> float:
        """calculates the total cost of all sub-items, using the components CostModel if it is provided, and using the declared_cost as a backup"""
        return self.sub_costs()

    @cost_property(mode="initial", category="unit")
    def item_cost(self) -> float:
        calc_item = self.calculate_item_cost()
        return numpy.nansum([0, calc_item])

    @system_property
    def combine_cost(self) -> float:
        """the sum of all cost properties"""
        return self.sum_costs()

    @system_property
    def itemized_costs(self) -> float:
        """sums costs of cost_property's in this item that are present at term=0"""
        initial_costs = self.costs_at_term(0)
        return numpy.nansum(list(initial_costs.values()))

    @system_property
    def future_costs(self) -> float:
        """sums costs of cost_property's in this item that do not appear at term=0"""
        initial_costs = self.costs_at_term(0, False)
        return numpy.nansum(list(initial_costs.values()))

    def sum_costs(self, saved: set = None, categories: tuple = None, term=0, econ=None):
        """sums costs of cost_property's in this item that are present at term=0, and by category if define as input"""
        if saved is None:
            saved = set((self,))  # item cost included!
        elif self not in saved:
            saved.add(self)
        itemcst = list(
            self.dict_itemized_costs(saved, categories, term, econ=econ).values()
        )
        csts = [self.sub_costs(saved, categories, term), numpy.nansum(itemcst)]
        return numpy.nansum(csts)

    def dict_itemized_costs(
        self,
        saved: set = None,
        categories: tuple = None,
        term=0,
        test_val=True,
        econ=None,
    ) -> dict:
        ccp = self.class_cost_properties()
        costs = {
            k: (
                obj.__get__(self)
                if obj.apply_at_term(self, term, econ) == test_val
                else 0
            )
            for k, obj in ccp.items()
            if categories is None
            or any([cc in categories for cc in obj.cost_categories])
        }
        return costs

    def sub_costs(self, saved: set = None, categories: tuple = None, term=0, econ=None):
        """gets items from CostModel's defined in a Slot attribute or in a slot default, tolerrant to nan's in cost definitions"""
        if saved is None:
            saved = set()

        sub_tot = 0

        for slot in self.slots_attributes():
            comp = getattr(self, slot)

            if comp in saved:
                # print(f'skipping {slot}:{comp}')
                continue

            elif isinstance(comp, Configuration):
                saved.add(comp)

            if isinstance(comp, CostModel):
                sub = comp.sum_costs(saved, categories, term, econ=econ)
                log.debug(
                    f"{self.identity} adding: {comp.identity if comp else comp}: {sub}+{sub_tot}"
                )
                cst = [sub_tot, sub]
                sub_tot = numpy.nansum(cst)

            elif (
                slot in self._slot_costs
                and (categories is None or "unit" in categories)
                and term == 0
            ):
                # Add default costs from direct slots
                dflt = self._slot_costs[slot]
                sub = eval_slot_cost(dflt, saved)
                log.debug(
                    f"sub: {self.identity} adding slot: {comp.identity if comp else comp}.{slot}: {sub}+{sub_tot}"
                )
                cst = [sub_tot, sub]
                sub_tot = numpy.nansum(cst)

            # add base class slot values when comp was nonee
            if comp is None:
                # print(f'skipping {slot}:{comp}')
                comp_cls = self.slots_attributes(attr_type=True)[slot].accepted
                for cc in comp_cls:
                    if issubclass(cc, CostModel):
                        if cc._slot_costs:
                            for k, v in cc._slot_costs.items():
                                sub = eval_slot_cost(v, saved)
                                log.debug(
                                    f"sub: {self.identity} adding dflt: {slot}.{k}: {sub}+{sub_tot}"
                                )
                                cst = [sub_tot, sub]
                                sub_tot = numpy.nansum(cst)
                            break  # only add once

        return sub_tot

    # Cost Term & Category Reporting
    def costs_at_term(self, term: int, test_val=True, econ=None) -> dict:
        """returns a dictionary of all costs at term i, with zero if the mode
        function returns False at that term

        :param econ: the economics component to apply "end" term mode
        """
        ccp = self.class_cost_properties()
        return {
            k: (
                obj.__get__(self)
                if obj.apply_at_term(self, term, econ) == test_val
                else 0
            )
            for k, obj in ccp.items()
        }

    @classmethod
    def class_cost_properties(cls) -> dict:
        """returns cost_property objects from this class & subclasses"""
        return {
            k: v
            for k, v in cls.system_properties_classdef(True).items()
            if isinstance(v, cost_property)
        }

    @property
    def cost_properties(self) -> dict:
        """returns the current values of the current properties"""
        ccp = self.class_cost_properties()
        return {k: obj.__get__(self) for k, obj in ccp.items()}

    @property
    def cost_categories(self):
        """returns itemized costs grouped by category"""
        base = {cc: 0 for cc in self.all_categories()}
        for k, obj in self.class_cost_properties().items():
            for cc in obj.cost_categories:
                base[cc] += obj.__get__(self)
        return base

    def cost_categories_at_term(self, term: int, econ=None):
        base = {cc: 0 for cc in self.all_categories()}
        for k, obj in self.class_cost_properties().items():
            if obj.apply_at_term(self, term, econ):
                for cc in obj.cost_categories:
                    base[cc] += obj.__get__(self)
        return base

    @classmethod
    def all_categories(self):
        return COST_CATEGORIES


cost_type = typing.Union[float, int, CostModel, dict]


def eval_slot_cost(slot_item: cost_type, saved: set = None):
    sub_tot = 0
    log.debug(f"evaluating slot: {slot_item}")
    if isinstance(slot_item, (float, int)):
        sub_tot += numpy.nansum([slot_item, 0])
    elif isinstance(slot_item, CostModel):
        sub_tot += numpy.nansum([slot_item.sum_costs(saved), 0])
    elif isinstance(slot_item, type) and issubclass(slot_item, CostModel):
        log.warning(
            f"slot {slot_item} has class CostModel, using its `item_cost` only, create an instance to fully model the cost"
        )
        sub_tot = numpy.nansum([sub_tot, slot_item.cost_per_item])
    elif isinstance(slot_item, dict):
        sub_tot += numpy.nansum(list(slot_item.values()))
    return sub_tot


def gend(deect: dict):
    for k, v in deect.items():
        if isinstance(v, dict):
            for kk, v in gend(v):
                yield f"{k}.{kk}", v
        else:
            yield k, v


parent_types = typing.Union[Component, "System"]


# TODO: automatically apply economics at problem level if cost_model present, no need for parent econ lookups
@forge
class Economics(Component):
    """
    Economics is a component that summarizes costs and reports the economics of a system and its components in a recursive format.
    Attributes:
        term_length (int): The length of the term for economic calculations. Default is 0.
        discount_rate (float): The discount rate applied to economic calculations. Default is 0.0.
        fixed_output (float): The fixed output value for the economic model. Default is numpy.nan.
        output_type (str): The type of output for the economic model. Default is "generic".
        terms_per_year (int): The number of terms per year for economic calculations. Default is 1.
        _calc_output (float): Internal variable to store calculated output.
        _costs (float): Internal variable to store calculated costs.
        _cost_references (dict): Internal dictionary to store cost references.
        _cost_categories (dict): Internal dictionary to store cost categories.
        _comp_categories (dict): Internal dictionary to store component categories.
        _comp_costs (dict): Internal dictionary to store component costs.
        parent (parent_types): The parent component or system.
    Methods:
        __on_init__(): Initializes internal dictionaries for cost and component categories.
        update(parent: parent_types): Updates the economic model with the given parent component or system.
        calculate_production(parent, term) -> float: Calculates the production output for the given parent and term. Must be overridden.
        calculate_costs(parent) -> float: Recursively calculates the costs for the given parent component or system.
        sum_cost_references(): Sums the cost references stored in the internal dictionary.
        sum_references(refs): Sums the values of the given references.
        get_prop(ref): Retrieves the property associated with the given reference.
        term_fgen(comp, prop): Generates a function to calculate the term value for the given component and property.
        sum_term_fgen(ref_group): Sums the term functions for the given reference group.
        internal_references(recache=True, numeric_only=False): Gathers and sets internal references for the economic model.
        lifecycle_output() -> dict: Returns lifecycle calculations for the levelized cost of energy (LCOE).
        lifecycle_dataframe() -> pandas.DataFrame: Simulates the economics lifecycle and stores the results in a term-based dataframe.
        _create_term_eval_functions(): Creates evaluation functions for term-based calculations grouped by categories and components.
        _gather_cost_references(parent: "System"): Gathers cost references from the parent component or system.
        _extract_cost_references(conf: "CostModel", bse: str): Extracts cost references from the given cost model configuration.
        cost_references(): Returns the internal dictionary of cost references.
        combine_cost() -> float: Returns the combined cost of the economic model.
        output() -> float: Returns the calculated output of the economic model.
    """

    term_length: int = attrs.field(default=0)
    discount_rate: float = attrs.field(default=0.0)
    fixed_output: float = attrs.field(default=numpy.nan)
    output_type: str = attrs.field(default="generic")
    terms_per_year: int = attrs.field(default=1)

    _calc_output: float = None
    _costs: float = None
    _cost_references: dict = None
    _cost_categories: dict = None
    _comp_categories: dict = None
    _comp_costs: dict = None
    parent: parent_types

    def __on_init__(self):
        self._cost_categories = collections.defaultdict(list)
        self._comp_categories = collections.defaultdict(list)
        self._comp_costs = dict()

    def update(self, parent: parent_types):
        if self.log_level <= 5:
            self.msg(f"econ updating costs: {parent}", lvl=5)

        self.parent = parent

        # this is kinda expensive to do every time, but we need to do it to get the costs
        self._gather_cost_references(parent)
        self._calc_output = self.calculate_production(parent, 0)
        self._costs = self.calculate_costs(parent)

        if self._calc_output is None:
            self.warning(f"no economic output!")
        if self._costs is None:
            self.warning(f"no economic costs!")

        self._anything_changed = True

    # TODO: expand this...
    @solver_cached
    def econ_output(self):
        return self.lifecycle_output

    @system_property(label="cost/output")
    def levelized_cost(self) -> float:
        """Price per kwh"""
        eco = self.econ_output
        return eco["summary.levelized_cost"]

    @system_property
    def total_cost(self) -> float:
        eco = self.econ_output
        return eco["summary.total_cost"]

    @system_property
    def levelized_output(self) -> float:
        """ouput per dollar (KW/$)"""
        eco = self.econ_output
        return eco["summary.levelized_output"]

    @system_property
    def term_years(self) -> float:
        """ouput per dollar (KW/$)"""
        eco = self.econ_output
        return eco["summary.years"]

    # Calculate Output
    def calculate_production(self, parent, term) -> float:
        """must override this function and set economic_output"""
        return numpy.nansum([0, self.fixed_output])

    def calculate_costs(self, parent) -> float:
        """recursively accounts for costs in the parent, its children recursively."""
        return self.sum_cost_references()

    # Reference Utilitly Functions
    # Set Costs over time "flat" through ref trickery...
    def sum_cost_references(self):
        cst = 0
        for k, v in self._cost_references.items():
            if k.endswith("item_cost"):
                val = v.value()
                if self.log_level < 2:
                    self.msg(f"add item cost: {k}|{val}")
                cst += val
            else:
                if self.log_level < 2:
                    self.msg(f"skip cost: {k}")
        return cst

    def sum_references(self, refs):
        return numpy.nansum([r.value() for r in refs])

    def get_prop(self, ref):
        if ref.use_dict:
            return ref.key
        elif ref.key in ref.comp.class_cost_properties():
            return ref.comp.class_cost_properties()[ref.key]
        # elif ref.key in ref.comp.system_properties_classdef():
        #     return ref.comp.system_properties_classdef()[ref.key]
        # else:
        #     raise KeyError(f'ref key doesnt exist as property: {ref.key}')

    def term_fgen(self, comp, prop):
        if isinstance(comp, dict):
            return lambda term: comp[prop] if term == 0 else 0
        return lambda term: (
            prop.__get__(comp) if prop.apply_at_term(comp, term, self) else 0
        )

    def sum_term_fgen(self, ref_group):
        term_funs = [self.term_fgen(ref.comp, self.get_prop(ref)) for ref in ref_group]
        return lambda term: numpy.nansum([t(term) for t in term_funs])

    # Gather & Set References (the magic!)
    # TODO: update internal_references callback to problem
    def internal_references(self, recache=True, numeric_only=False):
        """standard component references are"""

        recache = True  # override

        d = self._gather_references()
        self._create_term_eval_functions()
        # Gather all internal economic variables and report costs
        props = d["properties"]

        # calculate lifecycle costs
        lc_out = self.lifecycle_output

        if self._cost_references:
            props.update(**self._cost_references)

        # lookup ref from the cost categories dictionary, recreate every time
        if self._cost_categories:
            for key, refs in self._cost_categories.items():
                props[key] = Ref(
                    self._cost_categories,
                    key,
                    False,
                    False,
                    eval_f=self.sum_references,
                )

        if self._comp_categories:
            for key, refs in self._comp_categories.items():
                props[key] = Ref(
                    self._comp_categories,
                    key,
                    False,
                    False,
                    eval_f=self.sum_references,
                )

        for k, v in lc_out.items():
            props[k] = Ref(lc_out, k, False, False)

        return d

    def cost_summary(self, annualized=False, do_print=True, ignore_zero=True):
        """
        Generate a summary of costs, optionally annualized, and optionally print the summary.
        :param annualized: If True, include only annualized costs in the summary. Default is False.
        :type annualized: bool
        :param do_print: If True, print the summary to the console. Default is True.
        :type do_print: bool
        :param ignore_zero: If True, ignore costs with a value of zero. Default is True.
        :type ignore_zero: bool
        :return: A dictionary containing component costs, skipped costs, and summary.
        :rtype: dict
        """

        dct = self.lifecycle_output
        cols = list(dct.keys())

        skippd = set()
        comp_costs = collections.defaultdict(dict)
        comp_nums = collections.defaultdict(dict)
        summary = {}
        costs = {
            "comps": comp_costs,
            "skip": skippd,
            "summary": summary,
        }

        def abriv(val):
            evall = abs(val)
            if evall > 1e6:
                return f"{val/1E6:>12.4f} M"
            elif evall > 1e3:
                return f"{val/1E3:>12.2f} k"
            return f"{val:>12.2f}"

        for col in cols:
            is_ann = ".annualized." in col
            val = dct[col]

            # handle no value case
            if val == 0 and ignore_zero:
                continue

            if ".cost." in col and is_ann == annualized:
                base, cst = col.split(".cost.")
                if base == "lifecycle":
                    ckey = f"cost.{cst}"  # you're at the top bby
                else:
                    ckey = f"{base.replace('lifecycle.','')}.cost.{cst}"
                # print(ckey,str(self._comp_costs.keys()))
                comp_costs[base][cst] = val
                comp_nums[base][cst] = get_num_from_cost_prop(self._comp_costs[ckey])
            elif col.startswith("summary."):
                summary[col.replace("summary.", "")] = val
            else:
                self.msg(f"skipping: {col}")

        total_cost = sum([sum(list(cc.values())) for cc in comp_costs.values()])

        # provide consistent format
        hdr = "{key:<32}|\t{value:12.10f}"
        fmt = "{key:<32}|\t{fmt:<24} | {total:^12} | {pct:3.3f}%"
        title = f"COST SUMMARY: {self.parent.identity}"
        if do_print:
            self.info("#" * 80)
            self.info(f"{title:^80}")
            self.info("_" * 80)
            # possible core values
            if NI := getattr(self, "num_items", None):  # num items
                self.info(hdr.format(key="num_items", value=NI))
            if TI := getattr(self, "term_length", None):
                self.info(hdr.format(key="term_length", value=TI))
            if DR := getattr(self, "discount_rate", None):
                self.info(hdr.format(key="discount_rate", value=DR))
            if DR := getattr(self, "output", None):
                self.info(hdr.format(key="output", value=DR))
            # summary items
            for key, val in summary.items():
                self.info(hdr.format(key=key, value=val))
                itcst = "{val:>24}".format(val="TOTAL----->")
            self.info("=" * 80)
            self.info(
                fmt.format(key="COMBINED", fmt=itcst, total=abriv(total_cost), pct=100)
            )
            self.info("-" * 80)
            # itemization
            sgroups = lambda kv: sum(list(kv[-1].values()))
            for base, items in sorted(comp_costs.items(), key=sgroups, reverse=True):
                # skip if all zeros (allow for net negative costs)
                if (subtot := sum([abs(v) for v in items.values()])) > 0:
                    # self.info(f' {base:<35}| total ---> {abriv(subtot)} | {subtot*100/total_cost:3.0f}%')
                    pct = subtot * 100 / total_cost
                    itcst = "{val:>24}".format(val="TOTAL----->")
                    # todo; add number of items for cost comp
                    adj_base = base.replace("lifecycle.", "")
                    self.info(
                        fmt.format(
                            key=adj_base,
                            fmt=itcst,
                            total=abriv(subtot),
                            pct=pct,
                        )
                    )
                    # Sort costs by value
                    for key, val in sorted(
                        items.items(), key=lambda kv: kv[-1], reverse=True
                    ):
                        if val == 0 or numpy.isnan(val):
                            continue  # skip zero costs
                        # self.info(f' \t{key:<32}|{abriv(val)}             | {val*100/total_cost:^3.0f}%')
                        tot = abriv(val)
                        pct = val * 100 / total_cost
                        num = comp_nums[base][key]
                        itcst = (
                            f"{abriv(val/num):^18} x {num:3.0f}" if num != 0 else "0"
                        )
                        self.info(
                            fmt.format(key="-" + key, fmt=itcst, total=tot, pct=pct)
                        )
                    self.info("-" * 80)  # section break
            self.info("#" * 80)
            return costs

    @property
    def lifecycle_output(self) -> dict:
        """return lifecycle calculations for lcoe"""
        totals = {}
        totals["category"] = lifecat = {}
        totals["annualized"] = annul = {}
        summary = {}
        out = {"summary": summary, "lifecycle": totals}

        lc = self.lifecycle_dataframe
        for c in lc.columns:
            if "category" not in c and "cost" not in c:
                continue
            tot = lc[c].sum()  # lifecycle cost
            if "category" in c:
                c_ = c.replace("category.", "")
                lifecat[c_] = tot
            else:
                totals[c] = tot
            annul[c] = tot * self.terms_per_year / (self.term_length + 1)

        summary["total_cost"] = lc.term_cost.sum()
        summary["years"] = lc.year.max() + 1
        LC = lc.levelized_cost.sum()
        LO = lc.levelized_output.sum()
        summary["levelized_cost"] = LC / LO if LO != 0 else numpy.nan
        summary["levelized_output"] = LO / LC if LC != 0 else numpy.nan

        out2 = dict(gend(out))
        self._term_output = out2
        return self._term_output

    @property
    def lifecycle_dataframe(self) -> pandas.DataFrame:
        """simulates the economics lifecycle and stores the results in a term based dataframe"""
        out = []

        if self.term_length == 0:
            rng = [0]
        else:
            rng = list(range(0, self.term_length))

        for i in rng:
            t = i
            row = {"term": t, "year": t / self.terms_per_year}
            out.append(row)
            for k, sum_f in self._term_comp_category.items():
                row[k] = sum_f(t)
            for k, sum_f in self._term_cost_category.items():
                row[k] = sum_f(t)
            for k, sum_f in self._term_comp_cost.items():
                row[k] = sum_f(t)
            row["term_cost"] = tc = numpy.nansum(
                [v(t) for v in self._term_comp_cost.values()]
            )
            row["levelized_cost"] = tc * (1 + self.discount_rate) ** (-1 * t)
            row["output"] = output = self.calculate_production(self.parent, t)
            row["levelized_output"] = output * (1 + self.discount_rate) ** (-1 * t)

        return pandas.DataFrame(out)

    def _create_term_eval_functions(self):
        """uses reference summation grouped by categories & component"""
        self._term_comp_category = {}
        if self._comp_categories:
            for k, vrefs in self._comp_categories.items():
                self._term_comp_category[k] = self.sum_term_fgen(vrefs)

        self._term_cost_category = {}
        if self._cost_categories:
            for k, vrefs in self._cost_categories.items():
                self._term_cost_category[k] = self.sum_term_fgen(vrefs)

        self._term_comp_cost = {}
        if self._comp_costs:
            for k, ref in self._comp_costs.items():
                prop = self.get_prop(ref)
                self._term_comp_cost[k] = self.term_fgen(ref.comp, prop)

    def _gather_cost_references(self, parent: "System"):
        """put many tabulation.Ref objects into a dictionary to act as additional references for this economics model.

        References are found from a walk through the parent slots through all child slots
        """
        self._cost_references = CST = {}
        comps = {}
        comp_set = set()

        # reset data
        # groupings of components, categories, and pairs of components and categories
        self._cost_categories = collections.defaultdict(list)
        self._comp_categories = collections.defaultdict(list)
        self._comp_costs = dict()

        for key, level, conf in parent.go_through_configurations(check_config=False):
            # skip self
            if conf is self:
                continue

            bse = f"{key}." if key else ""
            # prevent duplicates'
            if conf in comp_set:
                continue

            elif isinstance(conf, Configuration):
                comp_set.add(conf)
            else:
                comp_set.add(key)

            _base = key.split(".")
            kbase = ".".join(_base[:-1])
            comp_key = _base[-1]

            self.debug(f"checking {key} {comp_key} {kbase}")

            # 0. Get Costs Directly From the cost model instance
            if isinstance(conf, CostModel):
                comps[key] = conf
                self.debug(f"adding cost model for {kbase}.{comp_key}")
                self._extract_cost_references(conf, bse)

            # Look For defaults!
            # 1. try looking for already parsed components (top down)
            elif kbase and kbase in comps:
                child = comps[kbase]
                if (
                    isinstance(child, CostModel)
                    and hasattr(child.parent, "_slot_costs")
                    and child.parent._slot_costs
                    and comp_key in child.parent._slot_costs
                ):
                    self.debug(f"adding cost for {kbase}.{comp_key}")
                    compcanidate = child._slot_costs[comp_key]
                    if isinstance(compcanidate, CostModel):
                        self.debug(f"dflt child costmodel {kbase}.{comp_key}")
                        self._extract_cost_references(compcanidate, bse + "cost.")
                    else:
                        _key = bse + "cost.item_cost"
                        self.debug(f"dflt child cost for {kbase}.{comp_key}")
                        CST[_key] = ref = Ref(
                            child._slot_costs,
                            comp_key,
                            False,
                            False,
                            eval_f=eval_slot_cost,
                        )
                        cc = "unit"
                        self._comp_costs[_key] = ref
                        self._cost_categories["category." + cc].append(ref)
                        self._comp_categories[bse + "category." + cc].append(ref)

            # 2. try looking at the parent
            elif (
                isinstance(parent, CostModel)
                and kbase == ""
                and comp_key in parent._slot_costs
            ):
                compcanidate = parent._slot_costs[comp_key]
                if isinstance(compcanidate, CostModel):
                    self.debug(f"dflt parent cost model for {kbase}.{comp_key}")
                    self._extract_cost_references(compcanidate, bse + "cost.")
                else:
                    self.debug(f"dflt parent cost for {kbase}.{comp_key}")
                    _key = bse + "cost.item_cost"
                    CST[_key] = ref = Ref(
                        parent._slot_costs,
                        comp_key,
                        False,
                        False,
                        eval_f=eval_slot_cost,
                    )
                    cc = "unit"
                    self._comp_costs[_key] = ref
                    self._cost_categories["category." + cc].append(ref)
                    self._comp_categories[bse + "category." + cc].append(ref)

            else:
                self.debug(f"unhandled cost: {key}")

        self._cost_references = CST
        self._anything_changed = True
        return CST

    def _extract_cost_references(self, conf: "CostModel", bse: str):
        # Add cost fields
        _key = bse + "item_cost"
        CST = self._cost_references
        if self.log_level < 5:
            self.msg(f"extracting costs from {bse}|{conf.identity}", lvl=5)

        # cost properties of conf item
        for cost_nm, cost_prop in conf.class_cost_properties().items():
            _key = bse + "cost." + cost_nm
            CST[_key] = ref = Ref(conf, cost_nm, True, False)
            self._comp_costs[_key] = ref

            # If there are categories we'll add references to later sum them
            if cost_prop.cost_categories:
                for cc in cost_prop.cost_categories:
                    self._cost_categories["category." + cc].append(ref)
                    self._comp_categories[bse + "category." + cc].append(ref)
            else:
                # we'll reference it as misc
                cc = "misc"
                self._cost_categories["category." + cc].append(ref)
                self._comp_categories[bse + "category." + cc].append(ref)

        comps_act = conf.internal_components()
        if self.log_level < 10:
            self.msg(
                f"{conf.identity if conf else conf} active components: {comps_act}",
                lvl=5,
            )
        # add slot costs with current items (skip class defaults)
        # TODO: remove defaults costs
        for slot_name, slot_value in conf._slot_costs.items():
            # Skip items that are internal components
            if slot_name in comps_act:
                self.debug(f"skipping slot {slot_name}")
                continue
            else:
                self.debug(f"adding slot {conf}.{slot_name}")
            # Check if current slot isn't occupied
            cur_slot = getattr(conf, slot_name)
            _key = bse + slot_name + ".cost.item_cost"
            if not isinstance(cur_slot, Configuration) and _key not in CST:
                CST[_key] = ref = Ref(
                    conf._slot_costs,
                    slot_name,
                    False,
                    False,
                    eval_f=eval_slot_cost,
                )

                cc = "unit"
                self._comp_costs[_key] = ref
                self._cost_categories["category." + cc].append(ref)
                self._comp_categories[bse + "category." + cc].append(ref)

            elif _key in CST:
                self.debug(f"skipping key {_key}")

        # add base class slot values when comp was none (recursively)
        for compnm, comp in conf.internal_configurations(False, none_ok=True).items():
            if comp is None:
                if self.log_level < 5:
                    self.msg(
                        f"{conf} looking up base class costs for {compnm}",
                        lvl=5,
                    )
                comp_cls = conf.slots_attributes(attr_type=True)[compnm].accepted
                for cc in comp_cls:
                    if issubclass(cc, CostModel):
                        if cc._slot_costs:
                            if self.log_level < 5:
                                self.msg(f"{conf} looking up base slot cost for {cc}")
                            for k, v in cc._slot_costs.items():
                                _key = bse + compnm + "." + k + ".cost.item_cost"
                                if _key in CST:
                                    if self.log_level < 10:
                                        self.debug(f"{conf} skipping dflt key {_key}")
                                    # break #skip if already added
                                    continue

                                if isinstance(v, CostModel):
                                    self._extract_cost_references(
                                        v, bse + compnm + "." + k + "."
                                    )
                                else:
                                    if self.log_level < 10:
                                        self.debug(
                                            f"adding missing cost for {conf}.{compnm}"
                                        )
                                    CST[_key] = ref = Ref(
                                        cc._slot_costs,
                                        k,
                                        False,
                                        False,
                                        eval_f=eval_slot_cost,
                                    )

                                    cc = "unit"
                                    self._comp_costs[_key] = ref
                                    self._cost_categories["category." + cc].append(ref)
                                    self._comp_categories[
                                        bse + "category." + cc
                                    ].append(ref)

                            break  # only add once

            elif isinstance(comp, CostModel):
                if self.log_level < 10:
                    self.debug(f"{conf} using actual costs for {comp}")

    @property
    def cost_references(self):
        return self._cost_references

    @system_property
    def combine_cost(self) -> float:
        if self._costs is None:
            return 0
        return self._costs

    @system_property
    def output(self) -> float:
        if self._calc_output is None:
            return 0
        return self._calc_output

    @property
    def cost_category_store(self):
        D = collections.defaultdict(list)
        Acat = set()
        for catkey, cdict in self._cost_categories.items():
            for ci in cdict:
                cprop = getattr(ci.comp.__class__, ci.key)
                ccat = set(cprop.cost_categories.copy())
                Acat = Acat.union(ccat)
                D[f"{ci.comp.classname}|{ci.key:>36}"] = ccat
        return D, Acat

    def create_cost_graph(self, plot=True):
        """creates a graph of the cost model using network X and display it"""
        import collections
        import networkx as nx

        D = collections.defaultdict(dict)
        for catkey, cdict in self._cost_categories.items():
            for ci in cdict:
                D[catkey][(ci.comp.classname, ci.key)] = ci

        G = nx.Graph()

        for d, dk in D.items():
            print(d.upper())
            cat = d.replace("category.", "")
            G.add_node(cat, category=cat)
            for kk, r in dk.items():
                cmp = kk[0]
                edge = kk[1]
                if cmp not in G.nodes:
                    G.add_node(cmp, component=cmp)
                G.add_edge(cmp, cat, cost=edge)
                # print(kk)

        # pos = nx.nx_agraph.graphviz_layout(G)
        # nx.draw(G, pos=pos)
        # nx.draw(G,with_labels=True)

        if plot:
            categories = nx.get_node_attributes(G, "category").keys()
            components = nx.get_node_attributes(G, "component").keys()

            cm = []
            for nd in G:
                if nd in categories:
                    cm.append("cyan")
                else:
                    cm.append("pink")

            pos = nx.spring_layout(G, k=0.2, iterations=20, scale=1)
            nx.draw(
                G,
                node_color=cm,
                with_labels=True,
                pos=pos,
                arrows=True,
                font_size=10,
                font_color="0.09",
                font_weight="bold",
                node_size=200,
            )

        return G

    def cost_matrix(self):
        D, Cats = self.cost_category_store
        X = list(sorted(Cats))
        C = list(sorted(D.keys()))
        M = []
        for k in C:
            cats = D[k]
            M.append([True if x in cats else numpy.nan for x in X])

        Mx = numpy.array(M)
        X = numpy.array(X)
        C = numpy.array(C)
        return Mx, X, C

    def create_cost_category_table(self):
        """creates a table of costs and categories"""
        Mx, X, C = self.cost_matrix()

        fig, ax = subplots(figsize=(12, 12))

        Mc = numpy.nansum(Mx, axis=0)
        x = numpy.argsort(Mc)

        Xs = X[x]
        ax.imshow(Mx[:, x])
        ax.set_yticklabels(C, fontdict={"family": "monospace", "size": 8})
        ax.set_yticks(numpy.arange(len(C)))
        ax.set_xticklabels(Xs, fontdict={"family": "monospace", "size": 8})
        ax.set_xticks(numpy.arange(len(Xs)))
        ax.grid(which="major", linestyle=":", color="k", zorder=0)
        xticks(rotation=90)
        fig.tight_layout()

    def determine_exclusive_cost_categories(
        self,
        include_categories=None,
        ignore_categories: set = None,
        min_groups: int = 2,
        max_group_size=None,
        min_score=0.95,
        include_item_cost=False,
    ):
        """looks at all possible combinations of cost categories, scoring them based on coverage of costs, and not allowing any double accounting of costs. This is an NP-complete problem and will take a long time for large numbers of items. You can add ignore_categories to ignore certain categories"""
        import itertools

        Mx, X, C = self.cost_matrix()

        bad = []
        solutions = []
        inx = {k: i for i, k in enumerate(X)}

        assert include_categories is None or set(X).issuperset(
            include_categories
        ), "include_categories must be subset of cost categories"

        # ignore categories
        if ignore_categories:
            X = [x for x in X if x not in ignore_categories]

        if include_categories:
            # dont include them in pair since they are added to the group explicitly
            X = [x for x in X if x not in include_categories]

        if not include_item_cost:
            C = [c for c in C if "item_cost" not in c]

        Num_Costs = len(C)
        goal_score = Num_Costs * min_score
        NumCats = len(X) // 2
        GroupSize = NumCats if max_group_size is None else max_group_size
        for ni in range(min_groups, GroupSize):
            print(f"level {ni}/{GroupSize}| {len(solutions)} answers")
            for cgs in itertools.combinations(X, ni):
                val = None

                # make the set with included if needed
                scg = set(cgs)
                if include_categories:
                    scg = scg.union(include_categories)

                # skip bad groups
                if any([b.issubset(scg) for b in bad]):
                    # print(f'skipping {cgs}')
                    # sys.stdout.write('.')
                    continue

                good = True  # innocent till guilty
                for cg in cgs:
                    xi = Mx[:, inx[cg]].copy()
                    xi[np.isnan(xi)] = 0
                    if val is None:
                        val = xi
                    else:
                        val = val + xi
                    # determine if any overlap (only pair level)
                    if np.nanmax(val) > 1:
                        print(f"bad {cgs}")
                        bad.append(scg)
                        good = False
                        break

                score = np.nansum(val)
                if good and score > goal_score:
                    print(f"found good: {scg}")
                    solutions.append({"grp": scg, "score": score, "gsize": ni})

        return solutions

    def cost_categories_from_df(self, df):
        categories = set()
        for val in df.columns:
            m = re.match(
                re.compile("economics\.lifecycle\.category\.(?s:[a-z]*)$"), val
            )
            if m:
                categories.add(val)
        return categories

    def plot_cost_categories(self, df, group, cmap="tab20c", make_title=None, ax=None):
        categories = self.cost_categories_from_df(df)
        from matplotlib import cm

        # if grps:
        # assert len(grps) == len(y_vars), 'groups and y_vars must be same length'
        # assert all([g in categories for g in grps]), 'all groups must be in categories'
        # TODO: project costs onto y_vars
        # TODO: ensure groups and y_vars are same length

        color = cm.get_cmap(cmap)
        styles = {
            c.replace("economics.lifecycle.category.", ""): {
                "color": color(i / len(categories))
            }
            for i, c in enumerate(categories)
        }

        if make_title is None:

            def make_title(row):
                return f'{row["name"]}x{row["num_items"]} @{"floating" if row["ldepth"]>50 else "fixed"}'

        # for j,grp in enumerate(groups):
        figgen = False
        if ax is None:
            figgen = True
            fig, ax = subplots(figsize=(12, 8))
        else:
            fig = ax.get_figure()

        titles = []
        xticks = []
        data = {}
        i = 0

        for inx, row in df.iterrows():
            i += 1
            tc = row["economics.summary.total_cost"]
            cat_costs = {
                k.replace("economics.lifecycle.category.", ""): row[k]
                for k in categories
            }
            # print(i,cat_costs)

            spec_costs = {k: v for k, v in cat_costs.items() if k in group}
            pos_costs = {k: v for k, v in spec_costs.items() if v >= 0}
            neg_costs = {k: v for k, v in spec_costs.items() if k not in pos_costs}
            neg_amt = sum(list(neg_costs.values()))
            pos_amt = sum(list(pos_costs.values()))

            data[i] = spec_costs.copy()

            com = {"x": i, "width": 0.5, "linewidth": 0}
            cur = neg_amt
            for k, v in neg_costs.items():
                opt = {} if i != 1 else {"label": k}
                ax.bar(height=abs(v), bottom=cur, **com, **styles[k], **opt)
                cur += abs(v)
            for k, v in pos_costs.items():
                opt = {} if i != 1 else {"label": k}
                ax.bar(height=abs(v), bottom=cur, **com, **styles[k], **opt)
                cur += abs(v)
            xticks.append(com["x"])
            titles.append(make_title(row))

        # Format the chart
        ax.legend(loc="upper right")
        ax.set_xlim([0, i + max(2, 0.2 * i)])
        ax.set_xticks(xticks)

        ax.set_xticklabels(titles, rotation=90)
        ylim = ax.get_ylim()
        ylim = ylim[0] - 0.05 * abs(ylim[0]), ylim[1] + 0.05 * abs(ylim[1])
        ax.set_yticks(numpy.linspace(*ylim, 50), minor=True)
        ax.grid(which="major", linestyle="--", color="k", zorder=0)
        ax.grid(which="minor", linestyle=":", color="k", zorder=0)
        if figgen:
            fig.tight_layout()
        return {"fig": fig, "ax": ax, "data": data}


# TODO: add costs for iterable components (wide/narrow modes)
# if isinstance(conf,ComponentIter):
#     conf = conf.current
#     #if isinstance(conf,CostModel):
#     #    sub_tot += conf.item_cost
# if isinstance(conf,ComponentIter):
#     item = conf.current
#     if conf.wide:
#         items = item
#     else:
#         items = [items]
# else:
#     items = [conf]
# for conf in items:


# if isinstance(self,CostModel):
#     sub_tot += self.item_cost

# accomodate ComponentIter in wide mode
# if isinstance(self,ComponentIter):
#     item = self.current
#     if self.wide:
#         items = item
#     else:
#         items = [items]
# else:
#     items = [self]

# accomodate ComponentIter in wide mode
# for item in items:
