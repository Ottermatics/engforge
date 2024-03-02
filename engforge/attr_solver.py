"""solver defines a SolverMixin for use by System.

Additionally the SOLVER attribute is defined to add complex behavior to a system as well as add constraints and transient integration.
"""

import attrs
import uuid
import numpy
import scipy.optimize as scopt
from contextlib import contextmanager
import copy
import datetime
import typing

from engforge.attributes import ATTR_BASE, AttributeInstance
from engforge.system_reference import Ref,maybe_attr_inst,maybe_ref
from engforge.properties import *

import itertools

INTEGRATION_MODES = ["euler", "trapezoid", "implicit"]
SOLVER_OPTIONS = ["root", "minimize"]


class AttrSolverLog(LoggingMixin):
    pass


log = AttrSolverLog()


indep_type = typing.Union[str]
dep_type = typing.Union[system_property,callable]
ref_type = typing.Union[str, system_property, callable, float, int]


# Solver minimizes residual by changing vars
class SolverInstance(AttributeInstance):
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    solver: "SOLVER"

    # compiled info
    obj: "Ref"
    var: "Ref"
    lhs: "Ref"
    rhs: "Ref"
    const_f: "Ref"

    __slots__ = ["system", "solver", "obj", "var", "lhs", "rhs","const_f"]

    def __init__(self, solver: "SOLVER", system: "System",**kw) -> None:
        """kwargs passed to compile"""
        self.solver = solver
        self.system = system
        self.compile(**kw)

    def compile(self, **kwargs):
        """establishes the references for the solver to use directly"""
        from engforge.solver import objectify,secondary_obj
        if self.solver.slvtype == "var":
            self.var = self.system.locate_ref(self.solver.var)
            self.system.debug(f"solving with indpendent: {self.var}")

        elif self.solver.slvtype in ["ineq", "eq"]:
            self.lhs = self.system.locate_ref(self.solver.lhs)
            
            if hasattr(self.solver, "rhs") and (isinstance(self.solver.rhs, str)or callable(self.solver.rhs)):
                self.rhs = self.system.locate_ref(self.solver.rhs)
            elif hasattr(self.solver, "rhs") and self.solver.rhs is not None:
                #raise Exception(f"bad rhs: {self.solver.rhs}")
                self.rhs = self.solver.rhs
                
            m = " = " if self.solver.slvtype == "eq" else " >= "
            self.system.debug(f"solving {self.lhs}{m}{self.rhs}")

            if hasattr(self,'rhs') and self.rhs is not None:
                fun = lambda sys,info: maybe_ref(self.lhs,SolverInstance,sys=sys,info=info) - maybe_ref(self.rhs,SolverInstance,sys=sys,info=info)
                fun.__name__ = f"{self.lhs}_{self.rhs}_{self.solver.slvtype}"
                objfunc = Ref(self.system,fun)
            else:
                objfunc = self.lhs

            self.const_f = objfunc

        elif self.solver.slvtype == 'obj':
            self.obj_ref = self.system.locate_ref(self.solver.obj)
            if self.solver.kind == 'max':
                mult=-1
                bias=None
                power=2
            else:
                mult=None
                bias=None
                power=2

            fun = lambda sys,info: maybe_ref(self.obj_ref,SolverInstance,sys=sys,info=info,mult=mult,bias=bias,power=power)  
            fun.__name__ = f"obj_{self.solver.obj}_{self.solver.kind}"
            self.obj = Ref(self.system,fun)
            #self.obj = self.system.locate_ref(fun)
            self.system.debug(f"solving with obj: {self.obj}")

    @property
    def indep_ref(self):
        if self.solver.slvtype == "var":
            return self.var
        return None
    
    @property
    def obj_key_in(self):
        if self.solver.slvtype == "obj":
            return self.solver.obj
        return None
    
    @property
    def constraint_refs(self):
        if self.solver.slvtype in ["eq","ineq"]:
            return self.const_f
        return None

    @property
    def constraints(self):
        if self.solver.slvtype in ['var','obj']:
            return self.solver.constraints
        else:
            #just me
            return [{'type':self.slvtype,'parm':self.solver.lhs,'value':self.const_f}]
        

    @property
    def slvtype(self):
        return self.solver.slvtype
    
    @property
    def combos(self):
        return self.solver.combos
    
    @property
    def normalize(self):
        return self.solver.normalize

    def as_ref_dict(self):
        out = {'var':None,'eq':None,'ineq':None,'obj':None}

        if self.solver.slvtype == "var":
                out['var'] = self.var
        elif self.solver.slvtype == "obj":
                out['obj'] = self.obj
        elif self.solver.slvtype == "eq":
                out['eq'] =  self.const_f
        if self.solver.slvtype == "ineq":
                out['ineq'] =  self.const_f
        
        return out
    
        
        
            
            

class Solver(ATTR_BASE):
    """solver creates subclasses per solver balance"""

    obj: dep_type = None
    var: indep_type = None
    rhs: ref_type = None
    lhs: ref_type = None
    slvtype: str
    constraints: dict
    combos: list = None
    normalize: ref_type
    allow_constraint_override: bool = True
    attr_prefix = "SOLVER"

    instance_class = SolverInstance

    define = None

    @classmethod
    def process_combos(cls, combos):
        if isinstance(combos, str):
            return combos.split(",")
        elif isinstance(combos, list):
            return combos
        
    # @classmethod
    # def collect_attr_inst(cls,system)->dict:
    #     """collects all the attributes for a system, grouped by solver type"""
    #     clsses = super(Solver,cls).collect_attr_inst(system)
    #     out = {'var':{},'eq':{},'ineq':{},'obj':{}}
    #     for k,v in clsses.items():
    #         out[v.solver.slvtype][k] = cls.handle_instance(v)
    #     return out

    # TODO: add normalize attribute to tune optimizations
    @classmethod
    def declare_var(cls, var: str, **kwargs):
        """
        Defines a solver variable for optimization in the class, constraints defined on this solver will correspond to the limits of that variable
        :param var: The var attribute for the solver variable.
        :param combos: The combinations of the solver variable.
        :return: The setup class for the solver variable.
        """
        # Create A New Signals Class
        combos = kwargs.get("combos", var)
        new_name = f"SOLVER_var_{var}".replace(".", "_")
        bkw = {"parm": var, "value": None}
        constraints = [{"type": "min", **bkw}, {"type": "max", **bkw}]
        new_dict = dict(
            name=new_name,
            var=var,
            slvtype="var",
            constraints=constraints,
            combos=cls.process_combos(combos),
            default_options=cls.default_options.copy(),
        )
        return cls._setup_cls(new_name, new_dict)


    @classmethod
    def objective(cls, obj: str, **kwargs):
        """
        Defines a solver variable for optimization in the class, constraints defined on this solver will correspond to the limits of that variable

        :param obj: The var attribute for the solver variable.
        :param combos: The combinations of the solver variable.
        :parma kind: the kind of optimization, either min or max
        :return: The setup class for the solver variable.
        """
        # Create A New Signals Class
        combos = kwargs.get("combos", "default")
        kind = kwargs.get("kind", "min")
        assert kind in ("min", "max")

        new_name = f"SOLVER_obj_{obj}_{kind}".replace(".", "_")
        bkw = {"parm": obj, "value": None}
        new_dict = dict(
            name=new_name,
            obj=obj,
            slvtype="obj",
            kind=kind,
            constraints=[],
            combos=cls.process_combos(combos),
            default_options=cls.default_options.copy(),
        )
        return cls._setup_cls(new_name, new_dict)

    obj = objective  

    @classmethod
    def equality_constraint(
        cls, lhs: "system_property", rhs: "system_property" = 0, **kwargs
    ):
        """Defines an equality constraint based on a required lhs of equation, and an optional rhs, the difference of which will be driven to zero"""
        combos = kwargs.get("combos", "default")

        # Create A New Signals Class
        new_name = f"SOLVER_coneq_{lhs}_{rhs}".replace(".", "_")
        new_dict = dict(
            name=new_name,
            lhs=lhs,
            rhs=rhs,
            slvtype="eq",
            constraints=[],
            combos=cls.process_combos(combos),
            default_options=cls.default_options.copy(),
        )
        return cls._setup_cls(new_name, new_dict)

    eq_con = equality_constraint

    @classmethod
    def inequality_constraint(cls, lhs: ref_type, rhs: ref_type = 0, **kwargs):
        """Defines an inequality constraint"""
        combos = kwargs.get("combos", "default")

        # Create A New Signals Class
        new_name = f"SOLVER_conineq_{lhs}_{rhs}".replace(".", "_")
        new_dict = dict(
            name=new_name,
            lhs=lhs,
            rhs=rhs,
            slvtype="ineq",
            constraints=[],
            combos=cls.process_combos(combos),
            default_options=cls.default_options.copy(),
        )
        return cls._setup_cls(new_name, new_dict)

    ineq_con = inequality_constraint

    # TODO: add minimize / maximize / objective options

    @classmethod
    def class_validate(cls, instance, **kwargs):
        from engforge.properties import system_property

        system = cls.config_cls

        # TODO: normalize references to ref or constant here
        # TODO: add a check for the solver to be a valid solver type

        if cls.slvtype == "var":
            parm_type = system.locate(cls.var)
            if parm_type is None:
                raise Exception(f"var not found: {cls.var}")
            assert isinstance(
                parm_type, attrs.Attribute
            ), f"bad parm {cls.var} not attribute: {parm_type}"
            assert parm_type.type in (
                int,
                float,
            ), f"bad parm {cls.var} not numeric"

        elif cls.slvtype in ["ineq", "eq"]:
            driv_type = system.locate(cls.lhs)
            if driv_type is None:
                raise Exception(f"LHS not found: {cls.lhs}")
            assert isinstance(
                driv_type, system_property
            ), f"bad LHS {cls.lhs} type: {driv_type}"
            if cls.rhs is not None:
                driv_type = system.locate(cls.rhs)
                if driv_type is None:
                    raise Exception(f"RHS not found: {cls.rhs}")
                assert isinstance(
                    driv_type, system_property
                ), f"bad RHS {cls.rhs} type: {driv_type}"

        else:
            raise Exception(f"bad slvtype: {cls.slvtype}")

    @classmethod
    def create_instance(cls, system: "System") -> SolverInstance:
        return SolverInstance(cls, system)

    @classmethod
    def add_var_constraint(cls, value, kind="ineq"):
        """adds a `type` constraint to the solver. If value is numeric it is used as a bound with `scipy` optimize.

        If value is a function it should be of the form value(Xarray) and will establish an inequality constraint that var parameter must be:
            1. less than for max
            2. more than for min

        During the evaluation of the limit function system.X should be set, and pre_execute() have already run.

        :param type: str, must be either min or max with var value comparison, or with a function additionally eq,ineq (same as max(0)) can be used
        :value: either a numeric (int,float), or a function, f(system)
        """
        assert cls is not SOLVER, f"must set constraint on SOLVER subclass"
        # assert not cls.constraint_exists(type=kind,parm=parm), f"constraint already exists!"
        assert isinstance(value, (int, float)) or callable(
            value
        ), f"only int,float or callables allow. Callables must take system as argument"

        parm = cls.var
        assert parm is not None, "must provide parm on non-var solvers"
        assert cls.slvtype == "var", "only Solver.declare_var can have constraints"
        assert kind in ("min", "max")
            

        if (
            cls.allow_constraint_override
            and (cinx := cls.constraint_exists(type=kind, parm=parm))
            is not None
        ):
            #print(f'replacing constraint {cinx} with {kind} {value} {parm}')
            constraint = cls.constraints[cinx]
            constraint["type"] = kind
            constraint["parm"] = parm
            constraint["value"] = value
        else:
            cls.constraints.append({"type": kind, "value": value, 'parm':parm})

    @classmethod
    def constraint_exists(cls, **kw):
        """check constraints on the system, return its index position if found, else None."""
        for i, c in enumerate(cls.constraints):
            if all([c[k] == v for k, v in kw.items()]):
                return i
        return None

    @classmethod
    def cls_solveable(cls):
        """returns True if the system has all the required references to solve the solver"""
        if cls.slvtype == "var":
            return cls.var
        else:
            return cls.lhs or cls.rhs

    @classmethod
    def sys_solveable(cls, system):
        """returns True if the system has all the required references to solve the solver"""
        raise NotImplementedError("#TODO")


# Support Previous SnakeCase
SOLVER = Solver
