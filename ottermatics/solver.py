"""solver defines a SolverMixin for use by System.

Additionally the SOLVER attribute is defined to add complex behavior to a system as well as add constraints and transient integration.
"""

import attrs
import uuid
import numpy
import scipy.optimize as scopt
from contextlib import contextmanager
import copy
from ottermatics.configuration import otterize
from ottermatics.properties import *

import itertools

INTEGRATION_MODES = ["euler", "trapezoid", "implicit"]
SOLVER_OPTIONS = ["root", "minimize"]


class SolverLog(LoggingMixin):
    pass


log = SolverLog()


class SolverMixin:
    _run_id: str = None
    _solved = None
    _trans_opts = None

    # TODO: implement custom solver with unified derivative equation.
    # _system_jacobean = None
    # _last_X = None
    # _current_X = None
    # _last_F = None
    # _current_F = None

    _converged = False
    # solver_option = attrs.field(default='root',validator=attrs.validators.in_(SOLVER_OPTIONS))
    solver_option = "root"

    # Configuration Information
    @instance_cached
    def signals(self):
        return {k: getattr(self, k) for k in self.signals_attributes()}

    @instance_cached
    def solvers(self):
        return {k: getattr(self, k) for k in self.solvers_attributes()}

    @instance_cached
    def transients(self):
        return {k: getattr(self, k) for k in self.transients_attributes()}

    @property
    def solved(self):
        return self._solved

    @classmethod
    def parse_run_kwargs(cls, **kwargs):
        """ensures correct input for simulation.
        :returns: first set of input for initalization, and all input dictionaries as tuple.

        """

        # TODO: handle slot argument and sub-system argumentsconda

        # Configure Extra Transient Arguments
        _trans_opts = None
        if cls.transients_attributes():
            # Get Timestep

            # timestep
            if "dt" not in kwargs:
                pass
            # f"transients require timestep input `dt`"
            dt = float(kwargs.pop("dt"))

            # endtime
            if "endtime" not in kwargs:
                pass

            # add data
            _trans_opts = {"dt": None, "endtime": None}
            _trans_opts["dt"] = dt

            # f"transients require `endtime` to specify "
            _trans_opts["endtime"] = endtime = float(kwargs.pop("endtime"))
            _trans_opts["Nrun"] = max(int(endtime / dt) + 1, 1)

        # Validate OTher Arguments By Parameter Or Comp-Recursive
        parm_args = {k: v for k, v in kwargs.items() if "." not in k}
        comp_args = {k: v for k, v in kwargs.items() if "." in k}

        # check parms
        argdiff = set(parm_args).difference(set(cls.input_fields()))
        assert not argdiff, f"bad input {argdiff}"

        # check components
        comps = set([k.split(".")[0] for k in comp_args.keys()])
        compdiff = comps.difference(set(cls.slots_attributes()))
        assert not compdiff, f"bad slot references {compdiff}"

        _input = {}
        _firsts = {}
        test = lambda v: isinstance(v, (int, float, str)) or v is None

        # parameters input
        for k, v in kwargs.items():
            # Ensure Its a List
            if isinstance(v, numpy.ndarray):
                v = v.tolist()

            if not isinstance(v, list):
                assert test(v), f"bad values {k}:{v}"
                v = [v]
            else:
                assert all([test(vi) for vi in v]), f"bad values: {k}:{v}"

            if k not in _input:
                _input[k] = v
                _firsts[k] = v[0]
            else:
                _input[k].extend(v)

        return _firsts, _input, _trans_opts

    @classmethod
    def sim(cls, _cb=None, **kwargs):
        """applies a permutation of input parameters for parameters not marked as transient, creates a system instance and returns the results of run()
        :param dt: timestep in s, required for transients
        :param endtime: when to end the simulation
        :returns: system or list of systems. If transient a set of systems that have been run with permutations of the input, otherwise a single system with all permutations input run
        """
        prev_item = None
        log.info(f"simulating analysis: {cls.__name__} with input {kwargs}")

        _firsts, _input, trs_opts = cls.parse_run_kwargs(**kwargs)

        # Create The System
        system = cls(**_firsts)
        return system.run(**kwargs, **trs_opts, _cb=_cb)

    def run(self, revert=True, _cb=None, **kwargs):
        """applies a permutation of input parameters for parameters not marked as transient, runs the system instance by applying input to the system and its slot-components, ensuring that the targeted attributes actualy exist. The run command additionally configures the transient parameters

        :param dt: timestep in s, required for transients
        :param endtime: when to end the simulation
        :param revert: will reset the values of X that were recorded at the beginning of the run.
        :param _cb: a callback function that takes the system as an argument cb(system)

        :returns: system or list of systems. If transient a set of systems that have been run with permutations of the input, otherwise a single system with all permutations input run
        """
        self.info(f"running system {self.identity} with input {kwargs}")

        # TODO: allow setting sub-component parameters with `slot1.slot2.attrs`. Ensure checking slots exist, and attrs do as well.

        # RUN
        if not self.solved or self.anything_changed:
            _firsts, _input, trs_opts = self.parse_run_kwargs(**kwargs)

            if revert:
                # TODO: revert all internal components too with `system_state` and set_system_state(**x,comp.x...)
                revert_x = self.system_state

            # prep references for keys
            refs = {}
            for k, v in _input.items():
                refs[k] = self.locate_ref(k)

            # Premute the input as per SS or Transient Logic
            ingrp = list(_input.values())
            keys = list(_input.keys())
            for parms in itertools.product(*ingrp):
                cur = {k: v for k, v in zip(keys, parms)}
                for k, v in cur.items():
                    refs[k].set_value(v)

                # Transeint
                if self.transients and trs_opts:
                    self._run_id = int(uuid.uuid4())
                    self.time = 0
                    self.run_transient(
                        dt=trs_opts["dt"], N=trs_opts["Nrun"], _cb=_cb
                    )

                else:
                    # stead=y state
                    if self._run_id is None:
                        self._run_id = int(uuid.uuid4())
                    self.evaluate(_cb=_cb)

            if revert and revert_x:
                self.set_system_state(**revert_x)

            self._solved = True
        else:
            raise Exception("Analysis Already Solved")

    def run_transient(self, dt, N, _cb=None):
        """integrates the time series over N points at a increment of dt"""
        for i in range(N):
            # Run The SS Timestep
            self.evaluate(_cb=_cb)
            # Run Integrators
            for k, v in self.transients.items():
                v.integrate(dt)
            # Mark Time
            self.time = self.time + dt

    # Single Point Flow
    def evaluate(self, _cb=None, **fields_input):
        """Evaluates the system with overrides for fields_input

        :param _cb: an optional callback taking the system as an argument
        """

        # Pre Execute Which Sets Fields And PRE Signals
        from ottermatics.system import System

        if self.log_level < 20:
            self.debug(f"running with X: {self.X} & {fields_input}")
        self.pre_execute(**fields_input)

        # Runs The Solver
        try:
            out = self.execute()
        except Exception as e:
            self.error(e, f"solver failed @ {self.X}")
            out = None
            raise e

        # Solve Each Internal System
        for key, comp in self.internal_components.items():
            if isinstance(comp, System):
                comp.evaluate()

        # Post Execute
        self.post_execute()

        # Save The Data
        self.save_data()

        if _cb:
            _cb(self)

        return out

    def pre_execute(self, **fields_input):
        """runs the solver of the system"""

        # TODO: set system fields from input
        for signame, sig in self.signals.items():
            if sig.mode == "pre" or sig.mode == "both":
                sig.apply()

    def post_execute(self):
        """runs the solver of the system"""

        for signame, sig in self.signals.items():
            if sig.mode == "post" or sig.mode == "both":
                sig.apply()

    def execute(self):
        """Solves the system's system of constraints and integrates transients if any exist"""

        # solvers
        def f(*x):
            res = self.calcF(x[0], pre_execute=True)
            return res

        def f_min(*x):
            res = self.calcF(x[0], pre_execute=True)
            ans = numpy.linalg.norm(res, 2)

            if ans < 1:
                return ans**0.5
            return ans

        if self.solver_option == "root" and not self.has_constraints:
            ans = scopt.root(f, x0=self.X)
            if ans.success:
                self.setX(ans.x, pre_execute=True)
                self._converged = True
            else:
                self._converged = False
                raise Exception(f"solver didnt converge: {ans}")
            return ans

        elif self.solver_option == "minimize" or self.has_constraints:
            cons = self.solver_constraints
            opts = {"rhobeg": 0.01, "catol": 1e-4, "tol": 1e-6}
            ans = scopt.minimize(
                f_min, x0=self.X, method="COBYLA", options=opts, **cons
            )
            if ans.success and abs(ans.fun) < 0.01:
                self.setX(ans.x)
                self._converged = True
            elif ans.success:
                self.setX(ans.x)
                self.warning(
                    f"solver didnt fully solve equations! {ans.x} -> residual: {ans.fun}"
                )
                self._converged = False
            else:
                self._converged = False
                raise Exception(f"solver didnt converge: {ans}")

            return ans

        else:
            self.warning(f"no solution attempted!")

    def setX(self, x_ordered=None, pre_execute=True, **x_kw):
        """Apply full X data to X. #TODO allow partial data"""
        # assert len(input_values) == len(self._X), f'bad input data'
        assert x_ordered is not None or x_kw, f"have to have some input"
        if x_ordered is not None:
            if isinstance(x_ordered, numpy.ndarray):
                x_ordered = x_ordered.tolist()
            assert not x_kw, f"there must only be ordered inputs or keyword"
            assert len(x_ordered) == len(self._X), f"must be a full sized"
            for f, x in zip(self._X.values(), x_ordered):
                f.set_value(x)
        else:
            assert set(x_kw).issubset(set(self._X)), f"incorrect parms {x_kw}"
            _Xref = self._X

            for k, v in x_kw.items():
                self.debug(f"setting X value {k}={v}")
                _Xref[k].set_value(v)

        if pre_execute:
            self.pre_execute()

    # Function Calculation
    def calcF(self, x_ordered=None, pre_execute=True, **x_kw):
        """calculates the orderdered or keyword input per Ref in `_F`"""
        assert x_ordered is not None or x_kw, f"have to have some input"
        with self.revert_X() as rx:
            if x_ordered is not None:
                # print(x_ordered)
                if isinstance(x_ordered, numpy.ndarray):
                    x_ordered = x_ordered.tolist()
                assert not x_kw, f"there must only be ordered inputs or keyword"
                assert len(x_ordered) == len(
                    self._X
                ), f"must be a full sized input"
                # Set Em
                self.setX(x_ordered, pre_execute=pre_execute)
                # Get Em
                o = [f.value() for f, x in zip(self.Flist, x_ordered)]
                # o = [f.value() for f  in self.Flist]
                return numpy.array(o)
            elif x_kw:
                assert (
                    set(x_kw) in self._X
                ), f"incompatable keyword args in {x_kw}"
                # Set Em
                self.setX(**x_kw, pre_execute=pre_execute)
                # Get Em
                o = [self._F[k].value(v) for k, v in x_kw.items()]
                return numpy.array(o)
        # Forget Em

        # raise NotImplemented('#TODO')

    def calcJacobean(self, x, pct=0.001, diff=0.0001):
        """
        returns the jacobiean by modifying X' <= X*pct + diff and recording the differences. When abs(x) < pct x' = x*1.1 + diff
        """
        with self.revert_X():
            # initalize here
            self.setX(x)
            self.pre_execute()

            rows = []
            dxs = []
            Fbase = self.F
            for k, v in self._X.items():
                x = v.value()
                if abs(x) > pct:
                    new_x = x * (1 + pct) + diff
                else:
                    new_x = x * (1.1) + diff * x
                dx = new_x - x
                dxs.append(dx)

                v.set_value(new_x)
                self.pre_execute()

                F_ = self.F - Fbase

                rows.append(F_ / dx)

        return numpy.column_stack(rows)

    # Useful Properties (F & X)
    @instance_cached
    def Flist(self):
        """returns F() for each solver dependent as an anonymous function"""
        return [self._F[self.F_keyword_order[i]] for i in range(len(self._F))]

    @instance_cached
    def _X(self):
        """stores the internal references to the solver references"""
        return {k: v.independent for k, v in self.solvers.items()}

    @instance_cached
    def _F(self):
        """stores the internal references to the solver references"""
        return {k: v.dependent for k, v in self.solvers.items()}

    @instance_cached
    def F_keyword_order(self):
        """defines the order of inputs in ordered mode for calcF"""
        return {i: k for i, k in enumerate(self.solvers)}

    @instance_cached
    def F_keyword_rev_order(self):
        """defines the order of inputs in ordered mode for calcF"""
        return {k: i for i, k in self.F_keyword_order.items()}

    @property
    def X(self) -> numpy.array:
        """The current state of the system"""
        return numpy.array([v.value() for v in self._X.values()])

    @property
    def F(self) -> numpy.array:
        """The current solution to the system"""
        return numpy.array([v.value() for v in self._F.values()])

    @contextmanager
    def revert_X(self, *ing):
        """
        Stores the _X parameter at present, the reverts to that state when done
        #TODO: add pre_execute / post_execute options
        """
        X_now = self.X
        try:  # Change Variables To Input
            yield self
        finally:
            self.setX(X_now)
            self.pre_execute()

    # Solution Managment
    @instance_cached
    def has_constraints(self):
        """checks for any active constrints"""
        for solvname, solv in self.solvers.items():
            soltype = solv.solver
            if any([s is not None for s in soltype.constraints.values()]):
                return True
        return False

    @instance_cached
    def constraints(self):
        """returns a list of constraitns by type"""
        out = []
        for solvname, solv in self.solvers.items():
            soltype = solv.solver
            for stype, sv in soltype.constraints.items():
                if sv is None:
                    continue
                out.append(
                    {
                        "name": solvname,
                        "type": stype,
                        "value": sv,
                        "solver": soltype,
                        "independent": soltype.independent,
                        "solv_instance": solv,
                    }
                )
        return out

    @instance_cached
    def solver_constraints(self):
        """formatted as arguments for the solver"""
        out = {"bounds": [], "constraints": []}

        # boudns must be X length wide
        for parm, solv in self.solvers.items():
            out["bounds"].append([None, None])

        groups = {}
        for const in self.constraints:
            name = const["name"]
            if name not in groups:
                groups[name] = [const]
            else:
                groups[name].append(const)

        for group, values in groups.items():
            for vcon in values:
                con = self.create_constraint(vcon, out, group)
                out["constraints"].append(con)
        return out

    def create_constraint(self, vcon, out, group):
        contype = vcon["type"]
        value = vcon["value"]
        x_inx = self.F_keyword_rev_order[vcon["name"]]
        # print(group,contype,value,x_inx,isinstance(value,(int,float)))

        if isinstance(value, (int, float)):
            # its a number
            val = float(value)
            if contype == "max":
                # make objective that is negative when x > lim
                def fun(x):
                    return val - x[x_inx]

            else:

                def fun(x):
                    return x[x_inx] + val

            cons = {"type": "ineq", "fun": fun}
            return cons
        else:
            val = copy.copy(value)
            # its a function
            if contype == "max":
                # make objective that is negative when x > lim
                def fun(x):
                    with self.revert_X():
                        self.setX(x)
                        return val(self) - x[x_inx]

            else:

                def fun(x):
                    with self.revert_X():
                        self.setX(x)
                        return x[x_inx] - val(self)

            cons = {"type": "ineq", "fun": fun}
            return cons


# Solver Attributes
class SOLVER_ATTR(attrs.Attribute):
    name: str
    on_system: "System"

    @classmethod
    def create_instance(cls, system: "System"):
        raise NotImplemented("need to implement on subclass")


# Solver minimizes residual by changing independents
class SolverInstance:
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    solver: "SOLVER"

    # compiled info
    dependent: "Ref"
    independent: "Ref"

    __slots__ = ["system", "solver", "dependent", "independent"]

    def __init__(self, solver: "SOLVER", system: "System") -> None:
        self.solver = solver
        self.system = system
        self.compile()

    def compile(self):
        self.dependent = self.system.locate_ref(self.solver.dependent)
        self.independent = self.system.locate_ref(self.solver.independent)
        self.system.info(f"solving {self.dependent} with {self.independent}")


class SOLVER(SOLVER_ATTR):
    """solver creates subclasses per solver balance"""

    dependent: str
    independent: str

    constraints: dict

    @classmethod
    def define(
        cls, dependent: "system_property", independent: "attrs.Attribute"
    ):
        """Defines a new dependent and independent variable for the system solver. The error term will not necessiarily be satisfied by changing this particular independent"""

        # Create A New Signals Class
        new_name = f"SOLVER_indep_{dependent}_{independent}".replace(".", "_")
        constraints = {"min": None, "max": None}
        new_dict = dict(
            name=new_name,
            dependent=dependent,
            independent=independent,
            constraints=constraints,
        )
        new_slot = type(new_name, (SOLVER,), new_dict)
        return new_slot

    @classmethod
    def configure_for_system(cls, name, system):
        """#TODO: add the system class, and check the dependent and independent values"""
        cls.name = name
        cls.on_system = system

    @classmethod
    def validate_parms(cls):
        from ottermatics.properties import system_property

        system = cls.on_system

        parm_type = system.locate(cls.independent)
        if parm_type is None:
            raise Exception(f"independent not found: {cls.independent}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.independent} not attribute: {parm_type}"
        assert parm_type.type in (
            int,
            float,
        ), f"bad parm {cls.independent} not numeric"

        driv_type = system.locate(cls.dependent)
        if parm_type is None:
            raise Exception(f"dependent not found: {cls.dependent}")
        assert isinstance(
            driv_type, system_property
        ), f"bad dependent {cls.dependent} type: {driv_type}"
        assert driv_type.return_type in (
            int,
            float,
        ), f"bad parm {cls.dependent} not numeric"

    @classmethod
    def make_solver_factory(cls):
        return attrs.Factory(cls.create_instance, takes_self=True)

    @classmethod
    def create_instance(cls, system: "System") -> SolverInstance:
        return SolverInstance(cls, system)

    @classmethod
    def add_constraint(cls, type_, value):
        """adds a `type` constraint to the solver. If value is numeric it is used as a bound with `scipy` optimize.

        If value is a function it should be of the form value(system) and will establish an inequality constraint that independent parameter must be:
            1. less than for max
            2. more than for min

        During the evaluation of the limit function system.X should be set, and pre_execute() have already run.

        :param type: str, must be either min or max
        :value: either a numeric (int,float), or a function, f(system)
        """
        assert cls is not SOLVER, f"must set constraint on SOLVER subclass"
        assert (
            cls.constraints[type_] is None
        ), f"existing constraint for {type in {cls.__name__}}"
        assert isinstance(value, (int, float)) or callable(
            value
        ), f"only int,float or callables allow. Callables must take system as argument"
        cls.constraints[type_] = value


# DYnamics
# Solver minimizes residual by changing independents
class IntegratorInstance:
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    transient: "TRANSIENT"

    # compiled info
    parameter: "Ref"
    derivative: "Ref"

    __slots__ = ["system", "solver", "parameter", "derivative"]

    # TODO: add forward implicit solver

    def __init__(self, solver: "TRANSIENT", system: "System") -> None:
        self.solver = solver
        self.system = system
        self.compile()

    def compile(self):
        self.parameter = self.system.locate_ref(self.solver.parameter)
        self.derivative = self.system.locate_ref(self.solver.derivative)
        self.system.info(f"integrating {self.parameter} with {self.derivative}")

    def integrate(self, dt):
        # TODO: support different integrator modes
        assert (
            self.solver.mode == "euler"
        ), "only euler integration supported currently"
        new_val = self.parameter.value() + self.derivative.value() * dt
        self.parameter.set_value(new_val)


class TRANSIENT(SOLVER_ATTR):
    """Transient is a base class for integrators over time"""

    mode: str
    parameter: str
    derivative: str

    @classmethod
    def define(
        cls,
        parameter: "attrs.Attribute",
        derivative: "system_property",
        mode: str = "euler",
    ):
        """Defines an ODE like integrator that will be integrated over time with the defined integration rule.

        Input should be of strings to look up the particular property or field
        """
        # Create A New Signals Class
        new_name = f"TRANSIENT_{mode}_{parameter}_{derivative}".replace(
            ".", "_"
        )
        new_dict = dict(
            mode=mode,
            name=new_name,
            parameter=parameter,
            derivative=derivative,
        )
        new_slot = type(new_name, (TRANSIENT,), new_dict)
        return new_slot

    @classmethod
    def configure_for_system(cls, name, system):
        """add the system class, and check the dependent and independent values"""
        cls.name = name
        cls.on_system = system

    @classmethod
    def validate_parms(cls):
        from ottermatics.properties import system_property

        system = cls.on_system

        parm_type = system.locate(cls.parameter)
        if parm_type is None:
            raise Exception(f"parameter not found: {cls.parameter}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.parameter} not attribute: {parm_type}"
        assert parm_type.type in (
            int,
            float,
        ), f"bad parm {cls.parameter} not numeric"

        driv_type = system.locate(cls.derivative)
        if parm_type is None:
            raise Exception(f"derivative not found: {cls.derivative}")
        assert isinstance(
            driv_type, system_property
        ), f"bad derivative {cls.derivative} type: {driv_type}"
        assert driv_type.return_type in (
            int,
            float,
        ), f"bad parm {cls.derivative} not numeric"

    @classmethod
    def make_solver_factory(cls):
        return attrs.Factory(cls.create_instance, takes_self=True)

    @classmethod
    def create_instance(cls, system: "System") -> IntegratorInstance:
        return IntegratorInstance(cls, system)
