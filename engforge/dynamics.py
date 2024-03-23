"""Combines the tabulation and component mixins to create a mixin for systems and components that have dynamics, such as state space models, while allowing nonlinear dynamics via matrix modification

This module is intended to work alongside the solver module and the Time integrating attributes, and will raise an error if a conflict is detected.

The DynamicsMixin works by establishing a state matricies A, B, C, and D, which are used to define the dynamics of the system. The state matrix A is the primary matrix, and is used to define the state dynamics of the system. The input matrix B is used to define the input dynamics of the system. The output matrix C is used to define the output dynamics of the system. The feedthrough matrix D is used to define the feedthrough dynamics of the system.

As opposed to the Time attribute, which modifies the state of the system, the DynamicsMixin copies the initial values of the state and input which then are integrated over time. At predefined intervals the control and output will stored in the tabulation.

#TODO: The top level system will collect the underlying dynamical systems and combine them to an index and overall state space model. This will allow for the creation of a system of systems, and the ability to create a system of systems with a single state space model.

#TODO: integration is done by the solver, where DynamicSystems have individual solver control, solver control is set for a smart default scipy 
"""

from engforge.configuration import Configuration, forge
from engforge.tabulation import TabulationMixin
from engforge import properties as prop
from engforge.attributes import ATTR_BASE
from engforge.properties import instance_cached, solver_cached
from engforge.system_reference import Ref
from engforge.problem_context import ProblemExec
from engforge.solveable import (
    SolveableMixin,
    refmin_solve,
    refset_get,
    refset_input,
)


from collections import OrderedDict
import numpy as np
import pandas
import expiringdict
import attr, attrs
from scipy.integrate import solve_ivp


# Index maps are used to translate between different indexes (local & global)

def valid_mtx(arr):
    if arr is None:
        return False
    elif arr.size == 0:
        return 0
    return True
class INDEX_MAP:
    oppo = {str: int, int: str}

    def __init__(self, datas: list):
        self.data = [
            data if not data.startswith(".") else data[1:] for data in datas
        ]
        self.index = {}

    def get(self, key):
        if key.startswith("."):
            key = key[1:]
        if key not in self.index:
            self.index[key] = self.data.index(key)
        return self.index[key]

    def __getitem__(self, key):
        if key.startswith("."):
            key = key[1:]
        return self.get(key)

    def __call__(self, key):
        if key.startswith("."):
            key = key[1:]
        return self.get(key)

    @staticmethod
    def indify(arr, *args):
        return [
            arr[arg] if isinstance(arg, int) else arr.index(arg) for arg in args
        ]

    def remap_indexes_to(self, new_index, *args, invert=False, old_data=None):
        if old_data is None:
            old_data = self.data
        opt1 = {arg: self.indify(old_data, arg)[0] for arg in args}
        opt2 = {
            arg: self.indify(old_data, val)[0]
            if (not isinstance(val, str))
            else val
            for arg, val in opt1.items()
        }
        oop1 = {
            arg: self.indify(new_index, val)[0] for arg, val in opt2.items()
        }
        oop2 = {
            arg: self.indify(new_index, val)[0]
            if (invert != isinstance(val, self.oppo[arg.__class__]))
            else val
            for arg, val in oop1.items()
        }
        return oop2


class DynamicsIntegratorMixin(SolveableMixin):
    nonlinear: bool = True  # enables matrix modification for nonlinear dynamics

    time: float = 0.0

    def rate(self, t, dt, X, U, *args, **kwargs):
        """simulate the system over the course of time.

        Args:
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input
            subsystems (bool, optional): simulate subsystems. Defaults to True.

        Returns:
            dataframe: tabulated data
        """
        if self.nonlinear:
            return self.rate_nonlinear(t, dt, X, U, *args, **kwargs)
        else:
            return self.rate_linear(t, dt, X, U, *args, **kwargs)

    def rate_linear(self, t, dt, X, U, *args, **kwargs):
        raise NotImplemented()

    def rate_nonlinear(self, t, dt, X, U, *args, **kwargs):
        raise NotImplemented()

    def reset_sim(self):
        """reset the system to the initial state"""
        self.time = 0.0



# Quickly create a state space model
# TODO: How to add delay, and feedback?
# TODO: How to add control and limits in general?
# TODO: add time as a state variable
@forge(auto_attribs=True)
class DynamicsMixin(Configuration, DynamicsIntegratorMixin):
    """dynamic mixin for components and systems that have dynamics, such as state space models, while allowing nonlinear dynamics via matrix modification. This mixin is intended to work alongside the solver module and the Time integrating attributes, and will raise an error if a conflict is detected #TODO."""

    dynamic_state_vars: list = attrs.field(factory=list)
    dynamic_input_vars: list = attrs.field(factory=list)
    dynamic_output_vars: list = attrs.field(factory=list)

    # state variables
    dynamic_A = None
    dynamic_B = None
    dynamic_C = None
    dynamic_D = None
    dynamic_F = None
    dynamic_K = None

    # Static linear state
    static_A= None
    static_B= None
    static_C= None
    static_D= None
    static_F= None
    static_K= None

    # TODO:
    # dynamic_control_module = None
    # control_interval:float = 0.0 ##TODO: how often to update the control

    update_interval: float = (
        0.0  # TODO: how often to update the physics, 0 is everytime
    )
    delay_ms: float = None  # TODO: delay in milliseconds for output buffering
    nonlinear: bool = False

    # TODO: add integration state dynamic cache to handle relative time steps
    # TODO: add control module with PID control and pole placement design

    def __pre_init__(self, **kwargs):
        """override this method to define the class"""
        # fields =
        fields = attrs.fields_dict(self.__class__)
        system_property = self.system_properties_def
        for p in self.dynamic_state_vars:
            assert p in fields, f"state var {p} not in attr: {fields}"
        for p in self.dynamic_output_vars:
            assert p in fields, f"output var {p} not in attr: {fields}"
        for p in self.dynamic_input_vars:
            assert p in fields, f"input var {p} not in attr: {fields}"

    @instance_cached
    def state_size(self):
        return len(self.dynamic_state_vars)

    @instance_cached
    def input_size(self):
        return len(self.dynamic_input_vars)

    @instance_cached
    def output_size(self):
        return len(self.dynamic_output_vars)

    @property
    def state(self) -> np.array:
        return np.array(
            [getattr(self, var, np.nan) for var in self.dynamic_state_vars]
        )

    @property
    def input(self) -> np.array:
        return np.array(
            [getattr(self, var, np.nan) for var in self.dynamic_input_vars]
        )

    @property
    def output(self) -> np.array:
        return np.array(
            [getattr(self, var, np.nan) for var in self.dynamic_output_vars]
        )


    def create_state_matrix(self, **kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.zeros((self.state_size, self.state_size))

    def create_state_input_matrix(self, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((self.state_size, max(self.input_size, 1)))

    def create_output_matrix(self, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called C"""
        return np.zeros((max(self.output_size, 1), self.state_size))

    def create_feedthrough_matrix(self, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called D"""
        return np.zeros((max(self.output_size, 1), max(self.input_size, 1)))

    def create_state_constants(self, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called F"""
        return np.zeros(self.state_size)

    def create_output_constants(self, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called O"""
        return np.zeros(max(self.output_size, 1))

    def create_dynamic_matricies(self, **kw):
        """creates a dynamics object for the system"""
        # State + Control
        self.static_A = self.create_state_matrix(**kw)
        self.static_B = self.create_state_input_matrix(**kw)
        # Output
        self.static_C = self.create_output_matrix(**kw)
        self.static_D = self.create_feedthrough_matrix(**kw)
        # Constants
        self.static_F = self.create_state_constants(**kw)
        self.static_K = self.create_output_constants(**kw)

    # Nonlinear Support
    # Override these callbacks to modify the state space model
    def update_state_nonlinear(self, t, A, X) -> np.ndarray:
        """override"""
        return A

    def update_input_nonlinear(self, t, B, X, U) -> np.ndarray:
        """override"""
        return B

    def update_output_matrix(self, t, C, X) -> np.ndarray:
        """override"""
        return C

    def update_feedthrough_nonlinear(self, t, D, X, U) -> np.ndarray:
        """override"""
        return D

    def update_state_constants(self, t, F, X) -> np.ndarray:
        """override"""
        return F

    def update_output_constants(self, t, O, X) -> np.ndarray:
        """override"""
        return O

    def update_dynamics(self, t, X, U):
        """Updates dynamics when nonlinear is enabled, otherwise it will do nothing"""
        if not self.nonlinear:
            return

        # State + Control
        self.dynamic_A = self.update_state_nonlinear(t, self.static_A, X)
        self.dynamic_B = self.update_input_nonlinear(t, self.static_B, X, U)

        # Output
        self.dynamic_C = self.update_output_matrix(t, self.static_C, X)
        self.dynamic_D = self.update_feedthrough_nonlinear(
            t, self.static_D, X, U
        )

        # Constants
        self.dynamic_F = self.update_state_constants(t, self.static_F, X)
        self.dynamic_K = self.update_output_constants(t, self.static_K, X)

    # linear and nonlinear system level IO
    def rate_linear(self, t, dt, X, U=None):
        """simulate the system over the course of time. Return time differential of the state."""
        
        O = 0
        if valid_mtx(self.static_A) and valid_mtx(X):
            O = self.static_A @ X

        if valid_mtx(U) and valid_mtx(self.static_B):
            O += self.static_B @ U

        if valid_mtx(self.static_F):
            O += self.static_F

        return O

    def linear_output(self, t, dt, X, U=None):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        O = 0
        if valid_mtx(self.static_C) and valid_mtx(X):
            O = self.static_C @ X

        if valid_mtx(U) and valid_mtx(self.static_D):
            O += self.static_D @ U

        if valid_mtx(self.static_K):
            O += self.static_K
        return O

    def rate_nonlinear(self, t, dt, X, U=None, update=True):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            t (float): time
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        if update:
            self.update_dynamics(t, X, U)
        
        O = 0
        if valid_mtx(self.dynamic_A) and valid_mtx(X):
            O = self.dynamic_A @ X

        if valid_mtx(U) and valid_mtx(self.dynamic_B):
            O += self.dynamic_B @ U

        if valid_mtx(self.dynamic_F):
            O += self.dynamic_F
        return O

    def nonlinear_output(self, t, dt, X, U=None, update=True):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        if update:
            self.update_dynamics(t, X, U)
        
        O = 0
        if valid_mtx(self.dynamic_C)and valid_mtx(X):
            O = self.dynamic_C @ X

        if valid_mtx(U) and valid_mtx(self.dynamic_D):
            O += self.dynamic_D @ U

        if valid_mtx(self.dynamic_K):
            O += self.dynamic_K
        return O

    # optimized convience funcitons
    def nonlinear_step(self, t, dt, X, U=None, set_Y=True):
        """Optimal nonlinear steps"""
        self.time = t
        self.update_dynamics(t, X, U)
        
        dXdt = self.rate_nonlinear(t, dt, X, U, update=False)
        out = self.nonlinear_output(t, dt, X, U, update=False)

        if set_Y:
            for i, p in enumerate(self.dynamic_output_vars):
                self.Yt_ref[p].set_value(out[p])

        return dXdt

    def linear_step(self, t, dt, X, U=None, set_Y=True):
        """Optimal nonlinear steps"""
        self.time = t
        self.update_dynamics(t, X, U)
        dXdt = self.rate_linear(t, dt, X, U)
        out = self.linear_output(t, dt, X, U)

        if set_Y:
            for i, p in enumerate(self.dynamic_output_vars):
                self.Yt_ref[p].set_value(out[p])

        return dXdt

    def step(self, t, dt, X, U=None, set_Y=True):
        if self.nonlinear:
            return self.nonlinear_step(t, dt, X, U, set_Y=set_Y)
        else:
            return self.linear_step(t, dt, X, U, set_Y=set_Y)

    #Solver Refs
    @property
    def Xt_ref(self):
        """alias for state values"""
        d = [(var, Ref(self, var)) for var in self.dynamic_state_vars]
        return OrderedDict(d)

    @property
    def Yt_ref(self):
        """alias for output values"""
        d = [(var, Ref(self, var)) for var in self.dynamic_output_vars]
        return OrderedDict(d)

    @property
    def Ut_ref(self):
        """alias for input values"""
        d = [(var, Ref(self, var)) for var in self.dynamic_input_vars]
        return OrderedDict(d)
    
    @property
    def dXtdt_ref(self):
        """a dictionary of state var rates"""
        d = [(var, self.ref_dXdt(var)) for var in self.dynamic_state_vars]
        return OrderedDict(d)

    @solver_cached
    def cache_dXdt(self):
        """caches the time differential of the state,
        uses current state of X and U to determine the dXdt
        """
        # TODO: gather timestep
        lt = getattr(self, "_last_cache_time", 0)
        dt = max(self.time - lt, 0)
        self.info(f"cache dXdt {self.time} {lt} {dt}")
        step = self.step(self.time, dt, self.state, self.input)
        self._last_cache_time = self.time
        return step

    def ref_dXdt(self, name: str):
        """returns the reference to the time differential of the state"""
        vars = self.dynamic_state_vars
        assert name in vars, f"name {name} not in state vars"
        inx = vars.index(name)
        accss = lambda comp: comp.cache_dXdt[inx]
        return Ref(self, name, accss)
    

    def determine_nearest_stationary_state(
        self, t=0, X=None, U=None
    ) -> np.ndarray:
        """determine the nearest stationary state"""

        if X is None:
            X = self.state
        if U is None:
            U = self.input

        if self.nonlinear:
            self.update_dynamics(t, X, U)
            Mb = self.dynamic_B @ U if self.input_size > 0 else 0
            Mx = self.dynamic_F + Mb
            return np.linalg.solve(self.dynamic_A, -Mx)

        # static state
        Mb = self.static_B @ U if self.input_size > 0 else 0
        Mx = Mb + self.static_F
        return np.linalg.solve(self.static_A, -Mx)

    def __hash__(self):
        return hash(id(self))


@forge(auto_attribs=True)
class GlobalDynamics(DynamicsMixin):
    """This object is inherited by configurations that collect other dynamicMixins and orchestrates their simulation, and steady state analysis

    #TODO: establish bounds in solver
    """

    def setup_global_dynamics(self, **kwargs):
        """recursively creates numeric matricies for the simulation"""
        for skey,lvl,conf in self.go_through_configurations():
            if isinstance(conf,DynamicsMixin):
                conf.create_dynamic_matricies(**kwargs)

    

    def sim_matrix(self, eval_kw=None, sys_kw=None, *args, **kwargs):
        """simulate the system over the course of time.
        return a dictionary of dataframes
        """
        tr_opts = self.parse_simulation_input(**kwargs)
        from engforge.solver import SolveableMixin

        dt = tr_opts["dt"]
        endtime = tr_opts["endtime"]

        if isinstance(self, SolveableMixin):
            sim = lambda *args, **kw: self.simulate(
                dt, endtime, eval_kw=eval_kw, sys_kw=sys_kw, *args, **kw
            )
            self._iterate_input_matrix(
                sim,
                dt,
                endtime,
                eval_kw=eval_kw,
                sys_kw=sys_kw,
                return_results=True,
                *args,
                **kwargs,
            )
        else:
            self.simulate(dt, endtime, eval_kw=eval_kw, sys_kw=sys_kw)

    def simulate(
        self,
        dt,
        endtime,
        X0=None,
        cb=None,
        eval_kw=None,
        sys_kw=None,
        min_kw=None,
        run_solver=True,
        return_system=False,
        return_data=False,
        return_all=False,
        **kwargs,
    ) -> pandas.DataFrame:
        """runs a simulation over the course of time, and returns a dataframe of the results.

        A copy of this system is made, and the simulation is run on the copy, so as to not affect the state of the original system.
        """
        min_kw_dflt = {"method": "SLSQP"}
        #'tol':1e-6,'options':{'maxiter':100}}
        # min_kw_dflt = {'doset':True,'reset':False,'fail':True}
        if min_kw is None:
            min_kw = min_kw_dflt
        else:
            min_kw_dflt.update(min_kw)
        mkw = min_kw_dflt

        # Data Storage {time: data}
        data = {}  # output storage
        solver = expiringdict.ExpiringDict(100, 600)  # temp solver storage
        Time = np.arange(0, endtime + dt, dt)

        max_step_dt = kwargs.get("max_step", dt)


        # Initial State
        data = {}

        #TODO: put copy system in problem context
        system = self.copy_config_at_state()

        #recursively initalize transient components
        system.setup_global_dynamics()

        #Time Iteration Context
        with ProblemExec(system,level_name='sim',dxdt=True,**kwargs) as pbx:

            #Unpack Transient Problem
            intl_refs = pbx.integrator_var_refs
            refs = pbx.sys_refs

            if self.log_level < 10:
                self.debug(f'initial state {X0} {intl_refs}| {refs}')

            x_cur = X0 = {k: v.value() for k, v in intl_refs.items()}
            if X0 is None:
                # get current
                X0 = x_cur
            
            #this will fail if X0 doesn't have solver vares
            X0 = np.array([X0[p] for p in intl_refs])

            #Gather data
            Xss = refs['attrs'].get("solver.var",{})
            Xpm = refs['attrs'].get('time.var',{})
            Xdy = refs['attrs'].get('dynamics.state',{})

            Yobj = refs['attrs'].get("solver.obj",{})
            Yeq = refs['attrs'].get("solver.eq",{})
            Yineq=refs['attrs'].get("solver.ineq",{})
            Ydn = refs['attrs'].get("dynamics.output",{})

            dXtmdt = refs['attrs'].get('time.rate',{})
            dXdt = refs['attrs'].get("dynamics.rate",{})

            #TODO: handle various cases of (obj,eq,ineq)
            if not Yobj and Yeq:
                #TODO: handle case of Yineq == None: root solve
                self.info(f'making Yobj from Yeq: {Yeq}')
                Yobj = { k: v.copy(eval_f = lambda v:1+v**2) 
                            for k,v in Yeq.items() }
            elif not Yobj:
                #minimize the product of all vars, so the smallest value is the best that satisfies all constraints
                self.info(f'making Yobj from X: {Xss}')
                dflt = lambda sys,prb: (np.product(1+v.value()**2 for v in pbx.problem_vars.items()))**0.5
                Yobj = {'smallness': Ref(system, dflt)}

            #TODO: add simulation time to context
            pbx.last_time = 0

            #our anonymous integrator!
            def sim_iter(t, x, *args):
                out = {p: np.nan for p in intl_refs}
                Xin = {p: x[i] for i, p in enumerate(intl_refs)}

                if self.log_level < 10:
                    self.info(f'sim_iter {t} {x} {Xin}')

                system.time = t

                with ProblemExec(system,level_name='tr_slvr',Xnew=Xin) as pbx:

                    # solver always gets a copy of x
                    solver[t] = x #TODO: stateful capture
                    
                    # test for record time 
                    #TODO: rates / events ect 
                    #TODO: faster way to get last time vs max(data)  
                    if not data or t > pbx.last_time + dt:
                        data[t] = cs = pbx.get_ref_values()
                        pbx.last_time = t
                        if self.log_level < 10:
                            self.info(f'record {t}| {Xin}')

                    #ad hoc time integration
                    for name, trdct in pbx.integrators.items():
                        out[trdct.var] = trdct.current_rate

                    # dynamics
                    for compnm, compdict in pbx.dynamic_comps.items():
                        comp = compdict#["comp"]
                        if not comp.dynamic_state_vars and not comp.dynamic_input_vars:
                            continue
                        Xds = np.array([r.value() for r in comp.Xt_ref.values()])
                        Uds = np.array([r.value() for r in comp.Ut_ref.values()])
                        # time updated in step
                        #system.info(f'comp {comp} {compnm} {Xds} {Uds}')
                        dxdt = comp.step(t, dt, Xds, Uds, True)

                        for i, (p, ref) in enumerate(comp.Xt_ref.items()):
                            out[(f"{compnm}." if compnm else "") + p] = dxdt[i]

                    # solvers
                    if run_solver and pbx.solveable:
                        # TODO: add in any transient
                        with ProblemExec(system,level_name='ss_slvr') as pbx:
                            
                            #TODO: handle various cases of (obj,eq,ineq)
                            #normally obj is handled via min/max objective
                            #in case of eq and inequal constraints, we need to adapt to the situation. For many eq's we can use the root solver, otherwise we can mutiply the normalize the equalities all together as multiply by something like normal of all inputs to provide a default objective of something like "find the smallest input satisfying constraints"


                            ss_out = pbx.solve_min(Xss, Yobj, **mkw)
                            if ss_out['ans'].success:
                                pbx.set_ref_values(ss_out['Xans'])
                                pbx.exit_to_level('ss_slvr',False)
                            else:
                                #TODO: handle failure options
                                if pbx.raise_on_opt_failure:
                                    pbx.exit_to_level('sim',pbx.fail_revert)
                                else:
                                    pbx.exit_to_level('ss_slvr',pbx.fail_revert)

                    #pbx.post_execute()
                    V_dxdt =  np.array([out[p] for p in intl_refs])
                    
                    pbx.exit_to_level('tr_slvr',False)

                if any( np.isnan(V_dxdt) ):
                    pbx.exit_and_revert()
                    #TODO: handle this better, seems to cause a warning
                    raise ValueError(f'infeasible! nan result {V_dxdt} {out} {Xin} {cs}')

                return V_dxdt

        
            ans = solve_ivp(sim_iter, [0, endtime], X0, method="RK45", t_eval=Time, max_step=max_step_dt)

        # convert to list with time
        data = [{"time": k, **v} for k, v in data.items()]

        df = pandas.DataFrame(data)
        self.format_columns(df)
        if return_all:
            return system,(data if return_data else df)
        if return_system:
            return system        
        if return_data:
            return data        
        return df

    def data_dict_tm(self, time, **kw):
        """returns the data dictionary"""
        dd = self.data_dict
        dd["time"] = time
        dd.update(kw)  # TODO: check for typing
        # print(dd)
        return dd

