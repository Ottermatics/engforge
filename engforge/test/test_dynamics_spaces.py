from engforge.test.solver_testing_components import *

import numpy as np
import unittest


class TestDynamics(unittest.TestCase):

    def test_dynamics(self,endtime=10,dt=0.01):
        dc = DynamicComponent()
        ds = DynamicSystem(comp=dc)
        ds.setup_global_dynamics()
        
        # ds.update_dynamics()
        ds.collect_dynamic_refs()
        # ds2 = ds.copy_config_at_state()
        sr = ds.collect_solver_refs()

        min_kw = {"normalize": np.array([1 / 1000])}
        sim, df = ds.simulate(dt, endtime, run_solver=False, return_all=True)
        
        # ax = df.plot("time", ["x", "comp_x", "trns_x", "comp_x0"])
        # # ax.set_ylim(-1,5)
        # ax2 = ax.twinx()
        # ax2.plot(df.time, df.a, "r--", label="acl")
        # ax2.plot(df.time, df["trns_a"], "b--", label="trans_acl")
        # ax2.plot(df.time, df["spring_accel"], "c--", label="spring_acl")
        # ax2.legend()

        self.assertGreaterEqual(df["time"].max(), endtime-dt)