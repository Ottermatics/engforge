
@forge
class SpringMass(System):
    
    k: float = attrs.field(default=50)
    m: float = attrs.field(default=1)
    g: float = attrs.field(default=9.81)
    u: float = attrs.field(default=0.3)

    a: float = attrs.field(default=0)
    x: float = attrs.field(default=0.0)
    v: float = attrs.field(default=0.0)

    wo_f: float = attrs.field(default=1.0)
    Fa: float = attrs.field(default=10.0)

    x_neutral: float = attrs.field(default=0.5)

    res =Solver.constraint_equality("sumF")
    var_a = Solver.declare_var("a",combos='a',active=False)
    var_b = Solver.declare_var("u",combos='u',active=False)
    var_b.add_var_constraint(0.0,kind="min")
    var_b.add_var_constraint(1.0,kind="max")

    vtx = Time.integrate("v", "accl")
    xtx = Time.integrate("x", "v")
    xtx.add_var_constraint(0,kind="min")

    pos = Trace.define(y="x", y2=["v", "a"])

    @system_property
    def dx(self) -> float:
        return self.x_neutral - self.x

    @system_property
    def Fspring(self) -> float:
        return self.k * self.dx

    @system_property
    def Fgrav(self) -> float:
        return self.g * self.m

    @system_property
    def Faccel(self) -> float:
        return self.a * self.m

    @system_property
    def Ffric(self) -> float:
        return self.u * self.v

    @system_property
    def sumF(self) -> float:
        return self.Fspring - self.Fgrav - self.Faccel - self.Ffric + self.Fext
    
    @system_property
    def Fext(self) -> float:
        return self.Fa * np.cos( self.time * self.wo_f )

    @system_property
    def accl(self) -> float:
        return self.sumF / self.m
    

#Run The System, Compare damping `u`=0 & 0.1
sm = SpringMass(x=0.0)
trdf = sm.simulate(dt=0.01,endtime=10,u=[0.0,0.1],combos='*',slv_vars='*')
trdf.groupby('run_id').plot('time','x')