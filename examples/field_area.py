from engforge import *

@forge
class SimpleField(System):

    x: float = field(default=1)
    y: float = field(default=1)
    budget: float = field(default=100)

    cost_per_x: float = field(default=2)
    cost_per_y: float = field(default=1)

    xvar = Solver.declare_var('x')
    xvar.add_var_constraint(0)
    yvar = Solver.declare_var('y')
    yvar.add_var_constraint(0)

    cost = Solver.constraint_inequality('remaining_budget',0)

    obj = Solver.objective('area',kind='max')
    perim = Solver.objective('perimeter',kind='max',combos='fence')

    @system_property
    def perimeter(self) -> float:
        return 2*self.x + 2*self.y

    @system_property
    def area(self) -> float:
        return self.x*self.y        

    @system_property
    def remaining_budget(self) -> float:
        return self.budget - self.cost_per_x*self.x - self.cost_per_y*self.y

if __name__ == '__main__':

    sf = SimpleField()
    sf.run(slv_vars='*')
    assert sf.x == 1 #keep original
    assert sf.y == 1 #keep original

    ans = sf.dataframe.iloc[0]
    assert abs(ans.x-25)<0.001
    assert abs(ans.y-50)<0.001
    assert abs(ans.area-1250)<0.001
    assert abs(ans.remaining_budget)<0.001
