from engforge.analysis import Analysis
from engforge.reporting import CSVReporter, DiskPlotReporter
from engforge.properties import system_property
from engforge import *

from matplotlib.pylab import *

import numpy as np
import os, pathlib
import attrs


@forge
class Fan(Component):
    n_frac: float = field(default=1)
    dp_design: float = field(default=100)
    w_design: float = field(default=2)

    @system_property
    def dP_fan(self) -> float:
        return self.dp_design * (self.n_frac * self.w_design) ** 2.0


@forge
class Filter(Component):
    w: float = field(default=0)
    k_loss: float = field(default=50)

    @system_property
    def dP_filter(self) -> float:
        return self.k_loss * self.w


@forge
class Airfilter(System):
    throttle: float = field(default=1)
    w: float = field(default=1)
    k_parasitic: float = field(default=0.1)

    fan: Fan = Slot.define(Fan)
    filt: Filter = Slot.define(Filter)

    set_fan_n = Signal.define("fan.n_frac", "throttle", mode="both")
    set_filter_w = Signal.define("filt.w", "w", mode="both")

    flow_var = Solver.declare_var("w", combos="flow")
    flow_var.add_var_constraint(0, "min", combos="flow")

    pr_eq = Solver.constraint_equality("sum_dP", 0, combos="flow")

    flow_curve = Plot.define(
        "throttle", "w", kind="lineplot", title="Flow Curve"
    )

    @system_property
    def dP_parasitic(self) -> float:
        return self.k_parasitic * self.w**2.0

    @system_property
    def sum_dP(self) -> float:
        return self.fan.dP_fan - self.dP_parasitic - self.filt.dP_filter


@forge
class AirfilterAnalysis(Analysis):
    """Does post processing on a system"""

    efficiency = attrs.field(default=0.95)

    @system_property
    def clean_air_delivery_rate(self) -> float:
        return self.system.w * self.efficiency

    def post_process(self, *run_args, **run_kwargs):
        pass
        # TODO: something custom!


if __name__ == "__main__":
    this_dir = str(pathlib.Path(__file__).parent)
    this_dir = os.path.join(this_dir, "airfilter_report")
    if not os.path.exists(this_dir):
        os.makedirs(this_dir, 0o754, exist_ok=True)

    csv = CSVReporter(path=this_dir, report_mode="daily")
    csv_latest = CSVReporter(path=this_dir, report_mode="single")

    plots = DiskPlotReporter(path=this_dir, report_mode="monthly")
    plots_latest = DiskPlotReporter(path=this_dir, report_mode="single")

    fan = Fan()
    filt = Filter()
    af = Airfilter(fan=fan, filt=filt)
    change_all_log_levels(af, 20)  # info

    # Make The Analysis
    sa = AirfilterAnalysis(
        system=af,
        table_reporters=[csv, csv_latest],
        plot_reporters=[plots, plots_latest],
    )

    # Run the analysis! Input passed to system
    sa.run(throttle=list(np.arange(0.1, 1.1, 0.1)), combos="*")

    # CSV's & Plots available in ./airfilter_report!
    af.run(throttle=list(np.arange(0.1, 1.1, 0.1)), combos="*")

    df = af.dataframe

    fig, (ax, ax2) = subplots(2, 1)
    ax.plot(df.throttle * 100, df.w, "k--", label="flow")
    ax2.plot(df.throttle * 100, df.filt_dp_filter, label="filter")
    ax2.plot(df.throttle * 100, df.dp_parasitic, label="parasitic")
    ax2.plot(df.throttle * 100, df.fan_dp_fan, label="fan")
    ax.legend(loc="upper right")
    ax.set_title("flow")
    ax.grid()
    ax2.legend()
    ax2.grid()
    ax2.set_title(f"pressure")
    ax2.set_xlabel(f"throttle%")
