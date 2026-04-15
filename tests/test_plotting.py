import matplotlib
import matplotlib.pyplot as plt
from openpytea import (
    direct_costs_data,
    sensitivity_data,
    tornado_data,
    plot_stacked_bar,
    plot_sensitivity,
    plot_tornado,
)
matplotlib.use("Agg")

plt.rcParams["text.usetex"] = False


def test_plot_stacked_bar_runs(test_plant):
    data = direct_costs_data(test_plant)
    ax = plot_stacked_bar(data, show=False)

    assert ax is not None
    assert hasattr(ax, "figure")


def test_plot_sensitivity_runs(test_plant):
    data = sensitivity_data(
        test_plant,
        parameter="interest_rate",
        plus_minus_value=0.2,
        n_points=5,
        metric="NPV",
    )
    ax = plot_sensitivity(data, show=False)

    assert ax is not None
    assert hasattr(ax, "figure")


def test_plot_tornado_runs(test_plant):
    data = tornado_data(
        test_plant,
        plus_minus_value=0.1,
        metric="NPV",
    )
    ax = plot_tornado(data, show=False)

    assert ax is not None
    assert hasattr(ax, "figure")
