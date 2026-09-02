import matplotlib
import matplotlib.pyplot as plt
from openpytea import (
    direct_costs_data,
    cash_flow_data,
    sensitivity_data,
    tornado_data,
    plot_stacked_bar,
    plot_cash_flow,
    plot_sensitivity,
    plot_tornado,
)
matplotlib.use("Agg")

plt.rcParams["text.usetex"] = False


def test_plot_stacked_bar_runs(test_plant):
    data = direct_costs_data(test_plant)
    fig, ax = plot_stacked_bar(data, show=False)

    assert fig is not None
    assert ax is not None


def test_plot_cash_flow_runs(test_plant):
    data = cash_flow_data(test_plant)
    fig, ax = plot_cash_flow(data, show=False)

    assert fig is not None
    assert ax is not None


def test_plot_cash_flow_multi_runs(test_plant, test_plant_b):
    data = cash_flow_data([test_plant, test_plant_b])
    fig, ax = plot_cash_flow(data, show=False)

    assert fig is not None
    assert ax is not None


def test_plot_sensitivity_runs(test_plant):
    data = sensitivity_data(
        test_plant,
        parameter="interest_rate",
        plus_minus_value=0.2,
        n_points=5,
        metric="NPV",
    )
    fig, ax = plot_sensitivity(data, show=False)

    assert fig is not None
    assert ax is not None


def test_plot_tornado_runs(test_plant):
    data = tornado_data(
        test_plant,
        plus_minus_value=0.1,
        metric="NPV",
    )
    fig, ax = plot_tornado(data, show=False)

    assert fig is not None
    assert ax is not None


def test_plot_tornado_height_grows_with_factor_count(test_plant):
    """A longer factor list gets a taller figure, not thinner bars."""
    short = tornado_data(test_plant, plus_minus_value=0.1, metric="LCOP")
    tall = tornado_data(test_plant, plus_minus_value=0.1, metric="LCOP",
                        include_process_params=True)
    assert len(tall["factors"]) > len(short["factors"])

    fig_short, _ = plot_tornado(short, show=False)
    fig_tall, _ = plot_tornado(tall, show=False)

    h_short = fig_short.get_size_inches()[1]
    h_tall = fig_tall.get_size_inches()[1]
    assert h_tall > h_short
    # Width is unaffected
    assert (fig_short.get_size_inches()[0]
            == fig_tall.get_size_inches()[0] == 3.4)


def test_plot_tornado_explicit_figsize_still_wins(test_plant):
    data = tornado_data(test_plant, plus_minus_value=0.1,
                        include_process_params=True)
    fig, _ = plot_tornado(data, figsize=(5.0, 5.0), show=False)

    assert tuple(fig.get_size_inches()) == (5.0, 5.0)


def test_input_histogram_grid_single_input():
    """A category with exactly one input must plot, not crash."""
    import numpy as np
    from openpytea.plotting import _plot_input_histogram_grid

    fig, axes = _plot_input_histogram_grid(
        {"Only input": np.random.default_rng(0).normal(5, 1, 200)},
        None, 30, "tab:blue", "Process parameters", show=False,
    )
    assert axes[0].has_data()
    # the two unused grid cells are turned off
    assert not axes[1].axison and not axes[2].axison
    plt.close(fig)


def test_tex_escape_follows_usetex():
    from openpytea.helpers import _tex_escape, _default_metric_label

    old = plt.rcParams["text.usetex"]
    try:
        plt.rcParams["text.usetex"] = False
        assert _tex_escape("%") == "%"
        assert "\%" not in _default_metric_label("USD", "roi")
        plt.rcParams["text.usetex"] = True
        assert _tex_escape("%") == r"\%"
        assert r"[\%]" in _default_metric_label("USD", "roi")
    finally:
        plt.rcParams["text.usetex"] = old


def test_tornado_legend_has_no_literal_backslash(test_plant):
    """With usetex off, legend labels must read -10%, not -10\%."""
    data = tornado_data(test_plant, plus_minus_value=0.1)
    fig, ax = plot_tornado(data, show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["-10%", "+10%"]
    plt.close(fig)
