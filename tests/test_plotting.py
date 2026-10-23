import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
                        category="both")
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
                        category="both")
    fig, _ = plot_tornado(data, figsize=(5.0, 5.0), show=False)

    assert tuple(fig.get_size_inches()) == (5.0, 5.0)


def test_input_histogram_grid_single_input():
    """A category with exactly one input must plot, not crash."""
    import numpy as np
    from openpytea.plotting import _plot_input_histogram_grid

    fig, axes = _plot_input_histogram_grid(
        {"Only input": np.random.default_rng(0).normal(5, 1, 200)},
        None, 30, "tab:blue", "Process parameters",
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
        assert r"\%" not in _default_metric_label("USD", "roi")
        plt.rcParams["text.usetex"] = True
        assert _tex_escape("%") == r"\%"
        assert r"[\%]" in _default_metric_label("USD", "roi")
    finally:
        plt.rcParams["text.usetex"] = old


def test_sensitivity_xlabel_has_no_literal_backslash(test_plant):
    r"""With usetex off, the x label must read [$\pm$ %], not [$\pm$ \%]."""
    for parameter in ("interest_rate",
                      "variable_opex_inputs.electricity",
                      "variable_opex_inputs.electricity.consumption"):
        data = sensitivity_data(test_plant, parameter, 0.2, n_points=3)
        assert data["xlabel"].endswith(r"[$\pm$ %]")


def test_tornado_legend_has_no_literal_backslash(test_plant):
    r"""With usetex off, legend labels must read -10%, not -10\%."""
    data = tornado_data(test_plant, plus_minus_value=0.1)
    fig, ax = plot_tornado(data, show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["-10%", "+10%"]
    plt.close(fig)


def test_plot_cash_flow_accepts_round_tripped_results(test_plant):
    """Results reloaded from JSON arrive as lists, not arrays."""
    import json

    data = cash_flow_data(test_plant)
    round_tripped = json.loads(
        json.dumps(data, default=lambda o: o.tolist())
    )
    assert isinstance(
        round_tripped["curves"][0]["cumulative"], list
    )

    fig, ax = plot_cash_flow(round_tripped, show=False)
    assert ax.has_data()
    plt.close(fig)


def test_plot_tornado_without_plus_minus_value(test_plant):
    """Documented-optional key: neutral legend, no crash."""
    data = tornado_data(test_plant, plus_minus_value=0.1)
    data.pop("plus_minus_value", None)
    fig, ax = plot_tornado(data, show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Low", "High"]
    plt.close(fig)


def test_plot_tornado_fractional_percentage_label(test_plant):
    data = tornado_data(test_plant, plus_minus_value=0.075)
    fig, ax = plot_tornado(data, show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["-7.5%", "+7.5%"]
    plt.close(fig)


def test_plot_multiple_monte_carlo_returns_live_figure(test_plant, test_plant_b):
    from openpytea import monte_carlo, plot_multiple_monte_carlo

    r1 = monte_carlo(test_plant, num_samples=100, batch_size=100, random_seed=1)
    r2 = monte_carlo(test_plant_b, num_samples=100, batch_size=100, random_seed=1)

    fig, _ = plot_multiple_monte_carlo([r1, r2], metric="LCOP", show=False)
    # the returned figure must still be registered with pyplot
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_plot_monte_carlo_sigma_label_uses_times():
    """A sigma outside [1, 10) is labelled as mantissa x 10^exp."""
    from openpytea.plotting import plot_monte_carlo

    values = np.random.default_rng(0).normal(500, 50, 1000)
    fig, ax = plot_monte_carlo(values, metric="LCOP", show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert r"$\times 10^{1}$" in labels[1]
    plt.close(fig)


def test_format_scripts():
    from openpytea.helpers import _format_scripts, _make_label

    assert _format_scripts("CO_2") == r"CO$_{\mathrm{2}}$"
    assert _format_scripts("T_{in}") == r"T$_{\mathrm{in}}$"
    assert _format_scripts("Fe^{3+}") == r"Fe$^{\mathrm{3+}}$"
    assert _format_scripts("x_1^2") == r"x$_{\mathrm{1}}^{\mathrm{2}}$"
    assert _format_scripts("my_plant") == "my_plant"  # usetex off here
    # Math and escaped currency signs are left alone; repeat is a no-op
    label = r"Cost / [\$$\cdot$unit$^{-1}$]"
    assert _format_scripts(label) == label
    assert _format_scripts(_format_scripts("H_2")) == _format_scripts("H_2")
    assert _make_label("H_2_price") == r"H$_{\mathrm{2}}$ price"
    assert _make_label("interest_rate") == "Interest rate"


def test_plot_names_with_scripts_render(test_plant):
    test_plant.name = "H_2 plant^{2}"
    test_plant.plant_products["H_2"] = test_plant.plant_products.pop(
        "hydrogen"
    )
    data = direct_costs_data(test_plant)
    fig, ax = plot_stacked_bar(data, show=False)
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_text() == (
        r"H$_{\mathrm{2}}$ plant$^{\mathrm{2}}$"
    )
    plt.close(fig)


def test_plot_sensitivity_x_axis(test_plant, test_plant_b):
    import pytest

    data = sensitivity_data(test_plant, "plant_products.hydrogen", 0.2,
                            n_points=3, metric="NPV")
    fig, ax = plot_sensitivity(data, show=False, x_axis="value")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [4.0, 5.0, 6.0])
    assert ax.get_xlabel().startswith("Hydrogen price")
    plt.close(fig)

    fig, ax = plot_sensitivity(data, show=False)  # default stays percent
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [-20.0, 0.0, 20.0])
    plt.close(fig)

    with pytest.raises(ValueError, match="x_axis"):
        plot_sensitivity(data, show=False, x_axis="values")

    test_plant_b.variable_opex_inputs["steam"] = {"consumption": 1, "price": 1}
    data = sensitivity_data([test_plant, test_plant_b],
                            "variable_opex_inputs.steam", 0.2, n_points=3)
    with pytest.warns(UserWarning, match="leaving it out"):
        fig, ax = plot_sensitivity(data, show=False, x_axis="value")
    assert len(ax.lines) == 1
    plt.close(fig)
