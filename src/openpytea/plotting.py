from itertools import cycle
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from openpytea.helpers import _default_metric_label

plt.style.use(["science", "ieee"])
_ = scienceplots  # mark as used for Flake8
cmap = plt.cm.plasma


# PLOTTING TOOLS
# ======================================================
def plot_stacked_bar(data, figsize=(1.2, 1.8), ax=None, show=True):
    """
    Create a stacked bar chart with sorted components.
    This function generates a stacked bar chart where components are sorted
    by their total values across all bars in descending order. Components are
    color-coded and displayed with a legend.
    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
        - "values" : list of lists
            List of lists where each inner list contains values for a component
            across all bars.
        - "labels" : tuple or list
            Labels for each component/stack in the bar chart.
        - "xlabels" : list
            Labels for the x-axis (one per bar).
        - "currency" : str
            Currency symbol or unit to display in y-axis label
            (ignored if pct=True).
        - "pct" : bool
            If True, values are interpreted as percentages and displayed
            accordingly.
        - "ylabel" : str
            Label for the y-axis.
    figsize : tuple of float, optional
        Figure size as (width, height) in inches. Default is (1.2, 1.8).
        The width is automatically adjusted based on the number of bars.
    ax : matplotlib.axes.Axes, optional
        Existing axes object to plot on. If None, a new figure and axes are
        created. Default is None.
    show : bool, optional
        If True and a new figure is created, display the plot. Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the stacked bar chart.
    Notes
    -----
    - Components are sorted in descending order by their total value across
        all bars.
    - Bar spacing is set to 0.75 and bar width is 0.45.
    - Colors are automatically assigned from a colormap and consistent across
        bars.
    - The legend is positioned to the right of the plot area.
    - When pct=True and n_bars=1, the percentage value is appended to the
        component label.
    """

    values = data["values"]
    labels = data["labels"]
    xlabels = data["xlabels"]
    currency = data["currency"]
    pct = data["pct"]

    # --- NEW: sort components ---
    # sum each component across all bars
    component_totals = np.sum(values, axis=0)

    n_bars = len(values)

    # get sorted indices (small -> large)
    sorted_idx = np.argsort(component_totals)[::-1]

    # reorder labels and values
    labels_sorted = [labels[0][i] for i in sorted_idx]
    values_sorted = [[v[i] for i in sorted_idx] for v in values]

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)

    spacing = 0.75  # < 1.0 pulls bars together
    bar_width = 0.45
    x = np.arange(n_bars) * spacing
    bottoms = np.zeros(n_bars, dtype=float)

    # --- Ax/fig handling ---
    created_fig = None
    if ax is None:
        if (
            isinstance(figsize, (tuple, list))
            and len(figsize) == 2
        ):
            base_w, base_h = figsize
        else:
            base_w, base_h = 1.2, 1.8
        auto_width = max(base_w * n_bars, base_w)
        created_fig, ax = plt.subplots(
            figsize=(auto_width, base_h)
        )

    colors = [cmap(i) for i in np.linspace(0.15, 0.95, len(labels_sorted))]
    color_map = dict(zip(labels_sorted, colors))

    if pct:
        ax.set_ylabel(data["ylabel"] + r" / [\%]")
        plot_labels = [
            rf"{lab} ({values_sorted[0][i]:.1f}\%)" if n_bars == 1 else lab
            for i, lab in enumerate(labels_sorted)
        ]
    else:
        ax.set_ylabel(data["ylabel"] + " / [" + currency + "]")
        plot_labels = labels_sorted

    # --- use sorted data ---
    for i in range(len(labels_sorted)):
        vals = [v[i] for v in values_sorted]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=bar_width,
            label=plot_labels[i],
            color=color_map[labels_sorted[i]],
            edgecolor="black",
            linewidth=0.3,
        )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    left = x[0] - bar_width / 2
    right = x[-1] + bar_width / 2
    ax.set_xlim(left - 0.2, right + 0.2)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="x-small",
        frameon=False,
    )

    if show and created_fig is not None:
        plt.show()

    return ax


def plot_sensitivity(data, figsize=(3.2, 2.2), ax=None, show=True):
    """
    Plot sensitivity analysis curves.
    Parameters
    ----------
    data : dict
        Dictionary containing plot data with the following keys:
        - 'curves' : list of dict
            List of curve dictionaries, each containing:
            - 'x' : array-like
                X-axis values for the curve
            - 'y' : array-like
                Y-axis values for the curve
            - 'plant' : str
                Label/name for the curve legend
        - 'xlabel' : str
            Label for the x-axis
        - 'ylabel' : str
            Label for the y-axis
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (3.2, 2.2).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plotted sensitivity curves.
    Notes
    -----
    - Multiple curves are plotted with colors from the Set2 colormap.
    - If a new figure is created, tight_layout() is applied before showing.
    - If an existing axes is provided, canvas is redrawn instead of calling
        plt.show().
    """

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    line_colors = cycle(plt.cm.Set2.colors)

    for curve in data["curves"]:
        ax.plot(
            curve["x"],
            curve["y"],
            linewidth=1.2,
            color=next(line_colors),
            label=curve["plant"],
            ls="-"
        )

    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.legend(loc="best")

    if created_fig is not None:
        created_fig.tight_layout()
        if show:
            plt.show()
    else:
        if show:
            ax.figure.canvas.draw_idle()

    return ax


def plot_tornado(
    data,
    figsize=(3.4, 2.4),
    ax=None,
    show=True,
):
    """
    Create a tornado plot visualization for sensitivity analysis.
    A tornado plot displays the impact of parameter variations on a base value,
    showing positive and negative deviations as horizontal bars.
    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
        - 'lows' : array-like
            Lower bound values for each parameter.
        - 'highs' : array-like
            Upper bound values for each parameter.
        - 'base_value' : float
            Reference/baseline value for comparison.
        - 'labels' : list of str
            Parameter names for y-axis labels.
        - 'xlabel' : str, optional
            Label for x-axis. If None, no label is set.
        - 'plus_minus_value' : float, optional
            Percentage variation value displayed in legend (e.g., 0.1 for 10%).
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (3.4, 2.4).
    ax : matplotlib.axes.Axes, optional
        Existing axes object to plot on. If None, a new figure and axes are
        created. Default is None.
    show : bool, optional
        If True and a new figure is created, display the plot using plt.show().
        Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the tornado plot.
    Notes
    -----
    - Blue bars represent negative deviations (lows from base_value).
    - Red bars represent positive deviations (highs from base_value).
    - A dashed black line indicates the base_value on the x-axis.
    - Automatic padding is applied to x-axis limits (5% of span or range).
    """
    lows = data["lows"]
    highs = data["highs"]
    base_value = data["base_value"]
    labels_sorted = data["labels"]

    xlabel = data.get("xlabel")
    pm = data.get("plus_minus_value")

    y_pos = np.arange(len(labels_sorted))

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(y_pos)):
        ax.barh(
            y_pos[i],
            abs(lows[i] - base_value),
            left=min(base_value, lows[i]),
            color="#87CEEB",
            edgecolor="black",
            linewidth=0.75,
            label=(rf"-{int(pm * 100)}\%" if i == 0 else ""),
        )

        ax.barh(
            y_pos[i],
            abs(highs[i] - base_value),
            left=min(base_value, highs[i]),
            color="#FF9999",
            edgecolor="black",
            linewidth=0.75,
            label=(rf"+{int(pm * 100)}\%" if i == 0 else ""),
        )

    ax.axvline(base_value, color="black", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.legend(loc="best")

    # x-limits with padding
    x_all = np.concatenate(
        [
            lows,
            highs,
            np.atleast_1d(base_value),
        ]
    )
    xmin, xmax = float(x_all.min()), float(x_all.max())

    if xmin == xmax:
        pad = 0.05 * (1.0 if xmax == 0 else abs(xmax))
        left_lim, right_lim = xmin - pad, xmax + pad
    else:
        span = xmax - xmin
        pad = 0.05 * span
        left_lim, right_lim = xmin - pad, xmax + pad

    ax.set_xlim(left_lim, right_lim)

    if created_fig is not None:
        created_fig.tight_layout()
        if show:
            plt.show()

    return ax


def plot_monte_carlo(
    data,
    metric: str = None,
    bins: int = 30,
    label: str | None = None,
    figsize=(3.2, 2.2),
    ax=None,
    show: bool = True,
):
    """
    Plot a histogram with a fitted normal distribution overlay for Monte Carlo
    simulation results. This function visualizes Monte Carlo data by
    displaying a histogram of sample values overlaid with a normal
    distribution curve fitted to the data. It supports multiple input types:
    Monte Carlo data dictionaries, Plant objects with monte_carlo_metrics,
    or raw numpy arrays.
    Parameters
    ----------
    data : dict, Plant, list of Plant, or array-like
        The Monte Carlo data to plot. Can be:
        - A dictionary containing a "metrics" key with metric values
        - A Plant object with a monte_carlo_metrics attribute
        - A list or tuple of Plant objects
        - A raw array-like of numeric values
    metric : str, optional
        The name of the metric to plot (case-insensitive). Default is "LCOP".
        Must be present in the data's available metrics.
    bins : int, optional
        Number of histogram bins. Default is 30.
    label : str or None, optional
        X-axis label. If None, a default label is generated based on the metric
        and currency. Default is None.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (3.2, 2.2).
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axes to plot on. If None, a new figure and axes
        are created. Default is None.
    show : bool, optional
        Whether to display the plot using plt.show(). Only applies if a new
        figure was created. Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    Raises
    ------
    ValueError
        If the specified metric is not found in the input data's available
        metrics.
    Notes
    -----
    - The normal distribution parameters (μ, σ) are fitted to the data using
    scipy.stats.norm.fit()
    - The standard deviation is formatted in scientific notation if
    |σ|>= 1000 or < 0.001
    - The histogram is semi-transparent (alpha=0.6) with black edges for
    better visibility
    - Colors are cycled from matplotlib's tab10 colormap for histogram and
    distribution curve
    """
    # --- Accept both Monte Carlo data dict and array ---
    # --- Normalize metric ---
    if metric is None:
        metric = "LCOP"
    metric = metric.upper()

    # --- Case 1: Monte Carlo data dict ---
    if isinstance(data, dict) and "metrics" in data:

        if metric not in data["metrics"]:
            available = ", ".join(data["metrics"].keys())
            raise ValueError(
                f"Metric '{metric}' not found. Available: {available}"
            )

        values = np.asarray(data["metrics"][metric], dtype=float)

        if label is None:
            currency = data.get("currency", r"\$")
            label = _default_metric_label(currency, metric)

    # --- Case 2: Plant object(s) ---
    elif hasattr(data, "monte_carlo_metrics") or (
        isinstance(data, (list, tuple)) and all(
            hasattr(p, "monte_carlo_metrics") for p in data
        )
    ):

        plants = data if isinstance(data, (list, tuple)) else [data]

        values_list = []
        currencies = []

        for plant in plants:
            if metric not in plant.monte_carlo_metrics:
                available = ", ".join(plant.monte_carlo_metrics.keys())
                raise ValueError(
                    f"Metric '{metric}' not found in "
                    f"plant.monte_carlo_metrics. "
                    f"Available: {available}"
                )

            values_list.append(
                np.asarray(plant.monte_carlo_metrics[metric], dtype=float)
            )
            currencies.append(getattr(plant, "currency", r"\$"))

        values = np.concatenate(values_list)

        if label is None:
            currency = currencies[0] if currencies else r"\$"
            label = _default_metric_label(currency, metric)

    # --- Case 3: Raw array ---
    else:
        values = np.asarray(data, dtype=float)

        if label is None:
            label = _default_metric_label(r"\$", metric)

    mu, std = norm.fit(values)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    hist_color = next(cycle(plt.cm.tab10.colors))
    line_color = next(cycle(plt.cm.tab10.colors))

    ax.hist(
        values,
        bins=bins,
        density=True,
        color=hist_color,
        edgecolor="black",
        alpha=0.6,
        zorder=1,
        label="Samples",
    )

    x = np.linspace(values.min(), values.max(), 1000)
    p = norm.pdf(x, mu, std)

    if std > 0:
        x = np.linspace(values.min(), values.max(), 1000)
        p = norm.pdf(x, mu, std)

        std_exp = int(np.floor(np.log10(std)))

        if std_exp == 0:
            stat_label = rf"$\mu$={mu:.3g}, $\sigma$={std:.3g}"
        else:
            std_mant = std / 10**std_exp
            stat_label = (
                rf"$\mu$={mu:.3g}, "
                rf"$\sigma$={std_mant:.2f}$\times 10^{{{std_exp}}}$")

        ax.plot(
                x,
                p,
                color=line_color,
                linewidth=1.2,
                zorder=2,
                linestyle="-",
                label=stat_label,
            )
    else:
        stat_label = rf"$\mu$={mu:.3g}, $\sigma$={std:.3g}"
        ax.axvline(
            mu,
            color=line_color,
            linewidth=1.2,
            zorder=2,
            linestyle="-",
            label=stat_label,
        )

    if created_fig is not None and show:
        created_fig.tight_layout()
        plt.show()

    return ax


def plot_monte_carlo_inputs(
    data,
    figsize=None,
    bins: int = 50,
    show: bool = True,
):
    """
    Plot histograms of Monte Carlo input parameters.
    This function creates a grid of histograms visualizing the distribution of
    input parameters from a Monte Carlo simulation. Each parameter is
    displayed in its own subplot, arranged in a grid layout with 3 columns.
    Parameters
    ----------
    data : dict or dict-like
        Input data containing Monte Carlo parameters. Can be either:
        - A dictionary with an "inputs" key containing the parameters dict
        - A dictionary directly mapping parameter names to arrays of values
    inputs : dict
        Dictionary where keys are parameter names (str) and values are array
        like objects containing the parameter samples.
    figsize : tuple of (float, float), optional
        Figure size as (width, height) in inches. If None, automatically
        calculated based on number of parameters (default: None).
    bins : int, optional
        Number of histogram bins for each parameter (default: 50).
    show : bool, optional
        If True, displays the figure and calls tight_layout(). If False, only
        returns the axes without displaying (default: True).
    Returns
    -------
    numpy.ndarray
        Array of matplotlib Axes objects corresponding to each subplot.
        For a single parameter, returns a 1D array; for multiple parameters,
        returns a flattened array of subplots.
    Notes
    -----
    - Histograms are plotted with density normalization enabled
    - Unused subplots (when n_params is not a multiple of 3) are hidden
    - Each histogram is displayed with black edges and 70% transparency
    """
    if isinstance(data, dict) and "inputs" in data:
        inputs = data["inputs"]
    else:
        inputs = data

    n_params = len(inputs)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 5, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_params == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes).flatten()

    color_cycle = cycle(plt.cm.tab10.colors)
    hist_color = next(color_cycle)
    hist_color = next(color_cycle)

    for idx, (label, arr) in enumerate(inputs.items()):
        ax = axes[idx]
        ax.hist(
            np.asarray(arr, dtype=float),
            bins=bins,
            density=True,
            color=hist_color,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(label, fontsize=9)

    for i in range(n_params, len(axes)):
        axes[i].axis("off")

    if show:
        fig.tight_layout()
        plt.show()

    return axes


def plot_multiple_monte_carlo(
    data_list,
    metric="LCOP",
    bins=30,
    figsize=None,
    label=None,
    ax=None,
    show: bool = True,
):
    """
    Plot multiple Monte Carlo simulation results as overlaid histograms
    with fitted normal distributions.

    Parameters
    ----------
    data_list : list of dict or list of Plant
        List of Monte Carlo data dictionaries or Plant objects.
    metric : str, optional
        Metric to plot from Monte Carlo results (default: "LCOP").
    bins : int, optional
        Number of histogram bins (default: 30).
    figsize : tuple, optional
        Figure size as (width, height).
    label : str, optional
        Label for the x-axis.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    matplotlib.axes.Axes
    """
    metric = metric.upper()

    created_fig = None
    if ax is None:
        if figsize is None:
            created_fig, ax = plt.subplots()
        else:
            created_fig, ax = plt.subplots(figsize=figsize)

    color_cycle = cycle(plt.cm.tab10.colors)
    currency = r"\$"
    plotted_any = False

    for i, item in enumerate(data_list):
        values = None
        name = f"Case {i+1}"

        # ---- Case 1: Monte Carlo dict ----
        if isinstance(item, dict) and "metrics" in item:
            if metric not in item["metrics"]:
                continue

            values = np.asarray(item["metrics"][metric], dtype=float)
            name = item.get("name", name)
            currency = item.get("currency", currency)

        # ---- Case 2: Plant object ----
        elif hasattr(item, "monte_carlo_metrics"):
            mc_data = getattr(item, "monte_carlo_metrics", None)

            if not isinstance(mc_data, dict) or metric not in mc_data:
                continue

            values = np.asarray(mc_data[metric], dtype=float)
            name = getattr(item, "name", name)
            currency = getattr(item, "currency", currency)

        else:
            continue

        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        plotted_any = True
        color = next(color_cycle)

        mu, std = norm.fit(values)

        ax.hist(
            values,
            bins=bins,
            alpha=0.5,
            density=True,
            edgecolor="black",
            linewidth=0.5,
            color=color,
            zorder=1,
            label=name,
        )

        if std > 0:
            x = np.linspace(values.min(), values.max(), 1000)
            p = norm.pdf(x, mu, std)

            std_exp = int(np.floor(np.log10(std)))

            if std_exp == 0:
                stat_label = rf"$\mu$={mu:.3g}, $\sigma$={std:.3g}"
            else:
                std_mant = std / 10**std_exp
                stat_label = (
                    rf"$\mu$={mu:.3g}, "
                    rf"$\sigma$={std_mant:.2f}$\times 10^{{{std_exp}}}$")

            ax.plot(
                x,
                p,
                color=color,
                linewidth=1.2,
                zorder=2,
                linestyle="-",
                label=stat_label,
            )
        else:
            stat_label = rf"$\mu$={mu:.3g}, $\sigma$={std:.3g}"
            ax.axvline(
                mu,
                color=color,
                linewidth=1.2,
                zorder=2,
                linestyle="-",
                label=stat_label,
            )

    if label is None:
        label = _default_metric_label(currency, metric)

    ax.set_xlabel(label)
    ax.set_ylabel("Probability density")

    handles, labels_list = ax.get_legend_handles_labels()
    n_items = len(labels_list)

    bbox = None
    if plotted_any and n_items > 0:
        if n_items <= 4:
            ncol, loc, bbox = 1, "best", None
        elif n_items <= 6:
            ncol, loc, bbox = 3, "upper center", (0.5, 1.15)
        else:
            ncol, loc, bbox = 4, "upper center", (0.5, 1.20)

        ax.legend(
            loc=loc,
            ncol=ncol,
            fontsize=4,
            frameon=True,
            facecolor="white",
            framealpha=0.6,
            fancybox=True,
            bbox_to_anchor=bbox,
        )

    if created_fig is not None:
        if bbox is not None:
            created_fig.tight_layout(rect=[0, 0, 1, 0.92])
        else:
            created_fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.close(created_fig)

    return ax
