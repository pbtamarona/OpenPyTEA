"""Matplotlib figure endpoints.

These render the exact figures the library produces in Jupyter — same
plotting.py code, same styling — so a downloaded chart matches the
notebook/publication output rather than the web view. Figures are
returned as PNG.
"""

import io
import threading

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from fastapi import APIRouter, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from openpytea.analysis import (  # noqa: E402
    direct_costs_data,
    fixed_capital_data,
    fixed_opex_data,
    variable_opex_data,
    levelized_cost_data,
    sensitivity_data,
    tornado_data,
)
from openpytea.plotting import (  # noqa: E402
    plot_stacked_bar,
    plot_sensitivity,
    plot_tornado,
    plot_monte_carlo,
    plot_multiple_monte_carlo,
    plot_monte_carlo_inputs,
)

from app import state  # noqa: E402
from app.plant_factory import require_active_plant  # noqa: E402
from app.routers.analysis import _plants_for  # noqa: E402
from app.schemas import ComparePlotIn, SensitivityPlotIn, TornadoIn, MCPlotIn  # noqa: E402

router = APIRouter()

# Matplotlib's pyplot state machine is not thread-safe, and FastAPI runs
# sync endpoints in a thread pool — one figure at a time.
_mpl_lock = threading.Lock()


def _png(fig) -> Response:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")


def _require_plant():
    return require_active_plant()


_STACKED_DATA_FUNCS = {
    "direct_costs": direct_costs_data,
    "capital": fixed_capital_data,
    "fixed_opex": fixed_opex_data,
    "variable_opex": variable_opex_data,
}


@router.get("/stacked/{kind}")
def stacked_plot(kind: str, pct: bool = False):
    """Stacked cost bar (direct costs / capital / fixed opex / variable opex), library-rendered."""
    data_func = _STACKED_DATA_FUNCS.get(kind)
    if data_func is None:
        raise HTTPException(status_code=404, detail=f"Unknown stacked plot '{kind}'")
    plant = _require_plant()
    with _mpl_lock:
        try:
            data = data_func(plant, pct=pct)
            fig, _ = plot_stacked_bar(data, show=False)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Plot failed: {e}")
        return _png(fig)


_COMPARE_DATA_FUNCS = {
    "capital": fixed_capital_data,
    "variable_opex": variable_opex_data,
    "fixed_opex": fixed_opex_data,
    "levelized_cost": levelized_cost_data,
}


@router.post("/compare/{kind}")
def compare_plot(kind: str, payload: ComparePlotIn):
    """Stacked breakdown of several plants side by side (Compare tab)."""
    data_func = _COMPARE_DATA_FUNCS.get(kind)
    if data_func is None:
        raise HTTPException(status_code=404, detail=f"Unknown compare plot '{kind}'")
    plants = _plants_for(payload)
    with _mpl_lock:
        try:
            fig, _ = plot_stacked_bar(data_func(plants, pct=payload.pct), show=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Plot failed: {e}")
        return _png(fig)


@router.post("/sensitivity")
def sensitivity_plot(payload: SensitivityPlotIn):
    """plot_sensitivity() over the same inputs the analysis endpoint uses."""
    plants = _plants_for(payload)
    with _mpl_lock:
        try:
            data = sensitivity_data(
                plants,
                parameter=payload.parameter,
                plus_minus_value=payload.plus_minus_value,
                n_points=payload.n_points,
                metric=payload.metric,
                additional_capex=payload.additional_capex,
            )
            fig, _ = plot_sensitivity(data, x_axis=payload.x_axis, show=False)
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Sensitivity plot failed: {e}")
        return _png(fig)


@router.post("/tornado")
def tornado_plot(payload: TornadoIn):
    """plot_tornado(); multiple plants become side-by-side subplots."""
    plants = _plants_for(payload)
    with _mpl_lock:
        try:
            datas = [
                tornado_data(
                    p,
                    plus_minus_value=payload.plus_minus_value,
                    metric=payload.metric,
                    additional_capex=payload.additional_capex,
                    category=payload.category,
                )
                for p in plants
            ]
            if len(datas) == 1:
                fig, _ = plot_tornado(datas[0], show=False)
            else:
                n = len(datas)
                height = max(2.4, 0.28 * max(len(d["labels"]) for d in datas) + 1.2)
                fig, axes = plt.subplots(1, n, figsize=(3.4 * n, height))
                for ax, d, p in zip(axes, datas, plants):
                    plot_tornado(d, ax=ax, show=False)
                    ax.set_title(getattr(p, "name", None) or "Plant", fontsize=8)
                fig.tight_layout()
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Tornado plot failed: {e}")
        return _png(fig)


@router.post("/monte-carlo")
def monte_carlo_plot(payload: MCPlotIn):
    """plot_monte_carlo()/plot_multiple_monte_carlo() over the cached raw run."""
    raw = state.mc_raw
    if not raw:
        raise HTTPException(status_code=400, detail="Run a Monte Carlo simulation first")
    if payload.plants is not None:
        if not payload.plants or any(not 0 <= i < len(raw) for i in payload.plants):
            raise HTTPException(status_code=400, detail="No such plant in the Monte Carlo run")
        raw = [raw[i] for i in payload.plants]
    with _mpl_lock:
        try:
            if len(raw) == 1:
                fig, _ = plot_monte_carlo(
                    raw[0], metric=payload.metric, show_fit=payload.show_fit, show=False)
            else:
                fig, _ = plot_multiple_monte_carlo(
                    raw, metric=payload.metric, show_fit=payload.show_fit, show=False)
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Monte Carlo plot failed: {e}")
        return _png(fig)


@router.get("/monte-carlo/inputs")
def monte_carlo_inputs_plot(category: str = "process", plant: int = 0):
    """plot_monte_carlo_inputs() for one plant (its position in the run) of
    the cached raw run."""
    raw = state.mc_raw
    if not raw:
        raise HTTPException(status_code=400, detail="Run a Monte Carlo simulation first")
    if not 0 <= plant < len(raw):
        raise HTTPException(status_code=400, detail="No such plant in the Monte Carlo run")
    if category not in ("process", "economic"):
        raise HTTPException(status_code=400, detail="category must be 'process' or 'economic'")
    with _mpl_lock:
        try:
            fig, _ = plot_monte_carlo_inputs(raw[plant], category=category, show=False)
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Inputs plot failed: {e}")
        if fig is None:
            raise HTTPException(status_code=404, detail=f"No {category} inputs in this run")
        return _png(fig)
