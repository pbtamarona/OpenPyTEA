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
    fixed_capital_data,
    fixed_opex_data,
    variable_opex_data,
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
from app.plant_factory import build_plant  # noqa: E402
from app.schemas import SensitivityIn, TornadoIn, MCPlotIn, PlantInput  # noqa: E402

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
    if state.plant is None:
        raise HTTPException(status_code=400, detail="Run calculations first")
    return state.plant


def _rehydrate_extras(extras: list[PlantInput]):
    plants = []
    for extra in extras:
        p = build_plant(extra.equipment, extra.plant)
        if extra.name:
            p.name = extra.name
        plants.append(p)
    return plants


_STACKED_DATA_FUNCS = {
    "capital": fixed_capital_data,
    "fixed_opex": fixed_opex_data,
    "variable_opex": variable_opex_data,
}


@router.get("/stacked/{kind}")
def stacked_plot(kind: str):
    """Stacked cost bar (capital / fixed opex / variable opex), library-rendered."""
    data_func = _STACKED_DATA_FUNCS.get(kind)
    if data_func is None:
        raise HTTPException(status_code=404, detail=f"Unknown stacked plot '{kind}'")
    plant = _require_plant()
    with _mpl_lock:
        try:
            data = data_func(plant)
            fig, _ = plot_stacked_bar(data, show=False)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Plot failed: {e}")
        return _png(fig)


@router.post("/sensitivity")
def sensitivity_plot(payload: SensitivityIn):
    """plot_sensitivity() over the same inputs the analysis endpoint uses."""
    plant = _require_plant()
    plants = [plant] + _rehydrate_extras(payload.extra_plants)
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
            fig, _ = plot_sensitivity(data, show=False)
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Sensitivity plot failed: {e}")
        return _png(fig)


@router.post("/tornado")
def tornado_plot(payload: TornadoIn):
    """plot_tornado(); multiple plants become side-by-side subplots."""
    plant = _require_plant()
    plants = [plant] + _rehydrate_extras(payload.extra_plants)
    with _mpl_lock:
        try:
            datas = [
                tornado_data(
                    p,
                    plus_minus_value=payload.plus_minus_value,
                    metric=payload.metric,
                    additional_capex=payload.additional_capex,
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
    with _mpl_lock:
        try:
            if len(raw) == 1:
                fig, _ = plot_monte_carlo(raw[0], metric=payload.metric, show=False)
            else:
                fig, _ = plot_multiple_monte_carlo(raw, metric=payload.metric, show=False)
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Monte Carlo plot failed: {e}")
        return _png(fig)


@router.get("/monte-carlo/inputs")
def monte_carlo_inputs_plot(category: str = "process"):
    """plot_monte_carlo_inputs() for the active plant's cached raw run."""
    raw = state.mc_raw
    if not raw:
        raise HTTPException(status_code=400, detail="Run a Monte Carlo simulation first")
    if category not in ("process", "economic"):
        raise HTTPException(status_code=400, detail="category must be 'process' or 'economic'")
    with _mpl_lock:
        try:
            fig, _ = plot_monte_carlo_inputs(raw[0], category=category, show=False)
        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Inputs plot failed: {e}")
        if fig is None:
            raise HTTPException(status_code=404, detail=f"No {category} inputs in this run")
        return _png(fig)
