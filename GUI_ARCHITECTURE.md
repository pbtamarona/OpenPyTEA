# OpenPyTEA GUI Architecture

## Overview

The GUI is a client-server application wrapping the OpenPyTEA Python library. The backend exposes OpenPyTEA functionality as a REST API; the frontend renders a tabbed single-page app for the full TEA workflow.

## Stack

- **Backend**: FastAPI (Python), runs on port 8000
- **Frontend**: React 18 + TypeScript + Vite, runs on port 5173
- **Charts**: Recharts (React-native charting, no matplotlib on the frontend)
- **Styling**: Plain CSS (no framework), defined in `App.css`
- **State**: In-memory on the backend (module-level singletons in `state.py`). No database. Single-user session.

## Directory Layout

```
backend/
  app/
    main.py           # FastAPI app, CORS config, router mounting
    state.py          # Module-level session state: equipment_list, plant, plant_config, results, mc_results
    schemas.py        # Pydantic request/response models (EquipmentIn, PlantConfigIn, SensitivityIn, etc.)
    util.py           # to_jsonable() — recursive numpy-to-native converter for JSON responses
    routers/
      equipment.py    # CRUD for equipment items + cost correlation DB lookup endpoints
      plant.py        # Plant config get/set + calculate endpoint (calls plant.calculate_all())
      analysis.py     # Sensitivity, tornado, Monte Carlo endpoints (calls openpytea.analysis functions)
      io.py           # Save/load full project as JSON (file upload/download)

frontend/
  src/
    api/client.ts     # Typed fetch wrapper — one function per API endpoint
    types/index.ts    # TypeScript interfaces mirroring backend schemas
    pages/
      EquipmentPage   # Equipment table + add/edit modal with cost DB category/type picker
      PlantConfigPage # Forms: general, financial, labor, products, variable OPEX
      ResultsPage     # Triggers calculate, shows metric cards + charts + cash flow table
      AnalysisPage    # Sensitivity line chart + tornado horizontal bar chart
      MonteCarloPage  # MC config, summary stats table, histogram per metric
    App.tsx           # Tab navigation (5 tabs) + save/load buttons in header
    App.css           # All styling: cards, forms, tables, modals, metric cards, spinner
```

## Data Flow

1. **Equipment** is defined on the frontend and synced to `state.equipment_list` on the backend via CRUD endpoints (`POST/PUT/DELETE /api/equipment`). Each Equipment object is constructed server-side using `openpytea.Equipment`, which evaluates cost correlations and inflation adjustment immediately.

2. **Plant config** is a JSON dict stored in `state.plant_config`. The frontend form collects all parameters and sends them via `PUT /api/plant/config`. This dict is NOT yet a Plant object — it's just config storage.

3. **Calculate** (`POST /api/plant/calculate`) combines `state.plant_config` + `state.equipment_list`, constructs a `Plant(config)`, calls `plant.calculate_all()`, and stores the Plant object in `state.plant`. Results are extracted into a flat JSON dict and cached in `state.results`.

4. **Analysis** endpoints (`/api/analysis/*`) require `state.plant` to exist (i.e., calculate must run first). They call `openpytea.analysis` functions directly on the stored plant object.

5. **Monte Carlo** results are summarized server-side (histogram bins + stats) instead of sending million-element arrays. Raw results are kept in `state.mc_results` for potential follow-up.

6. **Save/Load** serializes/restores equipment list + plant config. On load, equipment is re-instantiated from saved params and plant is reset (must recalculate).

## API Endpoints (22 total)

### Equipment (`/api/equipment`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | List all equipment (returns computed costs) |
| POST | `/` | Add equipment from params or direct cost |
| PUT | `/{index}` | Update equipment at index |
| DELETE | `/{index}` | Remove equipment at index |
| GET | `/cost-db/categories` | Cost correlation DB grouped by category (for dropdowns) |
| GET | `/process-types` | List: Solids, Fluids, Mixed, Electrical |
| GET | `/materials` | List: Carbon steel, 304 SS, etc. |

### Plant (`/api/plant`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/config` | Current plant config dict |
| PUT | `/config` | Set/update plant config |
| POST | `/calculate` | Run calculate_all(), return full results |
| GET | `/locations` | Plant.locFactors (country/region -> factor) |
| GET | `/process-types` | Plant.processTypes (Solids/Fluids/Mixed -> OS/DE/X factors) |

### Analysis (`/api/analysis`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/sensitivity/parameters` | Valid parameter names for current plant |
| POST | `/sensitivity` | Run sensitivity_data(), return curves |
| POST | `/tornado` | Run tornado_data(), return sorted bars |
| POST | `/monte-carlo` | Run monte_carlo(), return summarized stats + histograms |

### Project I/O (`/api/project`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/save` | Return full project state as JSON |
| POST | `/load` | Upload JSON file, restore equipment + config |
| GET | `/examples` | List available example presets (id, title, description) |
| POST | `/examples/{id}` | Load an example preset into the session |

### Utility
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check |

## Frontend Pages

### EquipmentPage
- Table with columns: #, Name, Category, Type, Material, Process, Param, Units, Purchased ($), Direct ($), actions
- Footer row with totals
- Modal dialog for add/edit with:
  - Category dropdown (populated from `/cost-db/categories`)
  - Type dropdown (cascading, shows param units and valid range)
  - Process type, material, target year dropdowns
  - Toggle between "size parameter" and "direct cost input" modes
- On submit, backend constructs Equipment and returns computed costs

### PlantConfigPage
- Four card sections: General, Financial, Labor, Products, Variable OPEX
- Country/region are cascading dropdowns from `Plant.locFactors`
- Products and variable OPEX are dynamic key-value editors (add/remove rows)
- Each product/opex item has: name, consumption/production, price, std, min, max (std/min/max used for Monte Carlo distributions)
- "Save Configuration" button sends to backend

### ResultsPage
- "Run Calculations" button (or "Recalculate" if results exist)
- Metric cards row: Levelized Cost, NPV, IRR, ROI, Payback Time
- Capital costs: table + vertical bar chart (ISBL, OSBL, D&E, Contingency)
- Fixed OPEX: table + horizontal bar chart (14 line items)
- Variable OPEX: table + bar chart
- Revenue breakdown table
- Cash flow table: year-by-year with CAPEX, Revenue, Costs, Gross Profit, Tax, Cash Flow

### AnalysisPage
- **Sensitivity section**: parameter dropdown, +/- variation, points, metric selector. Line chart showing metric vs % change.
- **Tornado section**: +/- variation, metric selector. Horizontal bar chart with low (blue) / high (red) deviations from baseline.
- Parameters list is fetched from `/sensitivity/parameters` (includes top-level + variable_opex + products)

### MonteCarloPage
- Config: num_samples (default 50k), batch_size
- Summary stats table per metric: mean, std, p5, p25, median, p75, p95, min, max
- Histogram chart per metric (from pre-binned server data)
- Input distributions summary table

## Key Design Decisions

1. **No matplotlib on frontend** — backend returns raw data; frontend renders charts with Recharts. The `openpytea.plotting` module is not used by the GUI.

2. **Monte Carlo summarization** — raw arrays (potentially millions of floats) are histogrammed server-side into ~80 bins. Only bin edges, counts, and summary stats are sent to the frontend (~10KB vs ~40MB).

3. **Equipment is constructed server-side** — the backend creates `Equipment()` objects immediately on add/update, so the cost correlation lookup and inflation adjustment happen before the response. The frontend just displays computed values.

4. **Plant is stateful** — `state.plant` holds the last-calculated Plant object. Analysis endpoints operate on this object. If equipment or config changes, the user must recalculate.

5. **Single-session model** — one equipment list, one plant, one set of results at a time. No multi-project support. Save/load provides persistence via JSON files.

6. **Example presets** — JSON preset files live in `backend/app/presets/`. Each contains a complete project (equipment + plant config) extracted from the case study notebooks. The frontend header has an "Examples" dropdown that lists them via `GET /api/project/examples` and loads one via `POST /api/project/examples/{id}`. Loading a preset replaces the current session and navigates to the Equipment tab. Adding a new example is just adding a `.json` file to the presets directory — no code changes needed.

## Available Example Presets

| ID | Title | Source |
|----|-------|--------|
| `h2_smr` | Hydrogen - Steam Methane Reforming | Case Study 1 (14 equipment, US Gulf Coast) |
| `h2_methane_pyrolysis` | Hydrogen - Methane Pyrolysis | Case Study 1 (13 equipment, carbon co-product) |
| `h2_electrolysis` | Hydrogen - Water Electrolysis | Case Study 1 (3 equipment, direct costs) |
| `lh2_smr_best` | Liquid H2 - SMR Best Case | Case Study 2 (11 equipment, Netherlands) |
| `geothermal_heat_pump` | Geothermal Heat Pump | Case Study 3 (8 equipment, 30yr lifetime) |
| `geothermal_power_plant` | Geothermal Power Plant | Case Study 3 (7 equipment, ORC cycle) |

## Known Limitations / Future Work

- No multi-plant comparison in the GUI (the library supports passing lists of plants to analysis functions)
- No depreciation method configuration UI (the field exists in PlantConfig but has no dedicated editor)
- No additional CAPEX editor UI (fields exist but no dedicated add/remove interface)
- Monte Carlo runs synchronously — large sample counts block the API response (could add background task + polling)
- No input validation feedback on the frontend forms (relies on backend 400 errors)
- Cash flow table assumes scenario index 0 (the library supports multi-scenario arrays)
- The cost DB category dropdown shows raw CSV keys — could benefit from nicer formatting
- No chart export/download functionality
- No dark mode

## How to Run

**Backend:**
```bash
cd backend
PYTHONPATH=../src python3.10 -m uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install   # first time only
npm run dev
```

Open http://localhost:5173

## Bug Fix Applied

`src/openpytea/__init__.py` had `from io import ...` (stdlib) instead of `from .io import ...` (relative). Fixed to allow the package to import correctly.
