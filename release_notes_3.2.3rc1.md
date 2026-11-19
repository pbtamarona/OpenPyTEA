# OpenPyTEA 3.2.3rc1 (pre-release)

This pre-release adds equipment and configuration discovery (`equipment_options()`, autocompletion and typo checks for names and plant settings), a cumulative Monte Carlo plot with percentile markers, a reversed lognormal distribution, named additional CAPEX, OPEX ramp-up schedules, and multi-plant project files that `openpytea run` can run directly. It also adds nine cost correlations, tidies up the cost correlation database and fixes ROI and payback time with additional CAPEX. **Existing code keeps working:** renamed functions, categories and correlation keys still work with a deprecation warning.

Try it from TestPyPI with `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openpytea==3.2.3rc1`, or with the installers attached below. Please report anything that doesn't work before the final 3.2.3 release.

## New features

- **Find valid names with `equipment_options()`.** Called without arguments it lists the cost database's categories; with a category it lists the types with their size units, valid range, default material and source. A misspelled category, type, process type, material or installation now raises an error that suggests the closest valid names (e.g. "Did you mean 'Reactors'?").
- **Cumulative Monte Carlo plot.** `plot_monte_carlo_cumulative` plots the cumulative distribution of Monte Carlo results, one curve per plant for a list, so probabilities such as P(LCOP ≤ x) read straight off the curve.
- **Reversed lognormal distribution (`dist_id` 13).** A lognormal mirrored at `max` (the cap, required), with its long tail down to `min`, for inputs such as conversions or efficiencies that have a ceiling but can occasionally fall well short of it. `loc`/`mean` and `std` are in real units, and the sampled mean equals `loc` after truncation.
- **Named additional CAPEX.** `additional_capex_cost` and `additional_capex_years` accept `{name: value}` dicts (unnamed lists still work). Each named item gets its own segment in `fixed_capital_data` and its own line in `calculate_fixed_capital`'s printout.
- **OPEX ramp-up.** Optional `fixed_opex_ramp` and `variable_opex_ramp` set the share of annual fixed or variable OPEX charged in each project year (values above 1 allowed for start-up overspend). Without them, fixed OPEX is charged in full from start-up and variable OPEX follows production, as before.
- **Multi-plant project files.** `run_openpytea` and `openpytea run` accept project files with several plants (the desktop app's project format). Every plant is calculated into its own output folder. The file's single `analysis` block applies to all plants, and its `sensitivity`, `tornado` and `monte_carlo` sections can list the plants they run on by name under `plants`. A section that lists several plants also writes comparison results (overlaid sensitivity, tornado per plant, overlaid Monte Carlo histograms and cumulative curves) to a comparison folder.
- **Nine new cost correlations:** biogas upgrading (water, amine and organic scrubbers, PSA, membrane separation) from Bauer et al. (2013) and hydrogen storage (low pressure, Type 1, 2 and 4 vessels) from Parks et al. (2014). Contributed by @marwenchaabouni in #13.

## Improvements

- **Autocompletion and typo checks.** `Equipment`'s `process_type`, `category`, `type` and `material` (and `CompositeEquipment`'s `process_type` and `installation`) are typed as their valid names, exported as `ProcessType`, `Category`, `EquipmentType` and `Material`. The new `PlantConfig` TypedDict (with `OpexInput`, `ProductInput`, `Uncertainty` and `Dependency`) describes `Plant`'s settings, so `config: PlantConfig = {...}` gets key and value completion in IDEs. Plain dicts still work. Valid calls that type checkers used to flag (free-text category/type, `param=None` with a quoted cost) now check cleanly.
- **Percentiles on Monte Carlo plots.** `plot_monte_carlo` and `plot_monte_carlo_cumulative` mark P10, P50 and P90 by default (other levels via `percentiles=`, none via `percentiles=None`) and list their values in a compact legend. `plot_monte_carlo`'s fitted normal curve is now off by default because Monte Carlo results are often skewed; `show_fit=True` brings it back.
- **One function for Monte Carlo histograms.** `plot_monte_carlo` now also takes a list of Monte Carlo results or plants and overlays them, one named histogram per case.
- **Removing settings with `update_configuration`.** A `None` value now deletes that key (a whole item or one field) instead of storing `None`, which crashed for removed items. Removing an item that another item's dependency still points to now warns straight away instead of failing later in `monte_carlo` or `tornado_data`.
- **Cleaner cost correlation database.** Correlation keys are lowercase without spaces (e.g. `pall_rings_cs_1in_seider_2013`), and the year in a key matches its cost year. Several categories are merged or renamed (see Deprecations). Category and type lookups ignore case and extra spaces. Units are written consistently (`m^2`, `m^3`). The CFB CHP correlation is now described as the equipment cost of a CFB boiler and steam turbine, as given in IRENA (2012) Table 5.5.

## Deprecations

These still work but raise a warning and will be removed in a future release:

- `plot_multiple_monte_carlo` → `plot_monte_carlo` with a list (FutureWarning, removed in 4.0).
- Categories (FutureWarning): `Adsorption & membrane separation`, `Gas separation` and `CO2 capture` → `Absorption, adsorption, & membrane separation`; `Biomass combustion` → `Boilers, heaters, & furnaces`; `Electrolysis` → `Electrolyzers & fuel cells`.
- Type (FutureWarning): `CFB CHP` → `Circulating fluidized bed (CFB), combined heat & power (CHP)`.
- Correlation keys passed as `cost_func` (DeprecationWarning): keys with spaces or capitals (e.g. `pall_rings_ cs_1in_seider_2013` → `pall_rings_cs_1in_seider_2013`), `propeller_fan_seider_2023` → `propeller_fan_seider_2013`, `biomass_cfb_chp_irena_2012` → `biomass_cfb_chp_irena_2010`, and `centrifugal_2-stage_3600_VSC_seider_2013` / `centrifugal_2+-stage_3600_VSC_seider_2013` → `..._hsc_seider_2013`.

If you read the bundled `cost_correlations.csv` directly: its `Remarks` and `doi /  link` columns are now `remarks` and `doi / link`.

## Bug fixes

- **ROI and simple payback time with `additional_capex=True`** now add only the additional CAPEX the cash flow actually spends in each sample, not the sum of every entry. They no longer charge spending scheduled outside the project lifetime and now agree with NPV, IRR, payback time and LCOP. **This can change ROI and simple payback time results for plants with additional CAPEX.**
- A plant's region is shown only for countries that have regional location factors (no more "Germany / Gulf Coast" in the summary and `to_dict`).
- Stacked bar charts (CAPEX, fixed and variable OPEX, LCOP) no longer draw components that are zero in every bar, such as interest on working capital when the plant isn't debt-funded, so these no longer leave empty legend entries.

## Desktop app

- **Installers** for Windows (`OpenPyTEA_3.2.3-rc.1_x64-setup.exe`) and macOS on Apple Silicon (`OpenPyTEA_3.2.3-rc.1_aarch64.dmg`) are attached to this release about 20 minutes after it's published.
- **Project-wide analysis.** The Analysis and Monte Carlo tabs cover the whole project instead of one plant at a time. Projects save their analyses and re-run them when opened, and a project file runs the same way with `openpytea run`.
- **Examples open whole case studies** as projects (all plants and project details) and run their analyses after loading.
- **Monte Carlo tab:** switch each metric between histogram and cumulative distribution, set the percentiles to mark, and see each metric's statistics beside its plot.
- **Results:** cumulative cash flow diagram beside the schedule, named additional CAPEX items as separate bars and rows, a Composite / Components toggle for the direct cost breakdown, editable fixed and variable OPEX ramp-up, and thousand separators in the input boxes.
- **Process flow diagrams:** each plant can hold a flowsheet image (upload or drag and drop), saved in the project file. The examples come with the case-study flowsheets.
- **Compare tab** overlays the ticked plants' cumulative cash flows. Every chart has a grid toggle that carries over to its downloaded figure.
- **Fixes:** changing a plant's country sets the region to the country's first region instead of leaving it blank and failing with "Region not found". The welcome page fits the minimum window size without scrolling.

## Docs

- The case study notebooks are re-run with this version and show the new features: cumulative Monte Carlo distributions, cumulative cash flow diagrams, sampled-input histograms, `equipment_options()`, and the case study 1 PSA as a `CompositeEquipment` instead of a 1994 placeholder cost. The case study 1 and 3 flowsheets are redrawn and labelled with the values the notebooks use.
- The I/O user guide and `case_study_1_with_JSON` describe the multi-plant project file and its project-wide analysis.
- The user guide and part 4 walkthrough cover the reversed lognormal distribution, percentiles and the cumulative plot. The cost correlations guide lists the renamed categories, types and keys.
- The docs homepage's workshop announcement is now a Workshop page recapping the November 6, 2026 workshop, with a link to the workshop materials.
