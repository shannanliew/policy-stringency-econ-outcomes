# Policy stringency and economic outcomes (2020–2022)

This repository builds a reproducible country–year panel (2020–2022) linking COVID-19 policy stringency to macro outcomes and estimates descriptive associations (not causal effects).

## Data
- Our World in Data COVID “compact” dataset: stringency index, cases/deaths per million, vaccination coverage
- World Bank World Development Indicators (WDI): GDP growth (annual %), unemployment (% labour force)

## What the pipeline produces
- `data/processed/owid_country_year_2020_2022_iso3.csv`  
  Country–year features from OWID (annual means + year-end vaccination) with ISO3 mapping and non-country entities removed.
- `data/processed/panel_2020_2022.csv`  
  Final merged OWID–WDI panel keyed by `(iso3, year)`.
- `outputs/figures/`  
  Pooled-trend scatter plots for unemployment and GDP growth vs stringency.
- `outputs/tables/`  
  Regression tables (pooled OLS, country fixed effects with clustered SEs) + robustness checks and summaries.

## Methods (high level)
- Pooled OLS with year fixed effects
- Panel specifications with **country fixed effects** and **standard errors clustered by country**
- Robustness: drop 2020 and re-estimate FE + clustered models (2021–2022 only)

## Key results (brief)
- Unemployment: higher annual average stringency is consistently associated with higher unemployment; the association remains positive with country fixed effects and clustered SEs.
- GDP growth: negative association in the full sample, but not robust when excluding 2020 (suggesting the GDP relationship is concentrated in the initial shock year).

## Reproduce
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 src/01_download_owid.py
python3 src/02_download_wdi.py
python3 src/03_build_panel.py
python3 src/04_analysis.py
