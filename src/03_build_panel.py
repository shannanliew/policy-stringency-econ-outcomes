"""
03_build_panel.py
-----------------
Merge processed OWID country-year dataset with WDI outcomes to build final panel.

Outputs
-------
data/processed/panel_2020_2022.csv
outputs/tables/panel_audit.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class Paths:
    owid: str = "data/processed/owid_country_year_2020_2022_iso3.csv"
    wdi: str = "data/raw/wdi_outcomes_2020_2022.csv"
    panel: str = "data/processed/panel_2020_2022.csv"
    audit: str = "outputs/tables/panel_audit.csv"


def ensure_dirs() -> None:
    for d in ["data/processed", "outputs/tables"]:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ensure_dirs()
    p = Paths()

    owid = pd.read_csv(p.owid)
    wdi = pd.read_csv(p.wdi)

    owid["year"] = owid["year"].astype(int)
    wdi["year"] = wdi["year"].astype(int)
    owid["iso3"] = owid["iso3"].astype(str)
    wdi["iso3"] = wdi["iso3"].astype(str)

    wdi = wdi[["iso3", "year", "gdp_growth", "unemployment"]].copy()
    panel = owid.merge(wdi, on=["iso3", "year"], how="inner").copy()

    panel.to_csv(p.panel, index=False)
    print("Saved:", p.panel, "| shape:", panel.shape, "| countries:", panel["iso3"].nunique())

    cols_check: List[str] = [
        "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean", "vax_end",
        "gdp_growth", "unemployment",
    ]
    cols_check = [c for c in cols_check if c in panel.columns]

    audit = pd.DataFrame({
        "n_rows": [len(panel)],
        "n_countries": [panel["iso3"].nunique()],
        "years": [",".join(str(y) for y in sorted(panel["year"].unique()))],
    })
    for c in cols_check:
        audit[f"missing_{c}"] = [panel[c].isna().mean()]

    audit.to_csv(p.audit, index=False)
    print("Saved:", p.audit)


if __name__ == "__main__":
    main()