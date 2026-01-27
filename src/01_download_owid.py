"""
01_download_owid.py
------------------
Download OWID "compact" COVID dataset and build a country-year panel (2020–2022)
with annual summary statistics and ISO3 mapping.

Outputs
-------
data/raw/owid_compact.csv
data/processed/owid_country_year_2020_2022_iso3.csv
outputs/tables/country_iso3_mapping.csv
outputs/tables/dropped_entities.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

OWID_URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"


@dataclass(frozen=True)
class Paths:
    raw_csv: str = "data/raw/owid_compact.csv"
    processed_csv: str = "data/processed/owid_country_year_2020_2022_iso3.csv"
    mapping_csv: str = "outputs/tables/country_iso3_mapping.csv"
    dropped_csv: str = "outputs/tables/dropped_entities.csv"


def ensure_dirs() -> None:
    for d in ["data/raw", "data/processed", "outputs/tables", "outputs/figures"]:
        os.makedirs(d, exist_ok=True)


def is_bad_entity(name: str) -> bool:
    s = str(name).strip().lower()

    # Exact-match aggregates
    continents = {"africa", "asia", "europe", "oceania", "north america", "south america"}
    if s in continents or s == "world":
        return True

    # Composites / exclusions / list-like entities
    if "excl" in s or "excluding" in s or "," in s:
        return True

    # Income groups / blocs / other aggregates
    if "income" in s or s.endswith("countries") or " countries" in s:
        return True
    if "european union" in s or s.startswith("eu ("):
        return True
    if "international" in s or "global" in s:
        return True

    # Subnational UK units
    uk_parts = {"england", "scotland", "wales", "northern ireland", "england and wales"}
    if s in uk_parts:
        return True

    # Events / non-geographic
    if "olympic" in s:
        return True

    # Disputed/special regions
    if "transnistria" in s:
        return True

    return False


def map_country_to_iso3(countries: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import country_converter as coco  # type: ignore
    except ImportError as e:
        raise ImportError("Install: pip install country_converter") from e

    map_df = pd.DataFrame({"country": countries})
    map_df["iso3_raw"] = coco.convert(map_df["country"].tolist(), to="ISO3")
    map_df["bad_entity"] = map_df["country"].apply(is_bad_entity)

    manual_fix: Dict[str, str] = {"Micronesia (country)": "FSM"}

    map_df["iso3"] = map_df["iso3_raw"]
    map_df.loc[map_df["country"].isin(manual_fix), "iso3"] = map_df["country"].map(manual_fix)

    # Override bad entities even if converter guessed
    map_df.loc[map_df["bad_entity"], "iso3"] = "not found"

    is_unmatched = map_df["iso3"].isin(["not found", None, ""])
    matched = map_df.loc[~is_unmatched, ["country", "iso3"]].copy()
    dropped = map_df.loc[is_unmatched, ["country", "iso3_raw", "bad_entity"]].copy()
    return matched, dropped


def build_country_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df = df[df["year"].between(2020, 2022)].copy()

    annual = (
        df.groupby(["country", "year"], as_index=False)
        .agg(
            stringency_index_mean=("stringency_index", "mean"),
            cases_pm_mean=("new_cases_per_million", "mean"),
            deaths_pm_mean=("new_deaths_per_million", "mean"),
        )
    )

    df_sorted = df.sort_values(["country", "year", "date"])
    vax_end = (
        df_sorted.dropna(subset=["people_vaccinated_per_hundred"])
        .groupby(["country", "year"], as_index=False)
        .tail(1)[["country", "year", "people_vaccinated_per_hundred"]]
        .rename(columns={"people_vaccinated_per_hundred": "vax_end"})
    )

    return annual.merge(vax_end, on=["country", "year"], how="left")


def main() -> None:
    ensure_dirs()
    p = Paths()

    usecols = [
        "country",
        "date",
        "stringency_index",
        "new_cases_per_million",
        "new_deaths_per_million",
        "people_vaccinated_per_hundred",
    ]
    df = pd.read_csv(OWID_URL, usecols=usecols)
    df.to_csv(p.raw_csv, index=False)

    annual = build_country_year(df)

    unique_countries = pd.Series(sorted(annual["country"].dropna().unique()), name="country")
    matched, dropped = map_country_to_iso3(unique_countries)

    matched.to_csv(p.mapping_csv, index=False)
    dropped.to_csv(p.dropped_csv, index=False)

    annual_clean = annual.merge(matched, on="country", how="inner")

    # Minimal filter: require stringency
    owid_proc = annual_clean.dropna(subset=["stringency_index_mean"]).copy()
    owid_proc.to_csv(p.processed_csv, index=False)

    print("Saved:", p.processed_csv, "| shape:", owid_proc.shape)
    print("Unique countries (iso3):", owid_proc["iso3"].nunique())


if __name__ == "__main__":
    main()