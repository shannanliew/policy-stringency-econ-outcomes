"""
04_analysis.py
--------------
Runs analysis for: Policy stringency and economic outcomes (2020–2022).

Inputs
------
data/processed/panel_2020_2022.csv

Outputs
-------
Figures (outputs/figures):
- scatter_stringency_unemployment_pooled_trend.png
- scatter_stringency_gdp_growth_pooled_trend.png

Tables (outputs/tables):
Unemployment (HC1):
- unemp_U1_baseline.csv
- unemp_U2_fullcontrols_listwise.csv
- unemp_U3_fullcontrols_vaxfilled_missingdummy.csv

Unemployment (Country FE + clustered SE):
- unemp_U4_countryFE_clustered.csv
- unemp_U5_countryFE_vaxfilled_clustered.csv

GDP growth (HC1):
- gdp_G1_baseline.csv
- gdp_G2_winsorised_baseline.csv

GDP growth (Country FE + clustered SE):
- gdp_G3_countryFE_clustered.csv
- gdp_G4_winsorised_countryFE_clustered.csv

Extension: drop 2020 (Country FE + clustered SE):
- unemp_U6_countryFE_clustered_drop2020.csv
- unemp_U7_countryFE_vaxfilled_clustered_drop2020.csv
- gdp_G5_countryFE_clustered_drop2020.csv
- gdp_G6_winsorised_countryFE_clustered_drop2020.csv

Comparison + summaries:
- compare_drop2020_countryFE_clustered.csv
- summary_stringency_all_models.csv

Notes
-----
- These are descriptive associations, not causal estimates.
- Country fixed effects + clustered SEs provide more conservative inference for panel data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class Paths:
    panel_csv: str = "data/processed/panel_2020_2022.csv"
    fig_dir: str = "outputs/figures"
    tab_dir: str = "outputs/tables"


def ensure_dirs(paths: Paths) -> None:
    os.makedirs(paths.fig_dir, exist_ok=True)
    os.makedirs(paths.tab_dir, exist_ok=True)


def winsorise(series: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsorise series at p and 1-p quantiles (keeps NaNs)."""
    s = series.astype(float)
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def pooled_scatter_with_pooled_line(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    yearcol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    point_size: int = 18,
    alpha: float = 0.35,
) -> None:
    """Scatter by year + one pooled linear trend line."""
    plt.figure()

    for yr in sorted(df[yearcol].dropna().unique()):
        sub = df[df[yearcol] == yr].dropna(subset=[xcol, ycol])
        plt.scatter(sub[xcol], sub[ycol], label=str(int(yr)), s=point_size, alpha=alpha)

    all_sub = df.dropna(subset=[xcol, ycol])
    x = all_sub[xcol].to_numpy(dtype=float)
    y = all_sub[ycol].to_numpy(dtype=float)

    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = m * x_line + b
        plt.plot(x_line, y_line, linewidth=3, alpha=0.9)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Year")
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_ols_hc1(formula: str, data: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """OLS with robust (HC1) standard errors; saves tidy CSV."""
    model = smf.ols(formula, data=data).fit(cov_type="HC1")
    tidy = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "se_hc1": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
        }
    )
    tidy["n_obs"] = int(model.nobs)
    tidy["r2"] = model.rsquared
    tidy.to_csv(out_csv, index=False)
    return tidy


def run_ols_clustered(
    formula: str, data: pd.DataFrame, out_csv: str, cluster_col: str = "iso3"
) -> pd.DataFrame:
    """OLS with clustered SEs by cluster_col; saves tidy CSV."""
    clusters = data[cluster_col]
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    tidy = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "se_cluster": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
        }
    )
    tidy["n_obs"] = int(model.nobs)
    tidy["r2"] = model.rsquared
    tidy["n_clusters"] = clusters.nunique()
    tidy.to_csv(out_csv, index=False)
    return tidy


def extract_term(table_csv: str, term: str = "stringency_index_mean") -> Optional[dict]:
    t = pd.read_csv(table_csv)
    r = t[t["term"] == term]
    if r.empty:
        return None
    return r.iloc[0].to_dict()


def main() -> None:
    paths = Paths()
    ensure_dirs(paths)

    # -------------------------
    # Load + prepare
    # -------------------------
    df = pd.read_csv(paths.panel_csv)

    required_cols = {
        "iso3", "year",
        "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean", "vax_end",
        "gdp_growth", "unemployment",
    }
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in panel: {sorted(missing_required)}")

    df["year"] = df["year"].astype(int)
    df["iso3"] = df["iso3"].astype(str)

    df["vax_missing"] = df["vax_end"].isna().astype(int)
    df["vax_filled"] = df["vax_end"].fillna(0.0)

    df["gdp_growth_w"] = winsorise(df["gdp_growth"], p=0.01)

    # -------------------------
    # Figures (pooled trend line)
    # -------------------------
    pooled_scatter_with_pooled_line(
        df=df,
        xcol="stringency_index_mean",
        ycol="unemployment",
        yearcol="year",
        xlabel="Average Stringency Index (year mean)",
        ylabel="Unemployment rate (% of labor force)",
        title="Stringency and unemployment (2020–2022): pooled trend",
        out_path=os.path.join(paths.fig_dir, "scatter_stringency_unemployment_pooled_trend.png"),
    )

    pooled_scatter_with_pooled_line(
        df=df,
        xcol="stringency_index_mean",
        ycol="gdp_growth",
        yearcol="year",
        xlabel="Average Stringency Index (year mean)",
        ylabel="GDP growth (annual %)",
        title="Stringency and GDP growth (2020–2022): pooled trend",
        out_path=os.path.join(paths.fig_dir, "scatter_stringency_gdp_growth_pooled_trend.png"),
    )

    # -------------------------
    # Unemployment models (HC1)
    # -------------------------
    u_base = df.dropna(subset=["unemployment", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean"]).copy()

    run_ols_hc1(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year)",
        u_base,
        os.path.join(paths.tab_dir, "unemp_U1_baseline.csv"),
    )

    u_listwise = df.dropna(
        subset=["unemployment", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean", "vax_end"]
    ).copy()

    run_ols_hc1(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + vax_end + C(year)",
        u_listwise,
        os.path.join(paths.tab_dir, "unemp_U2_fullcontrols_listwise.csv"),
    )

    run_ols_hc1(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + vax_filled + vax_missing + C(year)",
        u_base,
        os.path.join(paths.tab_dir, "unemp_U3_fullcontrols_vaxfilled_missingdummy.csv"),
    )

    # -------------------------
    # Unemployment models (Country FE + clustered SE)
    # -------------------------
    run_ols_clustered(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year) + C(iso3)",
        u_base,
        os.path.join(paths.tab_dir, "unemp_U4_countryFE_clustered.csv"),
        cluster_col="iso3",
    )

    run_ols_clustered(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + vax_filled + vax_missing + C(year) + C(iso3)",
        u_base,
        os.path.join(paths.tab_dir, "unemp_U5_countryFE_vaxfilled_clustered.csv"),
        cluster_col="iso3",
    )

    # -------------------------
    # GDP growth models (HC1)
    # -------------------------
    g_base = df.dropna(subset=["gdp_growth", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean"]).copy()

    run_ols_hc1(
        "gdp_growth ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year)",
        g_base,
        os.path.join(paths.tab_dir, "gdp_G1_baseline.csv"),
    )

    g_w_base = df.dropna(subset=["gdp_growth_w", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean"]).copy()

    run_ols_hc1(
        "gdp_growth_w ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year)",
        g_w_base,
        os.path.join(paths.tab_dir, "gdp_G2_winsorised_baseline.csv"),
    )

    # -------------------------
    # GDP growth models (Country FE + clustered SE)
    # -------------------------
    run_ols_clustered(
        "gdp_growth ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year) + C(iso3)",
        g_base,
        os.path.join(paths.tab_dir, "gdp_G3_countryFE_clustered.csv"),
        cluster_col="iso3",
    )

    run_ols_clustered(
        "gdp_growth_w ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year) + C(iso3)",
        g_w_base,
        os.path.join(paths.tab_dir, "gdp_G4_winsorised_countryFE_clustered.csv"),
        cluster_col="iso3",
    )

    # -------------------------
    # Extension: drop 2020, rerun FE + clustered (Unemployment + GDP)
    # -------------------------
    df_2122 = df[df["year"].isin([2021, 2022])].copy()

    u_2122 = df_2122.dropna(subset=["unemployment", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean"]).copy()
    run_ols_clustered(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year) + C(iso3)",
        u_2122,
        os.path.join(paths.tab_dir, "unemp_U6_countryFE_clustered_drop2020.csv"),
        cluster_col="iso3",
    )

    # Added: U7 (vax_filled + vax_missing, drop 2020)
    run_ols_clustered(
        "unemployment ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + vax_filled + vax_missing + C(year) + C(iso3)",
        u_2122,
        os.path.join(paths.tab_dir, "unemp_U7_countryFE_vaxfilled_clustered_drop2020.csv"),
        cluster_col="iso3",
    )

    g_2122 = df_2122.dropna(subset=["gdp_growth", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean"]).copy()
    run_ols_clustered(
        "gdp_growth ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year) + C(iso3)",
        g_2122,
        os.path.join(paths.tab_dir, "gdp_G5_countryFE_clustered_drop2020.csv"),
        cluster_col="iso3",
    )

    g_w_2122 = df_2122.dropna(subset=["gdp_growth_w", "stringency_index_mean", "cases_pm_mean", "deaths_pm_mean"]).copy()
    run_ols_clustered(
        "gdp_growth_w ~ stringency_index_mean + cases_pm_mean + deaths_pm_mean + C(year) + C(iso3)",
        g_w_2122,
        os.path.join(paths.tab_dir, "gdp_G6_winsorised_countryFE_clustered_drop2020.csv"),
        cluster_col="iso3",
    )

    # -------------------------
    # Comparison table: drop-2020 vs full-sample FE+clustered
    # -------------------------
    compare = []
    compare_models = [
        ("unemp_U4_countryFE_clustered", "full 2020–2022"),
        ("unemp_U6_countryFE_clustered_drop2020", "drop 2020"),
        ("unemp_U5_countryFE_vaxfilled_clustered", "full 2020–2022"),
        ("unemp_U7_countryFE_vaxfilled_clustered_drop2020", "drop 2020"),
        ("gdp_G3_countryFE_clustered", "full 2020–2022"),
        ("gdp_G5_countryFE_clustered_drop2020", "drop 2020"),
        ("gdp_G4_winsorised_countryFE_clustered", "full 2020–2022"),
        ("gdp_G6_winsorised_countryFE_clustered_drop2020", "drop 2020"),
    ]

    for model_name, sample_tag in compare_models:
        path = os.path.join(paths.tab_dir, f"{model_name}.csv")
        d = extract_term(path, term="stringency_index_mean")
        if d is None:
            continue
        compare.append(
            {
                "model": model_name,
                "sample": sample_tag,
                "coef": d.get("coef"),
                "se": d.get("se_cluster", d.get("se_hc1")),
                "p": d.get("p"),
                "n_obs": d.get("n_obs"),
                "n_clusters": d.get("n_clusters"),
                "r2": d.get("r2"),
            }
        )

    compare_df = pd.DataFrame(compare)
    compare_df.to_csv(os.path.join(paths.tab_dir, "compare_drop2020_countryFE_clustered.csv"), index=False)

    # -------------------------
    # Summary table: stringency coefficient across all models
    # -------------------------
    model_files = [
        "unemp_U1_baseline.csv",
        "unemp_U2_fullcontrols_listwise.csv",
        "unemp_U3_fullcontrols_vaxfilled_missingdummy.csv",
        "unemp_U4_countryFE_clustered.csv",
        "unemp_U5_countryFE_vaxfilled_clustered.csv",
        "gdp_G1_baseline.csv",
        "gdp_G2_winsorised_baseline.csv",
        "gdp_G3_countryFE_clustered.csv",
        "gdp_G4_winsorised_countryFE_clustered.csv",
        "unemp_U6_countryFE_clustered_drop2020.csv",
        "unemp_U7_countryFE_vaxfilled_clustered_drop2020.csv",
        "gdp_G5_countryFE_clustered_drop2020.csv",
        "gdp_G6_winsorised_countryFE_clustered_drop2020.csv",
    ]

    rows: List[dict] = []
    for f in model_files:
        path = os.path.join(paths.tab_dir, f)
        d = extract_term(path, term="stringency_index_mean")
        if d is None:
            continue
        se = d.get("se_hc1", d.get("se_cluster"))
        se_type = "HC1" if "se_hc1" in d else ("cluster" if "se_cluster" in d else "unknown")
        rows.append(
            {
                "model": f.replace(".csv", ""),
                "coef": d.get("coef"),
                "se": se,
                "se_type": se_type,
                "p": d.get("p"),
                "n_obs": d.get("n_obs"),
                "r2": d.get("r2"),
                "n_clusters": d.get("n_clusters"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(paths.tab_dir, "summary_stringency_all_models.csv"), index=False)

    print("Done.")
    print("Saved figures to:", paths.fig_dir)
    print("Saved tables to:", paths.tab_dir)


if __name__ == "__main__":
    main()