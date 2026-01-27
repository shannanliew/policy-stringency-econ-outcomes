"""
02_download_wdi.py
------------------
Download WDI outcomes for 2020–2022 via World Bank API (with retries + pagination).

Output
------
data/raw/wdi_outcomes_2020_2022.csv
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class Config:
    years: str = "2020:2022"
    per_page: int = 2000
    timeout_connect: int = 10
    timeout_read: int = 180
    out_csv: str = "data/raw/wdi_outcomes_2020_2022.csv"


INDICATORS: Dict[str, str] = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
}


def ensure_dirs() -> None:
    os.makedirs("data/raw", exist_ok=True)


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_indicator(session: requests.Session, code: str, cfg: Config) -> pd.DataFrame:
    api = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    base = {"format": "json", "per_page": cfg.per_page, "date": cfg.years}

    r1 = session.get(api, params={**base, "page": 1}, timeout=(cfg.timeout_connect, cfg.timeout_read))
    r1.raise_for_status()
    payload = r1.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected API response for {code}")

    meta = payload[0] or {}
    pages = int(meta.get("pages", 1))

    rows: List[dict] = []

    def consume(data):
        if data is None:
            return
        for d in data:
            val = d.get("value")
            iso3 = d.get("countryiso3code")
            year = d.get("date")
            if val is None or iso3 is None or year is None:
                continue
            if len(str(iso3)) != 3:
                continue
            rows.append({"iso3": str(iso3), "year": int(year), code: float(val)})

    consume(payload[1])

    for page in range(2, pages + 1):
        r = session.get(api, params={**base, "page": page}, timeout=(cfg.timeout_connect, cfg.timeout_read))
        r.raise_for_status()
        pl = r.json()
        if isinstance(pl, list) and len(pl) >= 2:
            consume(pl[1])
        time.sleep(0.1)

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    cfg = Config()
    session = make_session()

    tables = []
    for name, code in INDICATORS.items():
        df = fetch_indicator(session, code, cfg)
        tables.append(df)
        print(f"Fetched {name} ({code}): {df.shape}")

    wdi = tables[0]
    for t in tables[1:]:
        wdi = wdi.merge(t, on=["iso3", "year"], how="outer")

    wdi = wdi.rename(columns={
        INDICATORS["gdp_growth"]: "gdp_growth",
        INDICATORS["unemployment"]: "unemployment",
    })

    wdi.to_csv(cfg.out_csv, index=False)
    print("Saved:", cfg.out_csv, "| shape:", wdi.shape)


if __name__ == "__main__":
    main()