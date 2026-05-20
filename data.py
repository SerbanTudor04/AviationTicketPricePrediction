import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from config import DATA_DIR
from utils import _header

load_dotenv()


def fetch_data(start: str = "2010-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """
    Descarca date lunare din FRED:
      CUSR0000SETG01 — CPI bilete avion (NSA)
      POILBREUSDM    — Pret petrol Brent (USD/baril)
      EXUSEU         — Curs USD/EUR
    Adauga log-transformari si salveaza CSV.
    """
    _header("1. COLECTAREA DATELOR DIN FRED")

    series_map = {
        "Airline_Fares_CPI": "CUSR0000SETG01",
        "Brent_Oil":         "POILBREUSDM",
        "USD_EUR":           "EXUSEU",
    }

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Setati variabila de mediu FRED_API_KEY.\n"
            "Obtineti cheie gratuita la: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  export FRED_API_KEY=your_key_here"
        )
    fred = Fred(api_key=api_key)

    frames = {}
    for name, series_id in series_map.items():
        frames[name] = fred.get_series(series_id, observation_start=start, observation_end=end)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.resample("MS").first()
    df.dropna(inplace=True)

    inferred = pd.infer_freq(df.index)
    if inferred:
        df.index.freq = inferred

    for col in series_map:
        df[f"log_{col}"] = np.log(df[col])

    df.to_csv(f"{DATA_DIR}/date_aviatie_turism.csv")
    print(f"Perioada: {df.index[0].date()} → {df.index[-1].date()} | {len(df)} obs. lunare\n")
    print(df[list(series_map)].describe().round(3).to_string())
    return df
