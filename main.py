"""
====================================================================
PROIECT SERII DE TIMP — CSIE ANUL III
Aviație și Turism: Prețuri Bilete, Petrol și Curs Valutar

CERINTE ACOPERITE:
  UNIVARIATE:
    1. Trend determinist (OLS pătratic) și stochastic (I(1))
    2. Staționaritate — ADF + KPSS
    3. Netezire exponențială — SES, Holt, Holt-Winters
    4. Modele ARMA / ARIMA / SARIMA (auto-selecție)
    5. Predicție punctuală + IC 95%, delimitare train/test/orizont
    6. Comparare metode (RMSE, MAE, MAPE)
  MULTIVARIATE:
    7. Non-stationaritate, cointegrare Johansen, VAR/VECM,
       cauzalitate Granger, IRF, FEVD

Inainte de rulare:
  export FRED_API_KEY=your_key_here
  python main.py
====================================================================
"""

import glob
import os

import config  # aplica rcParams, seaborn theme, warnings filter

os.makedirs(config.DATA_DIR, exist_ok=True)

from data import fetch_data
from exploratory import (decompose_series, plot_correlation_matrix,
                         plot_raw_series)
from multivariate import multivariate_pipeline
from stationarity import full_stationarity_report
from univariate import univariate_pipeline
from utils import _header


def main() -> None:
    _header("PROIECT SERII DE TIMP — Aviatie, Petrol si Curs Valutar | CSIE 2026")

    df = fetch_data(start="2010-01-01", end="2023-12-31")

    _header("2. ANALIZA EXPLORATORIE")
    plot_raw_series(df)
    plot_correlation_matrix(df)
    decompose_series(df["Airline_Fares_CPI"], "CPI Bilete Avion")
    decompose_series(df["Brent_Oil"],         "Pret Petrol Brent")
    decompose_series(df["USD_EUR"],           "Curs USD-EUR")

    full_stationarity_report(df)

    univariate_pipeline(df, target="Airline_Fares_CPI", n_test=config.N_TEST)

    multivariate_pipeline(df)

    _header("PROIECT FINALIZAT")
    figs = sorted(glob.glob(f"{config.DATA_DIR}/*.png") + glob.glob(f"{config.DATA_DIR}/*.csv"))
    print("Fisiere generate:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
