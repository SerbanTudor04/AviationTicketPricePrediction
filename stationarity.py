import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from config import DATA_DIR
from utils import _header


def test_adf(series: pd.Series, name: str = "", verbose: bool = True) -> dict:
    """
    Augmented Dickey-Fuller.
    H0: radacina unitara (non-stationar). p < 0.05 → stationar.
    """
    r = adfuller(series.dropna(), autolag="AIC")
    out = {
        "stat": r[0], "p": r[1], "lags": r[2],
        "cv1": r[4]["1%"], "cv5": r[4]["5%"], "cv10": r[4]["10%"],
        "stationary": r[1] < 0.05,
    }
    if verbose:
        label = "STATIONAR" if out["stationary"] else "NON-STATIONAR"
        print(f"  ADF  [{name:45s}]: stat={out['stat']:8.4f}  p={out['p']:.4f}  → {label}")
    return out


def test_kpss(series: pd.Series, name: str = "", verbose: bool = True) -> dict:
    """
    KPSS.
    H0: serie stationara. p < 0.05 → non-stationara.
    """
    stat, p, lags, crits = kpss(series.dropna(), regression="c", nlags="auto")
    out = {
        "stat": stat, "p": p, "lags": lags,
        "cv1": crits["1%"], "cv5": crits["5%"], "cv10": crits["10%"],
        "stationary": p >= 0.05,
    }
    if verbose:
        label = "STATIONAR" if out["stationary"] else "NON-STATIONAR"
        print(f"  KPSS [{name:45s}]: stat={out['stat']:8.4f}  p={out['p']:.4f}  → {label}")
    return out


def full_stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raport ADF+KPSS la nivel, prima diferenta, log-nivel, log-diferenta.
    Evidentiaza ordinul de integrare I(d) al fiecarei variabile.
    """
    _header("3. ANALIZA DE STATIONARITATE")
    raw_cols = ["Airline_Fares_CPI", "Brent_Oil", "USD_EUR"]
    rows = []

    for col in raw_cols:
        for label, s in [
            (f"{col} (nivel)",     df[col]),
            (f"{col} (D1)",        df[col].diff()),
            (f"log_{col} (nivel)", df[f"log_{col}"]),
            (f"log_{col} (D1)",    df[f"log_{col}"].diff()),
        ]:
            s = s.dropna()
            a = test_adf(s,  label, verbose=False)
            k = test_kpss(s, label, verbose=False)
            rows.append({
                "Serie":           label,
                "ADF stat":        round(a["stat"], 4),
                "ADF p":           round(a["p"],    4),
                "ADF stationar?":  a["stationary"],
                "KPSS stat":       round(k["stat"], 4),
                "KPSS p":          round(k["p"],    4),
                "KPSS stationar?": k["stationary"],
            })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    report.to_csv(f"{DATA_DIR}/stationarity_report.csv", index=False)
    return report
