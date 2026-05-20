import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)

from arima import (find_best_sarima, fit_arima_manual, forecast_future,
                   plot_sarima_diagnostics, sarima_predict)
from config import COLORS, DATA_DIR, N_TEST
from exploratory import decompose_series, plot_acf_pacf
from smoothing import (holt_forecast, holt_winters_forecast,
                       plot_smoothing_comparison, ses_forecast)
from trend import analyze_stochastic_trend, fit_deterministic_trend
from utils import _header, _slug, _sub


def compute_metrics(actual: pd.Series, predicted: pd.Series, model_name: str) -> dict:
    """RMSE, MAE, MAPE pentru evaluarea acuratetei prognozei."""
    a, p = actual.values, predicted.values
    return {
        "Model":   model_name,
        "RMSE":    round(float(np.sqrt(mean_squared_error(a, p))), 4),
        "MAE":     round(float(mean_absolute_error(a, p)),         4),
        "MAPE(%)": round(float(mean_absolute_percentage_error(a, p)) * 100, 4),
    }


def compare_models_table(actual: pd.Series, forecasts: dict) -> pd.DataFrame:
    """Tabel comparativ — sortat dupa RMSE crescator."""
    rows = [compute_metrics(actual, fc, name) for name, fc in forecasts.items()]
    df   = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print("\n--- Comparare modele univariate ---")
    print(df.to_string(index=False))
    df.to_csv(f"{DATA_DIR}/model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"][:len(df)]
    ax.barh(df["Model"], df["MAPE(%)"], color=colors_bar)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Acuratete prognoza — MAPE (%)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig08_mape_comparison.png", bbox_inches="tight")
    plt.show()

    return df


def plot_final_forecast(train: pd.Series, test: pd.Series,
                        hw_fc: pd.Series, sarima_fc: pd.Series,
                        lower: pd.Series, upper: pd.Series, name: str) -> None:
    """Plot final: train + test + HW + SARIMA cu IC 95%."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(train.index, train,     color=COLORS["train"],  label="Antrenare",    linewidth=1.2)
    ax.plot(test.index,  test,      color=COLORS["test"],   label="Real (Test)",  linewidth=2)
    ax.plot(test.index,  hw_fc,     color=COLORS["hw"],     label="Holt-Winters", linestyle="--", linewidth=1.5)
    ax.plot(test.index,  sarima_fc, color=COLORS["sarima"], label="SARIMA",       linewidth=1.5)
    ax.fill_between(test.index, lower, upper,
                    color=COLORS["ci"], alpha=0.5, label="IC 95% SARIMA")
    ax.axvline(train.index[-1], color="gray", linestyle=":", linewidth=1.2)
    ax.set_title(f"Prognoza univariata: {name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig09_forecast_{_slug(name)}.png", bbox_inches="tight")
    plt.show()


def univariate_pipeline(df: pd.DataFrame, target: str = "Airline_Fares_CPI",
                         n_test: int = N_TEST) -> pd.DataFrame:
    """Pipeline univariat complet pe variabila tinta — Cerinte 1-6."""
    _header(f"ANALIZA UNIVARIATA: {target}")

    y = df[target].dropna()

    # Cerinta 1: trend
    fit_deterministic_trend(y, target)
    analyze_stochastic_trend(y, target)

    # Exploratorie
    decompose_series(y, target)
    plot_acf_pacf(y,                 target)
    plot_acf_pacf(y.diff().dropna(), f"{target} D1")

    # Cerinta 5: train/test split
    train, test = y.iloc[:-n_test], y.iloc[-n_test:]
    print(f"\nAntrenare: {len(train)} obs.  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Test:      {len(test)} obs.  ({test.index[0].date()} → {test.index[-1].date()})")

    # Cerinta 3: netezire exponentiala
    _sub("Netezire exponentiala")
    ses_fc = ses_forecast(train, test)
    hlt_fc = holt_forecast(train, test)
    hwa_fc = holt_winters_forecast(train, test, seasonal="add")
    plot_smoothing_comparison(
        train, test,
        {"SES": ses_fc, "Holt": hlt_fc, "Holt-Winters (add)": hwa_fc},
        target,
    )

    # Cerinta 4 & 5: SARIMA
    _sub("ARIMA / SARIMA")
    fit_arima_manual(train.diff().dropna(), order=(1, 0, 1))

    sarima_mdl = find_best_sarima(train, target)
    sarima_fc, lower, upper = sarima_predict(sarima_mdl, n_test, test.index)
    plot_sarima_diagnostics(sarima_mdl, target)

    plot_final_forecast(train, test, hwa_fc, sarima_fc, lower, upper, target)

    # Cerinta 6: comparare metode
    metrics = compare_models_table(
        test,
        {"SES": ses_fc, "Holt": hlt_fc, "Holt-Winters (add)": hwa_fc, "SARIMA": sarima_fc},
    )

    # Prognoza viitor 24 luni
    _sub("Prognoza viitor 24 luni (2024-2025)")
    fut_fc, fut_lo, fut_hi = forecast_future(sarima_mdl, n_future=24)
    print(f"  SARIMA 24 luni: min={fut_fc.min():.2f} | mean={fut_fc.mean():.2f} | max={fut_fc.max():.2f}")

    future_idx = pd.date_range(start=y.index[-1], periods=25, freq="MS")[1:]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y.index, y, color=COLORS["train"], label="Istoric", linewidth=1.2)
    ax.plot(future_idx, fut_fc.values, color=COLORS["sarima"], label="Prognoza SARIMA", linewidth=2)
    ax.fill_between(future_idx, fut_lo.values, fut_hi.values,
                    color=COLORS["ci"], alpha=0.5, label="IC 95%")
    ax.axvline(y.index[-1], color="gray", linestyle=":", linewidth=1.2)
    ax.set_title(f"Prognoza viitor 24 luni: {target}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fig09b_viitor_{_slug(target)}.png", bbox_inches="tight")
    plt.show()

    return metrics
