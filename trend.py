import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from config import COLORS, DATA_DIR
from stationarity import test_adf
from utils import _slug, _sub


def fit_deterministic_trend(series: pd.Series, name: str) -> None:
    """
    Trend determinist: regresie OLS pe trend liniar + patratic.
    Testeaza daca reziduurile sunt stationare.
    """
    _sub(f"Trend determinist patratic: {name}")
    t = np.arange(len(series))
    x = np.column_stack([np.ones(len(t)), t, t ** 2])
    model     = OLS(series.values, x).fit()
    fit_vals  = model.fittedvalues
    residuals = series.values - fit_vals

    adf_res = test_adf(pd.Series(residuals), f"Reziduu {name}")
    print(f"    R2={model.rsquared:.4f} | AIC={model.aic:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Trend determinist (patratic): {name}", fontweight="bold")

    axes[0].plot(series.index, series, alpha=0.7, label="Observat")
    axes[0].plot(series.index, fit_vals, "r--", linewidth=2, label="Trend patratic")
    axes[0].set_title("Serie + Trend")
    axes[0].legend()

    lbl = "Stationar" if adf_res["stationary"] else "Non-stationar"
    axes[1].plot(series.index, residuals, color="purple")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title(f"Reziduuri — {lbl}")

    axes[2].hist(residuals, bins=25, edgecolor="black", color="purple", alpha=0.7)
    axes[2].set_title("Distributie reziduuri")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig04_trend_det_{_slug(name)}.png", bbox_inches="tight")
    plt.show()


def analyze_stochastic_trend(series: pd.Series, name: str) -> None:
    """Trend stochastic: serie I(1) — prima diferenta devine stationara."""
    _sub(f"Trend stochastic (random walk): {name}")
    d1     = series.diff().dropna()
    adf_lv = test_adf(series, f"{name} nivel")
    adf_d1 = test_adf(d1,     f"{name} D1")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Trend stochastic: {name}", fontweight="bold")

    ax1.plot(series.index, series, color=COLORS["train"])
    ax1.set_title("Non-stationar" if not adf_lv["stationary"] else "Stationar")
    ax1.grid(True, alpha=0.3)

    ax2.plot(d1.index, d1, color=COLORS["test"])
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("D1 Stationar" if adf_d1["stationary"] else "D1 Non-stationar")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig05_trend_stoch_{_slug(name)}.png", bbox_inches="tight")
    plt.show()
