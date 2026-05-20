import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from config import COLORS, DATA_DIR
from utils import _slug


def ses_forecast(train: pd.Series, test: pd.Series) -> pd.Series:
    """Simple Exponential Smoothing — alfa optim prin MLE."""
    m  = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    fc = m.forecast(len(test))
    fc.index = test.index
    print(f"  SES:           alpha={m.params['smoothing_level']:.4f}")
    return fc


def holt_forecast(train: pd.Series, test: pd.Series) -> pd.Series:
    """Holt cu trend aditiv, fara sezonalitate."""
    m  = ExponentialSmoothing(train, trend="add",
                              initialization_method="estimated").fit(optimized=True)
    fc = m.forecast(len(test))
    fc.index = test.index
    print(f"  Holt:          alpha={m.params['smoothing_level']:.4f}  "
          f"beta={m.params['smoothing_trend']:.4f}")
    return fc


def holt_winters_forecast(train: pd.Series, test: pd.Series,
                           seasonal: str = "add", periods: int = 12) -> pd.Series:
    """Holt-Winters cu trend + sezonalitate lunara (add sau mul)."""
    m = ExponentialSmoothing(
        train, trend="add", seasonal=seasonal,
        seasonal_periods=periods, initialization_method="estimated",
    ).fit(optimized=True)
    fc = m.forecast(len(test))
    fc.index = test.index
    print(f"  HW ({seasonal}):  alpha={m.params['smoothing_level']:.4f}  "
          f"beta={m.params['smoothing_trend']:.4f}  "
          f"gamma={m.params['smoothing_seasonal']:.4f}")
    return fc


def plot_smoothing_comparison(train: pd.Series, test: pd.Series,
                               forecasts: dict, name: str) -> None:
    """Comparare vizuala SES / Holt / Holt-Winters vs. real."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train, color=COLORS["train"], label="Antrenare", linewidth=1.2)
    ax.plot(test.index,  test,  color=COLORS["test"],  label="Real (Test)", linewidth=2)

    palette = ["#FF9800", "#9C27B0", "#F44336"]
    for (lbl, fc), clr in zip(forecasts.items(), palette):
        ax.plot(fc.index, fc, "--", color=clr, label=lbl, linewidth=1.5)

    ax.axvline(train.index[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Netezire exponentiala: {name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig06_smoothing_{_slug(name)}.png", bbox_inches="tight")
    plt.show()
