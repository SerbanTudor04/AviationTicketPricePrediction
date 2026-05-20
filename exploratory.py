import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from config import COLORS, DATA_DIR
from utils import _slug


def plot_raw_series(df: pd.DataFrame) -> None:
    """Serii brute si log-transformate (3x2 subgrafice)."""
    config = [
        ("Airline_Fares_CPI", "log_Airline_Fares_CPI", "CPI Bilete Avion",  "Indice",    COLORS["train"]),
        ("Brent_Oil",         "log_Brent_Oil",          "Pret Petrol Brent", "USD/baril", COLORS["oil"]),
        ("USD_EUR",           "log_USD_EUR",             "Curs USD/EUR",      "USD/EUR",   COLORS["fx"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle("Serii de timp: niveluri si log-transformari", fontsize=14, fontweight="bold")

    for i, (col, lcol, title, ylabel, color) in enumerate(config):
        for j, (s, lbl, yl) in enumerate([
            (df[col],  title,           ylabel),
            (df[lcol], f"log({title})", f"log({ylabel})"),
        ]):
            axes[i, j].plot(s.index, s, color=color, linewidth=1.1)
            axes[i, j].set_title(lbl)
            axes[i, j].set_ylabel(yl)
            axes[i, j].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig01_serii_brute.png", bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap de corelatie pentru cele 3 variabile principale."""
    cols = ["Airline_Fares_CPI", "Brent_Oil", "USD_EUR"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                ax=ax, annot_kws={"size": 13})
    ax.set_title("Matrice de corelatie", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig02_correlatie.png", bbox_inches="tight")
    plt.show()


def decompose_series(series: pd.Series, name: str, period: int = 12) -> None:
    """Descompunere sezoniera aditivaeriment (trend + sezon + reziduu)."""
    result = seasonal_decompose(series.dropna(), model="additive", period=period)
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Descompunere sezoniera: {name}", fontsize=13, fontweight="bold")

    for ax, data, lbl in zip(
        axes,
        [series, result.trend, result.seasonal, result.resid],
        ["Observat", "Trend", "Sezonalitate", "Reziduu"],
    ):
        ax.plot(data.index, data, linewidth=1.1)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig03_decomp_{_slug(name)}.png", bbox_inches="tight")
    plt.show()


def plot_acf_pacf(series: pd.Series, name: str, lags: int = 36) -> None:
    """ACF si PACF pentru identificarea ordinului ARMA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"ACF / PACF: {name}", fontsize=13, fontweight="bold")
    plot_acf(series.dropna(),  lags=lags, ax=ax1, title="ACF")
    plot_pacf(series.dropna(), lags=lags, ax=ax2, title="PACF", method="ywm")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig_acf_{_slug(name)}.png", bbox_inches="tight")
    plt.show()
