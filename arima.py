import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

from config import DATA_DIR
from utils import _slug


def find_best_sarima(train: pd.Series, name: str, m: int = 12):
    """
    Auto-ARIMA cu cautare sezoniera (m=12 pentru date lunare).
    Criteriu de selectie: AIC.
    """
    print(f"\n  Cautare SARIMA pentru [{name}] (poate dura ~30s)...")
    model = auto_arima(
        train, seasonal=True, m=m,
        information_criterion="aic", stepwise=True,
        suppress_warnings=True, error_action="ignore", trace=False,
    )
    print(f"  Model selectat: SARIMA{model.order}x{model.seasonal_order}[{m}]"
          f" | AIC={model.aic():.2f}")
    return model


def sarima_predict(model, n: int, test_index):
    """Predictie punctuala + IC 95% pentru n perioade."""
    fc_arr, ci = model.predict(n_periods=n, return_conf_int=True, alpha=0.05)
    fc    = pd.Series(fc_arr,   index=test_index, name="Forecast")
    lower = pd.Series(ci[:, 0], index=test_index, name="Lower_95")
    upper = pd.Series(ci[:, 1], index=test_index, name="Upper_95")
    return fc, lower, upper


def fit_arima_manual(series: pd.Series, order: tuple):
    """ARIMA cu ordine specificat manual — demonstreaza Cerinta 4."""
    m = ARIMA(series, order=order).fit()
    print(f"  ARIMA{order}: AIC={m.aic:.2f} | BIC={m.bic:.2f}")
    return m


def plot_sarima_diagnostics(model, name: str) -> None:
    """Diagnostice reziduale: timp, histograma, ACF, QQ-plot."""
    res = pd.Series(model.resid())
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Diagnostice SARIMA: {name}", fontsize=13, fontweight="bold")

    axes[0, 0].plot(res.values, linewidth=1)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Reziduuri in timp")

    axes[0, 1].hist(res, bins=25, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 1].set_title("Distributie reziduuri")

    plot_acf(res, lags=30, ax=axes[1, 0], title="ACF reziduuri")
    stats.probplot(res, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot (normalitate)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/fig07_diag_{_slug(name)}.png", bbox_inches="tight")
    plt.show()


def forecast_future(model, n_future: int = 24):
    """Prognoza extensa dincolo de setul de date (2024-2025)."""
    fc_arr, ci = model.predict(n_periods=n_future, return_conf_int=True)
    fc    = pd.Series(fc_arr,   name="Future")
    lower = pd.Series(ci[:, 0], name="Lower")
    upper = pd.Series(ci[:, 1], name="Upper")
    return fc, lower, upper
