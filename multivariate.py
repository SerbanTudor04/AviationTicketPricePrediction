import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from config import COLORS, DATA_DIR
from stationarity import test_adf
from utils import _header, _sub


def prepare_mv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Log-niveluri I(1) pentru cele 3 variabile."""
    out = df[["log_Airline_Fares_CPI", "log_Brent_Oil", "log_USD_EUR"]].dropna().copy()
    out.columns = ["log_CPI", "log_Oil", "log_USDEUR"]
    return out


def granger_causality(df_diff: pd.DataFrame, max_lag: int = 6) -> None:
    """
    Test Granger pe primele diferente ale log-variabilelor.
    H0: X nu cauzeaza Granger Y. p < 0.05 → cauzalitate.
    Cheia economica: petrolul cauzeaza Granger CPI bilete?
    """
    _sub("Cauzalitate Granger (pe prime diferente)")
    cols    = df_diff.columns.tolist()
    results = []

    for target in cols:
        for cause in cols:
            if cause == target:
                continue
            data_pair = df_diff[[target, cause]].dropna()
            try:
                res   = grangercausalitytests(data_pair, maxlag=max_lag, verbose=False)
                min_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
                sig   = "***" if min_p < 0.01 else ("**" if min_p < 0.05 else ("*" if min_p < 0.1 else "ns"))
                results.append({"Cauza": cause, "Efect": target,
                                 "min_p": round(min_p, 4), "Semnif.": sig})
                print(f"  {cause:18s} → {target:18s}: min_p={min_p:.4f}  {sig}")
            except Exception as ex:
                print(f"  {cause} → {target}: Eroare ({ex})")

    pd.DataFrame(results).to_csv(f"{DATA_DIR}/granger_results.csv", index=False)


def johansen_test(df_mv: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> int:
    """
    Testul Johansen (trace test) — determina rangul de cointegrare.
    Returneaza rangul estimat r.
    det_order: -1=fara constanta, 0=constanta, 1=trend liniar.
    """
    _sub("Testul Johansen de Cointegrare")
    result = coint_johansen(df_mv, det_order=det_order, k_ar_diff=k_ar_diff)

    print(f"\n  {'H0':12s} {'Trace Stat':12s} {'CV 95%':12s} {'CV 99%':10s}")
    rank = 0
    for i, (stat, cv) in enumerate(zip(result.lr1, result.cvt)):
        reject = stat > cv[1]
        mark   = "  ← REJECT H0" if reject else ""
        print(f"  r <= {i:<8} {stat:<12.4f} {cv[1]:<12.4f} {cv[0]:<10.4f}{mark}")
        if reject:
            rank = i + 1

    print(f"\n  Rang estimat cointegrare: r = {rank}")
    return rank


def select_var_lag(df_diff: pd.DataFrame, max_lag: int = 12) -> int:
    """Selectie lag optim VAR pe baza AIC."""
    sel     = VAR(df_diff).select_order(max_lag)
    optimal = sel.selected_orders["aic"]
    print(f"  Lag optim (AIC): {optimal}")
    print(sel.summary())
    return optimal


def fit_var(df_diff: pd.DataFrame, lag: int):
    """Ajustare VAR(p) in prime diferente."""
    return VAR(df_diff).fit(max(1, lag))


def fit_vecm(df_mv: pd.DataFrame, coint_rank: int, k_ar_diff: int = 1):
    """Ajustare VECM cu rang de cointegrare dat."""
    model  = VECM(df_mv, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="ci")
    fitted = model.fit()
    print(fitted.summary())
    return fitted


def plot_irf(var_result, periods: int = 24, name: str = "VAR") -> None:
    """
    Functia de Raspuns la Impuls (IRF) ortogonalizat.
    Intrebarea cheie: cat de repede se transmite scumpirea petrolului
    in CPI biletelor de avion?
    """
    _sub(f"IRF — {periods} perioade")
    irf = var_result.irf(periods)

    irf.plot(orth=True)
    plt.suptitle(f"IRF ortogonalizat: {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"fig10_irf_{name}.png", bbox_inches="tight")
    plt.show()

    irf.plot_cum_effects(orth=True)
    plt.suptitle(f"Efecte cumulative IRF: {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"fig11_irf_cum_{name}.png", bbox_inches="tight")
    plt.show()


def plot_fevd(var_result, periods: int = 24, name: str = "VAR") -> None:
    """
    Descompunerea variantei (FEVD).
    Cat % din varianța prognozei CPI e explicat de socuri in petrol?
    """
    _sub(f"FEVD — {periods} perioade")
    fevd = var_result.fevd(periods)

    fevd.plot()
    plt.suptitle(f"Descompunerea variantei (FEVD): {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"fig12_fevd_{name}.png", bbox_inches="tight")
    plt.show()

    var_names = var_result.names
    print(f"\n  FEVD log_CPI (% varianța explicat de fiecare variabila):")
    print(f"  {'Orizont':>8}  " + "  ".join(f"{v:>12}" for v in var_names))
    for h in [1, 3, 6, 12, 24]:
        if h <= periods:
            row  = fevd.decomp[0, h - 1, :]   # decomp shape: (neqs, periods, neqs)
            pcts = "  ".join(f"{v * 100:12.2f}%" for v in row)
            print(f"  h={h:>6}   {pcts}")


def multivariate_pipeline(df: pd.DataFrame) -> None:
    """
    Pipeline multivariat complet — Cerinta 7.
    Johansen → VAR sau VECM → Granger → IRF → FEVD.
    """
    _header("8. ANALIZA MULTIVARIATA (VAR / VECM)")

    df_mv   = prepare_mv_data(df)
    df_diff = df_mv.diff().dropna()

    print(f"\nVariabile (log-niveluri): {df_mv.columns.tolist()}")
    print(f"Observatii: {len(df_mv)}\n")

    _sub("Stationaritate variabile (log-nivel si D1)")
    for col in df_mv.columns:
        test_adf(df_mv[col],                 col)
        test_adf(df_mv[col].diff().dropna(), f"D_{col}")

    granger_causality(df_diff, max_lag=6)

    rank = johansen_test(df_mv, det_order=0, k_ar_diff=1)

    if rank == 0:
        _sub("VAR in prime diferente (fara cointegrare)")
        lag     = select_var_lag(df_diff, max_lag=12)
        var_fit = fit_var(df_diff, lag)
        print(var_fit.summary())
        plot_irf(var_fit,  periods=24, name="VAR_diff")
        plot_fevd(var_fit, periods=24, name="VAR_diff")
    else:
        _sub(f"VECM (rang cointegrare = {rank})")
        fit_vecm(df_mv, coint_rank=rank, k_ar_diff=1)
        lag     = select_var_lag(df_diff, max_lag=12)
        var_fit = fit_var(df_diff, max(1, lag))
        plot_irf(var_fit,  periods=24, name="VECM_VAR")
        plot_fevd(var_fit, periods=24, name="VECM_VAR")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Serii multivariate (log-niveluri)", fontsize=13, fontweight="bold")
    mv_colors = [COLORS["train"], COLORS["oil"], COLORS["fx"]]
    for ax, col, clr in zip(axes, df_mv.columns, mv_colors):
        ax.plot(df_mv.index, df_mv[col], color=clr, linewidth=1.2)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig13_multivariate.png", bbox_inches="tight")
    plt.show()
