from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------------------------------------------
# 1. MODEL DANYCH – spójny obiekt dla kart ryzyka i historii
# ---------------------------------------------------------

@dataclass
class RiskSnapshot:
    date: datetime

    # wartości portfela
    nav: float
    cash: float
    equity_value: float
    weights: pd.Series        # wagi portfela (float)

    # zmienność
    daily_vol: float
    annual_vol: float

    # VaR / ES
    var_1d: float
    es_1d: float
    var_h: float
    es_h: float

    # inne ryzyko
    max_drawdown: float

    # wskaźniki rynku
    sharpe: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    beta: Optional[float] = None

    # udział w ryzyku
    rc: Optional[pd.Series] = None     # Risk Contribution (%)
    mrc: Optional[pd.Series] = None    # Marginal RC


# ---------------------------------------------------------
# 2. FUNKCJE POMOCNICZE
# ---------------------------------------------------------

def _compute_drawdown(prices: pd.Series) -> float:
    """Max drawdown na podstawie szeregu cen portfela."""
    running_max = prices.cummax()
    dd = (prices - running_max) / running_max
    return dd.min()


def _compute_var_es(ret: pd.Series, confidence: float) -> tuple[float, float]:
    """Empiryczny VaR i ES dla rozkładu zwrotów."""
    ret = ret.dropna()
    if len(ret) < 10:
        return np.nan, np.nan

    var = ret.quantile(1 - confidence)
    es = ret[ret <= var].mean()
    return float(var), float(es)


def _compute_risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    RC_i = w_i * (Σ w)_i
    MRC_i = (Σ w)_i
    Zwraca RC% (udział procentowy) oraz MRC.
    """
    w = weights.values
    Sigma = cov.values

    marginal = Sigma @ w                   # MRC
    total_variance = float(w.T @ marginal)

    if total_variance <= 0:
        rc_pct = np.zeros_like(w)
    else:
        rc = w * marginal
        rc_pct = rc / total_variance       # procentowy udział w ryzyku

    return (
        pd.Series(rc_pct, index=weights.index),
        pd.Series(marginal, index=weights.index)
    )


# ---------------------------------------------------------
# 3. GŁÓWNA FUNKCJA RYZYKA
# ---------------------------------------------------------

def compute_risk_snapshot(
    *,
    prices: pd.DataFrame,
    holdings: pd.Series,
    cash: float = 0.0,
    risk_window_days: int = 252,
    var_conf: float = 0.99,
    var_h: int = 20,
    trading_days: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    r_f: float = 0.0,
) -> RiskSnapshot:
    """
    Liczy kompletny zestaw metryk ryzyka dla danego dnia.
    Zwraca obiekt RiskSnapshot – baza do kart ryzyka i historii.
    """

    # --- NAV, wartości ---
    last_prices = prices.ffill().iloc[-1]
    equity_value = float((holdings * last_prices).sum())
    nav = equity_value + float(cash)

    # --- Zwroty ---
    rets = prices.pct_change().dropna()
    if len(rets) == 0:
        raise ValueError("Za mało danych cenowych do policzenia ryzyka.")

    window_rets = rets.tail(risk_window_days)

    # --- Wagi portfela ---
    weights = (holdings * last_prices) / equity_value if equity_value > 0 else pd.Series(0, index=holdings.index)

    # --- Zmienność ---
    daily_vol = float(window_rets.mul(weights, axis=1).sum(axis=1).std())
    annual_vol = daily_vol * np.sqrt(trading_days)

    # --- VaR i ES ---
    port_ret = window_rets.mul(weights, axis=1).sum(axis=1)
    var_1d, es_1d = _compute_var_es(port_ret, var_conf)
    var_h = var_1d * np.sqrt(var_h)
    es_h = es_1d * np.sqrt(var_h)

    # --- Max drawdown ---
    port_price = (1 + port_ret).cumprod()
    max_dd = float(_compute_drawdown(port_price))

    # --- Benchmark metrics ---
    sharpe = None
    tracking_error = None
    information_ratio = None
    beta = None

    if benchmark_returns is not None and not benchmark_returns.dropna().empty:
        bench = benchmark_returns.tail(risk_window_days).dropna()
        common = port_ret.align(bench, join="inner")

        if len(common[0]) > 10:
            excess = common[0] - common[1]

            # Sharpe
            sharpe = float((common[0].mean() - r_f) / common[0].std()) if common[0].std() > 0 else None

            # TE
            tracking_error = float(excess.std())

            # IR
            information_ratio = float(excess.mean() / tracking_error) if tracking_error and tracking_error > 0 else None

            # Beta
            cov = np.cov(common[0], common[1])
            var_b = cov[1, 1]
            beta = float(cov[0, 1] / var_b) if var_b > 0 else None

    # --- Risk contributions ---
    cov_mat = window_rets.cov()
    rc_pct, mrc = _compute_risk_contributions(weights, cov_mat)

    # --- Snapshot ---
    return RiskSnapshot(
        date=datetime.today(),
        nav=nav,
        cash=cash,
        equity_value=equity_value,
        weights=weights,
        daily_vol=daily_vol,
        annual_vol=annual_vol,
        var_1d=var_1d,
        es_1d=es_1d,
        var_h=var_h,
        es_h=es_h,
        max_drawdown=max_dd,
        sharpe=sharpe,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        beta=beta,
        rc=rc_pct,
        mrc=mrc,
    )
