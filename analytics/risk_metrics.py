import numpy as np
import pandas as pd
from .risk_utils import returns, to_simple, portfolio_nav_and_weights

# Empirycznie (na danych historycznych)
def compute_empirical_risk(
    prices: pd.DataFrame,
    holdings: pd.Series,
    horizon_days: int,
    trading_days: int,
    risk_window_days: int,
    use_log_returns: bool = True,
    confidence: float = 0.99
):
    """
    Oblicza empiryczne (historyczne) miary ryzyka portfela inwestycyjnego,
    czyli takie, które nie zakładają żadnego konkretnego rozkładu danych
    (np. normalnego). Wszystkie wskaźniki liczone są bezpośrednio z 
    historycznych obserwacji zwrotów.
    """

    # Całkowita wartość portfela + wagi z ostatnich cen
    nav, weights_map, weights = portfolio_nav_and_weights(prices, holdings)

    # Zwroty portfela
    rets = returns(prices, log=use_log_returns).dropna(axis=1, how="all")
    rets = rets.tail(risk_window_days)
    rets = rets.reindex(columns=weights_map.index).fillna(0.0)
    port_rets_log = pd.Series(rets.to_numpy() @ weights, index=rets.index, name="Rp_log")
    port_rets_simple = to_simple(port_rets_log, use_log_returns)

    # Odch. stand. (dzienne) na log-zwrotach / Zmienność dzienna (+ roczne)
    daily_vol = float(np.std(port_rets_log, ddof=1))
    annual_vol = daily_vol * np.sqrt(trading_days)

    # VaR/ES historyczny (na prostych stopach)
    alpha = 1.0 - confidence
    var_1d_ret = float(np.quantile(port_rets_simple, alpha))
    tail = port_rets_simple[port_rets_simple <= var_1d_ret]
    es_1d_ret = float(tail.mean()) if len(tail) else var_1d_ret

    root_h = np.sqrt(max(int(horizon_days), 1))
    var_1d = -var_1d_ret * nav
    es_1d  = -es_1d_ret  * nav
    var_h  = var_1d * root_h
    es_h   = es_1d * root_h

    # Max Drawdown (na prostych)
    cum = (1.0 + port_rets_simple).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1.0
    mdd = float(drawdown.min())

    return {
        "nav": nav,
        "weights": weights,
        "weights_map": weights_map,
        "daily_vol": daily_vol,
        "annual_vol": annual_vol,
        "var_1d_ret": var_1d_ret,
        "es_1d_ret": es_1d_ret,
        "var_1d": var_1d,
        "es_1d": es_1d,
        "var_h": var_h,
        "es_h": es_h,
        "max_drawdown": mdd,
        "covariance": None
    }