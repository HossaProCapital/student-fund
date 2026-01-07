import numpy as np
import pandas as pd


def compute_basic_risk_snapshot_main(
    *,
    prices: pd.DataFrame,
    holdings: pd.Series,
    cash: float,
    risk_window_days: int,
    var_conf: float,
    var_h: int,
    trading_days: int,
    r_f: float = 0.0,
    benchmark: pd.Series | None = None,
):
    """
    Dodatkowe podstawowe wskaźniki ryzyka dla Summary w Excelu:
    Sharpe, TE, IR, Beta + VaR/ES/MDD/vol.
    """

    # Ceny i wartości
    last = prices.ffill().iloc[-1]
    equity_value = float((holdings * last).sum())
    nav = equity_value + float(cash)

    # Zwroty
    rets = prices.pct_change().dropna()
    if len(rets) == 0:
        raise ValueError("Za mało danych cenowych.")

    window = rets.tail(risk_window_days)
    weights = (holdings * last) / equity_value if equity_value > 0 else pd.Series(0, index=holdings.index)
    port_ret = window.mul(weights, axis=1).sum(axis=1)

    # Volatility
    daily_vol = float(port_ret.std())
    annual_vol = daily_vol * np.sqrt(trading_days)

    # VaR / ES
    alpha = 1 - var_conf
    var_1d_ret = float(port_ret.quantile(alpha))
    es_1d_ret = float(port_ret[port_ret <= var_1d_ret].mean())

    var_1d = -var_1d_ret * nav
    es_1d = -es_1d_ret * nav

    var_h_port = var_1d * np.sqrt(var_h)
    es_h_port = es_1d * np.sqrt(var_h)

    # Max Drawdown
    port_price = (1 + port_ret).cumprod()
    run_max = port_price.cummax()
    mdd = float((port_price / run_max - 1).min())

    # Wskaźniki zależne od benchmarku
    sharpe = None
    if len(port_ret) >= 10 and daily_vol > 0:
        r_f_daily = (1 + r_f) ** (1 / trading_days) - 1
        sharpe = float((port_ret.mean() - r_f_daily) / daily_vol)

    tracking_error = None
    information_ratio = None
    beta = None

    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        bench_ret = bench_ret.tail(risk_window_days)
        p, b = port_ret.align(bench_ret, join="inner")

        if len(p) > 10:
            if p.std() > 0:
                sharpe = float((p.mean() - r_f) / p.std())

            diff = p - b
            tracking_error = float(diff.std())

            if tracking_error and tracking_error > 0:
                information_ratio = float(diff.mean() / tracking_error)

            cov = np.cov(p, b)
            if cov[1, 1] > 0:
                beta = float(cov[0, 1] / cov[1, 1])

    else:
        # Brak benchmarku → Sharpe na dzisiaj będzie None
        pass

    return {
        "nav": nav,
        "daily_vol": daily_vol,
        "annual_vol": annual_vol,
        "var_1d": var_1d,
        "es_1d": es_1d,
        "var_h": var_h_port,
        "es_h": es_h_port,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
    }
