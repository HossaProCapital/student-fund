"""
analytics/stock_risk_analyzer.py

Funkcje do tworzenia kart ryzyka pojedynczych spółek.

"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from .risk_utils import returns

logger = logging.getLogger(__name__)


def _get_returns(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    window = int(cfg.get("risk_window_days", 252))
    use_log = bool(cfg.get("use_log_returns", True))

    rets = returns(prices, log=use_log)
    if window and len(rets) > window:
        rets = rets.tail(window)
    return rets


def _squeeze_if_needed(x: Union[pd.Series, pd.DataFrame, np.ndarray, float]) -> Union[float, pd.Series]:
    """Jeśli wynik jest jednoelementowym Series/array -> scalar, w przeciwnym wypadku zwróć oryginał."""
    if isinstance(x, pd.DataFrame):
        # jeśli DataFrame ma jedną kolumnę zwracamy Series, inne zostawiamy
        if x.shape[1] == 1:
            return x.iloc[:, 0].squeeze()
        return x
    if isinstance(x, pd.Series):
        return x.squeeze()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x
    return x


def _to_float(value) -> float:
    """Konwertuje jednoelementowy wynik na float; dla wieloelementowych -> raise (wyłapie błąd wcześniej)."""
    v = _squeeze_if_needed(value)
    if isinstance(v, (pd.Series, np.ndarray)):
        arr = np.asarray(v)
        if arr.size == 1:
            return float(arr.item())
        raise ValueError("Expected single numeric value, got multiple values.")
    try:
        fv = float(v)
    except Exception:
        return np.nan
    return fv if not np.isinf(fv) else np.nan


def calculate_volatility(prices: pd.DataFrame, cfg: dict) -> float:
    trading_days = int(cfg.get("trading_days", 252))
    rets = _get_returns(prices, cfg)
    std_daily = _squeeze_if_needed(rets.std())
    if std_daily is None or (isinstance(std_daily, float) and np.isnan(std_daily)):
        return np.nan
    return _to_float(std_daily * np.sqrt(trading_days))


def calculate_downside_risk(prices: pd.DataFrame, cfg: dict) -> float:
    trading_days = int(cfg.get("trading_days", 252))
    rets = _get_returns(prices, cfg)

    mean_return = _squeeze_if_needed(rets.mean())
    downside_rets = rets[rets < mean_return]

    # jeśli brak obserwacji poniżej średniej
    if getattr(downside_rets, "size", 0) == 0:
        return 0.0

    sq = ((downside_rets - mean_return) ** 2).to_numpy()
    # mean ignoruje NaN; axis=0 -> per-column if applicable
    var_down = np.nanmean(sq, axis=0)
    semi_dev_daily = np.sqrt(var_down)
    semi_dev_annual = semi_dev_daily * np.sqrt(trading_days)

    arr = np.asarray(semi_dev_annual)
    if np.isnan(arr).any():
        return np.nan
    if arr.size == 1:
        return float(arr.item())
    # multi-column not expected for card generator; return first element to keep behavior predictable
    return float(arr[0])


def calculate_sharpe_ratio(prices: pd.DataFrame, cfg: dict) -> float:
    trading_days = int(cfg.get("trading_days", 252))
    risk_free_rate = float(cfg.get("risk_free_rate", 0.0))
    rets = _get_returns(prices, cfg)

    mean_return = _squeeze_if_needed(rets.mean()) * trading_days
    vol = calculate_volatility(prices, cfg)

    if vol == 0 or np.isnan(vol):
        return np.nan

    return _to_float((mean_return - risk_free_rate) / vol)


def calculate_sortino_ratio(prices: pd.DataFrame, cfg: dict) -> float:
    trading_days = int(cfg.get("trading_days", 252))
    risk_free_rate = float(cfg.get("risk_free_rate", 0.0))
    rets = _get_returns(prices, cfg)

    mean_return = _squeeze_if_needed(rets.mean()) * trading_days
    daily_rf = risk_free_rate / trading_days

    downside_rets = rets[rets < daily_rf]
    if getattr(downside_rets, "size", 0) == 0:
        return np.nan

    sq = ((downside_rets - daily_rf) ** 2).to_numpy()
    var_down = np.nanmean(sq, axis=0)
    downside_dev_daily = np.sqrt(var_down)
    downside_dev_annual = downside_dev_daily * np.sqrt(trading_days)

    arr = np.asarray(downside_dev_annual)
    if np.isnan(arr).any() or np.all(arr == 0):
        return np.nan
    val = (mean_return - risk_free_rate) / arr
    # squeeze and return single float
    return _to_float(val)


def calculate_max_drawdown(prices: pd.DataFrame, cfg: dict) -> float:
    window = int(cfg.get("risk_window_days", 252))

    if window and len(prices) > window:
        prices = prices.tail(window)

    prices = prices.dropna()
    if prices.empty:
        return np.nan

    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    mdd = drawdown.min()
    mdd = _squeeze_if_needed(mdd)
    return _to_float(mdd)


def _ensure_series_from_returns(x, use_log=True):
    """Helper: returns(...) może zwrócić DataFrame lub Series — chcemy Series."""
    r = returns(x, log=use_log)
    if isinstance(r, pd.DataFrame):
        if r.shape[1] >= 1:
            return r.iloc[:, 0].squeeze()
        return r.squeeze()
    return r.squeeze()


def calculate_beta_vs_portfolio(stock_prices: pd.DataFrame, portfolio_prices: pd.Series, cfg: dict) -> float:
    try:
        use_log = bool(cfg.get("use_log_returns", True))
        lookback = int(cfg.get("beta_lookback_days", 60))
        min_periods = int(cfg.get("min_periods", 20))

        stock_rets = _ensure_series_from_returns(stock_prices, use_log).tail(lookback)
        port_rets = _ensure_series_from_returns(portfolio_prices.to_frame(), use_log).tail(lookback)

        common_idx = stock_rets.index.intersection(port_rets.index)
        stock_rets = stock_rets.loc[common_idx].dropna()
        port_rets = port_rets.loc[common_idx].dropna()

        if len(stock_rets) < min_periods:
            return np.nan

        cov = stock_rets.cov(port_rets)
        var_port = port_rets.var()
        if var_port == 0 or np.isnan(var_port):
            return np.nan
        beta = cov / var_port
        return _to_float(beta)
    except Exception as e:
        logger.exception("calculate_beta_vs_portfolio failed: %s", e)
        return np.nan


def calculate_correlation_with_portfolio(stock_prices: pd.DataFrame, portfolio_prices: pd.Series, cfg: dict) -> float:
    try:
        use_log = bool(cfg.get("use_log_returns", True))
        lookback = int(cfg.get("beta_lookback_days", 60))
        min_periods = int(cfg.get("min_periods", 20))

        stock_rets = _ensure_series_from_returns(stock_prices, use_log).tail(lookback)
        port_rets = _ensure_series_from_returns(portfolio_prices.to_frame(), use_log).tail(lookback)

        common_idx = stock_rets.index.intersection(port_rets.index)
        stock_r = stock_rets.loc[common_idx].dropna()
        port_r = port_rets.loc[common_idx].dropna()

        if len(stock_r) < min_periods:
            return np.nan
        corr = stock_r.corr(port_r)
        return _to_float(corr)
    except Exception as e:
        logger.exception("calculate_correlation_with_portfolio failed: %s", e)
        return np.nan


def create_risk_card(ticker: str, prices_all: pd.DataFrame, cfg: dict, portfolio_prices: pd.Series = None,
                     upside_from_valuation: float = None, risk_parity_contribution: float = None) -> dict:
    stock_prices = prices_all[[ticker]].dropna()
    if len(stock_prices) < 60:
        raise ValueError(f"Za mało danych dla {ticker}")

    return {
        'Ticker': ticker,
        'Zmienność (roczna)': calculate_volatility(stock_prices, cfg),
        'Ryzyko asymetryczne': calculate_downside_risk(stock_prices, cfg),
        'Sharpe Ratio': calculate_sharpe_ratio(stock_prices, cfg),
        'Sortino Ratio': calculate_sortino_ratio(stock_prices, cfg),
        'Max Drawdown': calculate_max_drawdown(stock_prices, cfg),
        'Beta vs. Portfel': calculate_beta_vs_portfolio(stock_prices, portfolio_prices,
                                                        cfg) if portfolio_prices is not None else np.nan,
        'Korelacja z portfelem': calculate_correlation_with_portfolio(stock_prices, portfolio_prices,
                                                                      cfg) if portfolio_prices is not None else np.nan,
        'Wkład w ryzyko portfela': risk_parity_contribution if risk_parity_contribution is not None else np.nan,
        'Upside (prognoza)': upside_from_valuation if upside_from_valuation is not None else np.nan,
    }


def create_risk_cards_batch(tickers: list, prices_df: pd.DataFrame, cfg: dict, portfolio_prices: pd.Series = None,
                            valuation_dict: dict = None, risk_parity_dict: dict = None) -> pd.DataFrame:
    cards = []
    for ticker in tickers:
        try:
            card = create_risk_card(
                ticker=ticker,
                prices_all=prices_df,
                cfg=cfg,
                portfolio_prices=portfolio_prices,
                upside_from_valuation=valuation_dict.get(ticker) if valuation_dict else None,
                risk_parity_contribution=risk_parity_dict.get(ticker) if risk_parity_dict else None
            )
            cards.append(card)
            print(f"✓ {ticker}")
        except Exception as e:
            print(f"✗ {ticker}: {e}")
            logger.exception("create_risk_card failed for %s: %s", ticker, e)

    if not cards:
        return pd.DataFrame()

    df = pd.DataFrame(cards)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)
    return df
