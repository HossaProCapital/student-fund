import numpy as np
import pandas as pd

# Zwroty log i proste
def returns(prices: pd.DataFrame, log: bool = True):
    """Zwraca dzienne zwroty log lub proste, bez pustych wierszy."""
    prices = prices.copy()
    return (np.log(prices / prices.shift(1)) if log else prices.pct_change()).dropna(how="any")

def to_simple(r: pd.Series, log: bool):
    """Zamienia log-zwroty na proste (jeśli trzeba do np. Max Drawdown)."""
    return np.expm1(r) if log else r

def portfolio_nav_and_weights(prices: pd.DataFrame, holdings: pd.Series):
    """
    Z ostatnich cen i liczby akcji:
    - wyznacza NAV (wartość portfela),
    - oblicza udziały (wagi) każdej pozycji.
    """
    last = prices.iloc[-1].ffill()
    qty = pd.to_numeric(holdings.reindex(last.index), errors="coerce").fillna(0.0)
    values = qty * last
    nav = float(values.sum()) if values.sum() != 0 else 1.0
    weights_map = (values / nav).fillna(0.0)
    weights_vec = weights_map.to_numpy() # Jaki procent nav przypada na każdą spółkę
    return nav, weights_map, weights_vec
