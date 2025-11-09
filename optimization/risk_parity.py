import numpy as np
import pandas as pd
from scipy.optimize import minimize

def shrink_cov(returns, eps=1e-8):
    """
    Czyścimy dane i liczymy kowariancję.
    Dodajemy mały 'ridge' na przekątnej (eps).
    """

    rs = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any") # Czyścimy

    Sigma = rs.cov()
    Sigma.values[np.diag_indices_from(Sigma)] += eps # Dla zabezpieczenia przed macierzą osobliwą
    return Sigma


def risk_parity_weights(Sigma, w_min=0.0, w_max=1.0):
    """
    Minimalna RP: SLSQP na udziały w ryzyku, ograniczenia: sum(w)=1, w∈[w_min, w_max].
    Zwraca wagi jako Series w kolejności indeksu Sigma.
    """

    S = Sigma.values
    n = S.shape[0] # Liczba spółek

    x0 = np.full(n, 1.0 / n) # Startujemy od równych wag

    def obj(w):
        Sw = S @ w # Udział w całkowitej wariancji dla każdego aktywa = funkcja celu
        sigma2 = max(w @ Sw, 1e-16) # Wariancja portfela
        rc_share = (w * Sw) / sigma2 # Udział ryzyka
        return np.sum((rc_share - 1.0 / n) ** 2) # Wszystkie udziały chcemy równe 1/n

    bounds = [(w_min, w_max)] * n # Przedziały
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},) # Suma wag = 1

    # Solver
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})

    # Przycinamy wagi (błędy numeryczne)
    w = np.clip(res.x, w_min, w_max)

    # Normalizujemy po przycięciu
    w = w / w.sum()

    return pd.Series(w, index=Sigma.index, name="w_RP")
