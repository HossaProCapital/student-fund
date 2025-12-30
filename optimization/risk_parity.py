import numpy as np
import pandas as pd
from scipy.optimize import minimize


def shrink_cov(returns, eps=1e-8):
    """
    Czyścimy dane i liczymy kowariancję.
    Dodajemy mały 'ridge' na przekątnej (eps).
    """
    rs = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    Sigma = rs.cov()
    Sigma.values[np.diag_indices_from(Sigma)] += eps
    return Sigma


def risk_parity_weights(
    Sigma: pd.DataFrame,
    w_min: float = 0.0,
    w_max: float = 1.0,
    *,
    sector_map: dict | None = None,
    max_sector_weight: float | None = None,
):
    """
    Minimalna RP: SLSQP na udziały w ryzyku, ograniczenia:
    - sum(w)=1
    - w ∈ [w_min, w_max]
    - (opcjonalnie) sum_{i in sektor} w_i <= max_sector_weight

    Zwraca wagi jako Series w kolejności indeksu Sigma.
    """
    S = Sigma.values
    n = S.shape[0]
    tickers = list(Sigma.index)

    if max_sector_weight is not None:
        if not (0.0 < float(max_sector_weight) <= 1.0):
            raise ValueError(f"max_sector_weight musi być w (0,1], a jest: {max_sector_weight}")

        if sector_map is None:
            raise ValueError("Podano max_sector_weight, ale sector_map=None (brak mapy sektorów).")

        missing = [t for t in tickers if t not in sector_map or sector_map[t] is None]
        if missing:
            raise ValueError(f"Brak sektora dla tickerów (w RP): {missing}")

    x0 = np.full(n, 1.0 / n)

    def obj(w):
        Sw = S @ w
        sigma2 = max(w @ Sw, 1e-16)
        rc_share = (w * Sw) / sigma2
        return np.sum((rc_share - 1.0 / n) ** 2)

    bounds = [(w_min, w_max)] * n

    # constraints jako LISTA (łatwo dopinać kolejne)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # --- sektorowe constrainty: max na sektor ---
    if max_sector_weight is not None and sector_map is not None:
        sec_to_idx: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            sec = sector_map[t]
            sec_to_idx.setdefault(str(sec), []).append(i)

        for sec, idx_list in sec_to_idx.items():
            idx = np.array(idx_list, dtype=int)

            # musi być >= 0 dla 'ineq'
            # max_sector_weight - sum(w[idx]) >= 0
            cons.append({
                "type": "ineq",
                "fun": (lambda w, idx=idx: float(max_sector_weight) - float(np.sum(w[idx])))
            })

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )

    if not res.success:
        raise RuntimeError(f"RP optimizer failed: {res.message}")

    w = np.clip(res.x, w_min, w_max)
    s = w.sum()
    if s <= 0:
        raise RuntimeError("RP optimizer zwrócił wagi o sumie 0 (nieprawidłowe rozwiązanie).")
    w = w / s

    return pd.Series(w, index=Sigma.index, name="w_RP")
