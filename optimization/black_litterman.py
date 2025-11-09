import numpy as np
from pypfopt.black_litterman import BlackLittermanModel

def bl_minimal(Sigma, w_mkt, delta, tau=0.05, P=None, Q=None, Omega=None, omega_scale=1.0):
    """
    Implementacja modelu Blacka–Littermana z użyciem PyPortfolioOpt.

    Cel:
    ----
    Połączyć „implied returns” z RP (zależne od macierzy kowariancji, wagi portfela RP
    i współczynnika awersji do ryzyka) z subiektywnymi poglądami (P, Q, Omega), aby otrzymać
    posterior (mu_bl), a następnie wyznaczyć wagi.

    Uwagi:
    ------
    - Gdy P i Q są None (brak poglądów), BL redukuje się do priory („rynkowych”) i wagi w_bl
      wyjdą równe w_mkt (bo pi = delta * Sigma @ w_mkt => w = (1/delta) * Sigma^{-1} pi = w_mkt).
    - Wagi w_bl nie są ograniczane (mogą wyjść spoza [0,1] i nie sumować się do 1) — to czysty MV.
    """

    # Przygotowanie macierzy kowariancji
    Sigma = np.asarray(Sigma, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float).reshape(-1)

    n = Sigma.shape[0] # Liczba aktywów

    # Priory: implied returns
    pi = delta * (Sigma @ w_mkt)

    # Przygotowanie macierzy poglądów
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float).reshape(-1)
    k = P.shape[0] # Liczba poglądów
    Omega = np.asarray(Omega, dtype=float)

    # Model
    bl = BlackLittermanModel(
        cov_matrix=Sigma,
        pi=pi,
        P=P,
        Q=Q,
        tau=tau,
        omega=Omega
    )

    # Wyciągamy zwroty oczekiwane posteriori
    mu_bl = bl.bl_returns()

    # Wyznaczamy wagi
    inv_Sigma = np.linalg.inv(Sigma)
    w_bl = (1.0 / delta) * (inv_Sigma @ mu_bl) # Klasyczny Markowitz

    return {'pi': pi, 'mu_bl': mu_bl, 'w_bl': w_bl, 'Omega': Omega}