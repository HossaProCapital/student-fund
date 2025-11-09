import numpy as np

def project_boxed_simplex(v, lb=0.05, ub=0.12, s=1.0, tol=1e-12, it=100):
    """
    Projekcja wektora v na tzw. „ograniczony sympleks” (boxed simplex):
        { w : sum(w) = s,  lb_i <= w_i <= ub_i }

    Oznacza to, że szukamy takiego wektora w, który:
        - leży możliwie blisko v (w sensie odjęcia jednej stałej od wszystkich elementów),
        - ma sumę równą s,
        - i każdy składnik w_i jest w przedziale [lb_i, ub_i].

    Idea:
    -----
    Szukamy jednej liczby t (przesunięcia), takiej że po przycięciu wartości (v - t)
    do przedziału [lb, ub], ich suma wynosi dokładnie s.

    Formalnie: znajdź t takie, że sum( clip(v - t, lb, ub) ) = s.
    To równanie rozwiązywane jest metodą bisekcji.
    """

    # Zamieniamy v na wektor 1D typu floa
    v  = np.ravel(v).astype(float)

    # Rozszerzamy lb i ub do tego samego kształtu co v (jeśli są skalarami)
    lb = np.broadcast_to(lb, v.shape).astype(float)
    ub = np.broadcast_to(ub, v.shape).astype(float)

    # Obliczamy minimalną i maksymalną możliwą sumę elementów
    s_min, s_max = lb.sum(), ub.sum()

    # Jeśli żądana suma jest mniejsza niż najmniejsza możliwa, zwracamy dolne granice
    if s <= s_min + tol:
        return lb.copy()

    # Jeśli żądana suma jest większa niż największa możliwa, zwracamy górne granice
    if s >= s_max - tol:
        return ub.copy()

    # Wyznaczamy początkowy zakres dla parametru t:
    #  lo -> taki, że wszystkie v - lo >= ub (czyli suma maksymalna)
    #  hi -> taki, że wszystkie v - hi <= lb (czyli suma minimalna)
    lo, hi = (v - ub).min(), (v - lb).max()

    # Główna pętla bisekcji – szukamy t, które daje sumę bliską s
    for _ in range(it):
        # Środek aktualnego przedziału
        t  = (lo + hi) / 2.0

        # Obliczamy wektor w = clip(v - t, lb, ub)
        # Czyli przesuwamy wszystkie wartości v o t, a następnie przycinamy do [lb, ub]
        w  = np.clip(v - t, lb, ub)

        # Liczymy sumę tego wektora
        sm = w.sum()

        # Sprawdzamy, czy suma jest wystarczająco bliska żądanej
        if abs(sm - s) <= tol:
            return w  # sukces – znaleźliśmy odpowiednie t

        # Jeśli suma jest za duża, t musimy zwiększyć (więcej „ucięcia”)
        # Jeśli suma za mała, t musimy zmniejszyć (mniej „ucięcia”)
        (lo, hi) = (t, hi) if sm > s else (lo, t)

    # Jeśli pętla się skończyła bierzemy najlepszy środek jako przybliżenie rozwiązania
    return np.clip(v - (lo + hi) / 2.0, lb, ub)
