import numpy as np

def project_boxed_simplex(v, lb=0.05, ub=0.12, s=1.0, tol=1e-12, it=100):
    """
    Rzutuje wektor v na tzw. „ograniczony sympleks” (boxed simplex), czyli zbiór wektorów
    wag o zadanej sumie s i z ograniczeniami dolnymi/górnymi:

        { w : sum(w) = s oraz lb_i <= w_i <= ub_i dla każdego i }

    Z praktycznego punktu widzenia:
        - na wejściu mamy „surowy” wektor v (np. wagi z modelu),
        - na wyjściu dostajemy wektor w, który:
            * ma sumę równą s (np. 1.0),
            * spełnia ograniczenia lb_i <= w_i <= ub_i,
            * jest możliwie podobny do v.

    Jak działa algorytm:
        1) Szukamy jednej liczby t (wspólnego przesunięcia),
        2) Odejmujemy t od wszystkich elementów v,
        3) Wynik przycinamy do przedziału [lb, ub] funkcją clip,
        4) Dobieramy t tak, aby suma powstałego wektora była równa s.

    Innymi słowy szukamy takiego t, że:
        sum( clip(v - t, lb, ub) ) = s

    Liczbę t znajdujemy numerycznie metodą bisekcji (zawężamy przedział,
    w którym leży rozwiązanie, aż do uzyskania wymaganej dokładności).
    """

    # Zamieniamy v na wektor 1D typu float
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
