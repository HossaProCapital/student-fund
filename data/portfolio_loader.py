import numpy as np
import pandas as pd

BUY, SELL = {"BUY"}, {"SELL"} # Transakcje giełdowe

def _to_float(col: pd.Series):
    return pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False),
                         errors="coerce").fillna(0.0).to_numpy()

def load_trades(path):
    """
    Czyta Excela i zwraca:
      - trades: [Data, Ticker, Typ, Ilość] tylko dla BUY/SELL
      - cash_balance: float = wpłaty + SELL - BUY
    """
    df = pd.read_excel(path, dtype=str)

    # Sprawdzamy czy są kolumny, które potrzebujemy
    req = ["Ticker", "Liczba akcji", "Czynność",
           "Data zrealizowania transakcji", "Wartość kupna/sprzedaży",
           "Wpłacona kwota"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Brak kolumn lub literówka: {miss}")

    # Normalizujemy kolumny
    df["Typ"] = df["Czynność"].astype(str).str.upper().str.strip()
    df["Data"] = pd.to_datetime(df["Data zrealizowania transakcji"], errors="coerce")
    df["Ticker"] = (df["Ticker"].astype(str)
                              .str.replace("WSE:", "", regex=False)
                              .str.strip().str.upper()
                              .apply(lambda x: f"{x}.WA" if x and not x.endswith(".WA") else x))
    df["Ilosc"] = _to_float(df["Liczba akcji"])
    df["Wartosc_tx"] = _to_float(df["Wartość kupna/sprzedaży"])

    # Gotówka
    cash_in = _to_float(df["Wpłacona kwota"]).sum()
    
    buy_spent   = df.loc[df["Typ"].isin(BUY),  "Wartosc_tx"].sum() # Wydatki na BUY
    sell_income = df.loc[df["Typ"].isin(SELL), "Wartosc_tx"].sum() # Wpływy z SELL
    cash_balance = float(cash_in + sell_income - buy_spent)

    # Transakcje
    trades = (df.loc[df["Typ"].isin(BUY | SELL), ["Data", "Ticker", "Typ", "Ilosc"]]
                .dropna(subset=["Data", "Ticker"])
                .sort_values("Data")
                .reset_index(drop=True))

    return trades, cash_balance


def build_holdings(trades: pd.DataFrame):
    """
    Funkcja przelicza transakcje (BUY / SELL) na:
      - końcowy stan posiadania akcji (holdings),
      - historię zmian portfela po każdej transakcji (holdings_history).
    """

    tickers = trades["Ticker"].unique()
    
    # Słownik ticker : numer kolumny (np. {'DOM.WA':0, 'NEU.WA':1})
    t2i = {t: i for i, t in enumerate(tickers)}
    
    # m = transakcje (wiersze)
    # n = spółki (kolumny)
    m, n = len(trades), len(tickers)

    # Liczba akcji w każdej transakcji
    qty = trades["Ilosc"].to_numpy(dtype=float)
    
    # BUY - +1, SELL - -1 (określa kierunek transakcji)
    # Wtedy BUY 10 = +10, SELL 5 = -5    
    sign = np.where(trades["Typ"].to_numpy() == "BUY", +1.0, -1.0)
    
    # Tworzymy macierz
    flow = np.zeros((m, n), dtype=float)

    # Przechodzimy po wszystkich transakcjach (wierszach)
    for r, (tic, s, q) in enumerate(zip(trades["Ticker"], sign, qty)):
        # Dla danego tickera wpisujemy +ilość lub -ilość
        # np. BUY 10 DOM = +10 w kolumnie DOM
        # np. SELL 3 NEU = -3 w kolumnie NEU
        flow[r, t2i[tic]] = s * q

    # Sumujemy transakcje
    cum = np.cumsum(flow, axis=0)
    
    # Stan portfela + historia transakcji
    holdings = pd.Series(cum[-1], index=tickers).sort_index()
    history = (pd.DataFrame(cum, index=trades["Data"], columns=tickers)
                 .sort_index()
                 .ffill())

    return holdings, history
