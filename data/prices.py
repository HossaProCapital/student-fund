import yfinance as yf

def get_prices(tickers, start_date, end_date=None):
    """
    Pobiera ceny z Yahoo Finance dla podanych tickerów.
    Zwraca DataFrame z kolumnami (tickery) i wierszami (daty).
    """

    def norm(t):
        t = str(t).replace("WSE:", "").strip().upper()
        return t if t.endswith(".WA") else f"{t}.WA"
    if isinstance(tickers, (list, tuple, set)):
        tickers = [norm(t) for t in tickers]
    else:
        tickers = norm(tickers)

    # Pobieramy dane
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True # Gdy True wtedy Close = AdjClose
    )

    prices = data['Close']

    # Usuwamy dni bez notowań (np. weekendy)
    prices = prices.dropna(how="all")

    return prices
