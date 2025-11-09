import pandas as pd
import numpy as np
import re

def _parse_pln(x):
    """
    Parsuje liczby zapisane po polsku/angielsku, np.:
    '2 915,00 zł', '4\xa0241.72', '4 241,72 PLN', '1.234,56', '1234.56'
    """
    if x is None:
        return np.nan
    s = str(x).strip()

    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan

    # Usuń walutę i białe znaki specjalne
    s = (s.replace("zł", "")
           .replace("PLN", "")
           .replace("\xa0", "")
           .replace("\u202f", "")
           .replace(" ", "")
           .strip())

    # Jeżeli mamy jednocześnie kropki i przecinki, to typowo: kropki = tysiące, przecinek = dziesiętne
    # Np. "1.234,56" -> usuń kropki, przecinek zamień na kropkę.
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        if "," in s:
            s = s.replace(",", ".")

    # Usuń wszystkie znaki poza cyframi, kropką i minusem (na wszelki wypadek)
    s = re.sub(r"[^0-9\.-]", "", s)

    try:
        return float(s)
    except ValueError:
        return np.nan

def load_valuation_sheet(path):
    """
    Czyta arkusz z danymi wyceny (excel) i zwraca tabelę z:
    Ticker, TargetPrice, PriceAtPublication, Confidence, Upside
    """
    df = pd.read_excel(path)

    # Poprawiamy tickery
    df["Ticker"] = (
        df["Ticker"].astype(str)
        .str.replace("WSE:", "", regex=False)  # Usuwamy prefiks
        .str.strip().str.upper()
        .apply(lambda x: x if x.endswith(".WA") else f"{x}.WA")
    )

    # Poprawiamy ceny
    df["TargetPrice"] = df["Cena docelowa"].apply(_parse_pln)
    df["PriceAtPublication"] = df["Cena przy publikacji"].apply(_parse_pln)

    # Pewność wyceny
    df["Confidence"] = df["Zaufanie"]
    conf = df["Confidence"].astype(str).str.replace(",", ".", regex=False).astype(float)
    conf = np.where(conf > 1.0, conf / 100.0, conf)
    df["Confidence"] = np.clip(conf, 1e-6, 1.0)

    # Upside / Views
    df["Views"] = df["TargetPrice"] / df["PriceAtPublication"] - 1

    # Wybieramy kolumny
    return df[["Ticker", "TargetPrice", "PriceAtPublication", "Confidence", "Views"]]

def load_tickers_from_valuation(path):
    """Zwraca listę tickerów z arkusza (np. ['PKN.WA', 'CDR.WA'])"""
    df = load_valuation_sheet(path)
    return sorted(df["Ticker"].dropna().unique().tolist())
