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
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        if "," in s:
            s = s.replace(",", ".")

    # Usuń wszystkie znaki poza cyframi, kropką i minusem
    s = re.sub(r"[^0-9\.-]", "", s)

    try:
        return float(s)
    except ValueError:
        return np.nan


def load_valuation_sheet(path: str) -> pd.DataFrame:
    """
    Czyta arkusz z danymi wyceny (excel) i zwraca tabelę z:
    Ticker, ValuationDate, TargetPrice, PriceAtPublication, Confidence, Views, Sector

    UWAGA:
    - Jeśli ten sam ticker jest w arkuszu wiele razy, bierzemy NAJNOWSZĄ wycenę po 'Data wyceny'
      (a jeśli brak daty - bierzemy ostatni wiersz po sortowaniu).
    """
    df = pd.read_excel(path)

    # Ujednolicenie nazw kolumn (czasem Excel ma spacje)
    df.columns = df.columns.astype(str).str.strip()

    # Normalizacja tickerów
    df["Ticker"] = (
        df["Ticker"].astype(str)
        .str.replace("WSE:", "", regex=False)
        .str.strip()
        .str.upper()
        .apply(lambda x: x if x.endswith(".WA") else f"{x}.WA")
    )

    # Daty wyceny (jeśli są)
    if "Data wyceny" in df.columns:
        df["ValuationDate"] = pd.to_datetime(df["Data wyceny"], errors="coerce")
    else:
        df["ValuationDate"] = pd.NaT

    # Ceny
    df["TargetPrice"] = df["Cena docelowa"].apply(_parse_pln)
    df["PriceAtPublication"] = df["Cena przy publikacji"].apply(_parse_pln)

    # Pewność (Zaufanie)
    df["Confidence"] = df["Zaufanie"]
    conf = df["Confidence"].astype(str).str.replace(",", ".", regex=False)
    conf = pd.to_numeric(conf, errors="coerce")
    conf = np.where(conf > 1.0, conf / 100.0, conf)  # jeśli wpisane np. 60 zamiast 0.6
    df["Confidence"] = np.clip(conf, 1e-6, 1.0)

    # Upside / Views
    df["Views"] = df["TargetPrice"] / df["PriceAtPublication"] - 1

    # Sektor (kolumna "Sektor" w portfolio2.xlsx)
    if "Sektor" in df.columns:
        df["Sector"] = df["Sektor"].astype(str).str.strip()
        df.loc[df["Sector"].isin(["", "nan", "None"]), "Sector"] = np.nan
    else:
        df["Sector"] = np.nan

    # Duplikaty tickerów -> zostawiamy najnowszą wycenę
    df = df.sort_values(["Ticker", "ValuationDate"]).drop_duplicates("Ticker", keep="last")

    return df[["Ticker", "ValuationDate", "TargetPrice", "PriceAtPublication", "Confidence", "Views", "Sector"]]


def load_tickers_from_valuation(path: str) -> list[str]:
    """Zwraca listę tickerów z arkusza (np. ['PKN.WA', 'CDR.WA'])."""
    df = load_valuation_sheet(path)
    return sorted(df["Ticker"].dropna().unique().tolist())
