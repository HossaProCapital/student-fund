from data.valuation_loader import load_valuation_sheet
import pandas as pd
import sys

def filter_upside(path: str, min_upside: float = 0.2) -> pd.DataFrame:
    """
    Zwraca DataFrame tylko z tymi spółkami, które mają 'Views' >= min_upside.
    Wynik jest posortowany malejąco po kolumnie 'Views'.
    """

    # Wczytanie arkusza wyceny
    df = load_valuation_sheet(path)

    # Filtrowanie spółek z wystarczającym 'upside'
    filtered_df = df[df["Views"] >= min_upside]

    # Sortowanie malejąco po 'Views' i reset indeksu
    filtered_df = filtered_df.sort_values("Views", ascending=False).reset_index(drop=True)

    return filtered_df