"""
Główny skrypt systemu analitycznego Funduszu Hossa Procapital.

Zakres:
- wczytanie konfiguracji z pliku config.yaml,
- rekonstrukcja stanu portfela (holdings, gotówka) na podstawie arkusza transakcji,
- wybór wszechświata spółek (tickery) i filtracja po „upside”,
- pobranie danych cenowych z Yahoo Finance,
- obliczenie empirycznych miar ryzyka portfela (VaR, ES, MDD, zmienność),
- wyznaczenie wag portfela Risk Parity (RP) i Black–Litterman (BL, BL_Box),
- wygenerowanie raportu w Excelu z podsumowaniem ryzyka i wagami portfeli.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd

from analytics.risk_metrics import compute_empirical_risk
from analytics.risk_utils import returns
from data.portfolio_loader import load_trades, build_holdings
from data.prices import get_prices
from data.valuation_loader import load_valuation_sheet, load_tickers_from_valuation
from optimization.risk_parity import shrink_cov, risk_parity_weights
from optimization.black_litterman import bl_minimal
from optimization.constraints import project_boxed_simplex
from reporting.exporter import export_report_xlsx


# FUNKCJE POMOCNICZE

def read_config(path):
    """
    Czyta plik YAML z konfiguracją (config.yaml).

    Zwraca:
        słownik z parametrami konfiguracyjnymi.
    Jeżeli plik jest pusty, zwraca pusty słownik.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def choose_tickers(cfg, trades_df):
    """
    Wybiera tickery, z którymi będziemy pracować.

    Logika:
    1) Jeżeli dostępny jest arkusz wycen (valuation_excel_path) – bierzemy tickery stamtąd.
       (To jest preferowane: pracujemy na spółkach, które mają własne „views” i wyceny.)
    2) Jeżeli arkusza wycen nie ma lub nie da się go wczytać – bierzemy tickery z arkusza transakcji.
    """

    val_path = cfg.get("valuation_excel_path")
    if val_path and Path(val_path).exists():
        try:
            return load_tickers_from_valuation(val_path)
        except Exception:
            pass

    if trades_df is not None and not trades_df.empty and "Ticker" in trades_df.columns:
        return sorted(trades_df["Ticker"].dropna().astype(str).str.upper().unique().tolist())

    return []


def filter_tickers_by_upside(valuation_path: str, tickers: list[str], min_upside_raw: float) -> list[str]:
    """
    Filtruje wszechświat tickerów po 'Views' (upside) z arkusza wycen.
    Zwraca tickery, których 'Views' >= min_upside_raw (ułamek dziesiętny).
    """

    # Wybieramy tickery
    path = Path(valuation_path)
    val = load_valuation_sheet(str(path))
    sub = val[val["Ticker"].isin(tickers)].copy()

    # Konwersja na liczby + filtr po progu
    sub["Views"] = pd.to_numeric(sub["Views"], errors="coerce")
    passed = sub.loc[sub["Views"] >= min_upside_raw, "Ticker"]

    return passed.dropna().astype(str).str.upper().tolist()


# MAIN

def main():
    # Wczytanie config.yaml
    cfg_path = Path("config.yaml")
    if not cfg_path.is_file():
        print("[ERROR] Nie znaleziono pliku konfiguracyjnego config.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"Używam konfiguracji: {cfg_path}")
    cfg = read_config(cfg_path)

    # ŚCIEŻKI DO PLIKÓW WEJŚCIOWYCH
    valuation_path = cfg.get("valuation_excel_path", "input/portfolio2.xlsx")
    trades_path = cfg.get("trades_excel_path", "input/portfolio.xlsx")
    output_path = cfg.get("output_file", "output/portfolio_risk_report.xlsx")

    # PARAMETRY RYZYKA I MODELI CZYTANE Z CONFIG.YAML (+ WARTOŚCI DOMYŚLNE)
    var_conf = float(cfg.get("var_confidence", 0.99))
    var_h = int(cfg.get("var_horizon_days", 20))
    use_log = bool(cfg.get("use_log_returns", True))
    risk_window_days = int(cfg.get("risk_window_days", 252))
    trading_days = int(cfg.get("trading_days", 252))
    w_max = float(cfg.get("w_max", 0.20))
    bl_tau = float(cfg.get("bl_tau", 0.05))
    bl_delta = float(cfg.get("bl_delta", 2.5))
    bl_omega_scale = float(cfg.get("bl_omega_scale", 1.0))
    bl_box_lb = float(cfg.get("bl_box_lb", 0.05))
    bl_box_ub = float(cfg.get("bl_box_ub", 0.12))
    raw_min_upside = cfg.get("min_upside", None)
    min_tickers_after_filter = int(cfg.get("min_tickers_after_filter", 9))
    r_f = float(cfg.get("risk_free_rate", 0.0))

    # DATY
    start_date = cfg.get("start_date") or (datetime.today().date() - timedelta(days=730))
    end_date = cfg.get("end_date") or datetime.today().date()

    # WCZYTYWANIE TRANSAKCJI I REKONSTRUKCJA PORTFELA
    try:
        trades_df, cash_balance = load_trades(trades_path)
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać transakcji z {trades_path}: {e}", file=sys.stderr)
        trades_df, cash_balance = pd.DataFrame(), 0.0

    if trades_df is not None and not trades_df.empty:
        holdings, _ = build_holdings(trades_df)
    else:
        holdings = pd.Series(dtype=float)

    # WYBIERAMY TICKERY
    tickers = choose_tickers(cfg, trades_df)
    if not tickers:
        print("[ERROR] Brak tickerów do pobrania cen.", file=sys.stderr)
        sys.exit(2)

    # FILTROWANIE SPÓŁEK PO UPSIDE
    tickers_filtered = filter_tickers_by_upside(valuation_path, tickers, raw_min_upside)

    if not tickers_filtered:
        print(f"[ERROR] Po filtrze min_upside={raw_min_upside} nie ma żadnych spółek.", file=sys.stderr)
        sys.exit(2)

    tickers = tickers_filtered
    if len(tickers) < min_tickers_after_filter:
        print(f"[ERROR] Za mało spółek po filtrze ({len(tickers)} < {min_tickers_after_filter}).", file=sys.stderr)
        sys.exit(2)

    # POBIERAMY CENY Z YAHOO FINANCE
    print(f"Pobieram ceny dla {len(tickers)} spółek od {start_date} do {end_date}.")
    prices = get_prices(tickers, start_date=start_date, end_date=end_date)
    if prices is None or prices.empty:
        print("[ERROR] Brak danych cenowych.", file=sys.stderr)
        sys.exit(3)

    # Synchronizujemy holdings z cenami
    prices = prices.reindex(columns=tickers)

    if holdings is None or holdings.empty:
        holdings = pd.Series(0.0, index=prices.columns, name="qty")
    else:
        holdings = holdings.reindex(prices.columns).fillna(0.0)
        holdings.name = "qty"

    # RYZYKO EMPIRYCZNE
    # Obliczenie empirycznych miar ryzyka portfela (VaR, ES, MDD, zmienność, NAV)
    # Funkcja compute_empirical_risk działa na:
    # - cenach spółek,
    # - aktualnym holdings,
    # i zwraca słownik z kluczowymi miarami ryzyka i wagami bieżącego portfela.
    risk_emp = compute_empirical_risk(
        prices=prices,
        holdings=holdings,
        horizon_days=var_h,
        trading_days=trading_days,
        risk_window_days=risk_window_days,
        use_log_returns=use_log,
        confidence=var_conf,
    )

    # WYZNACZAMY WAGI DO RISK PARITY
    # liczymy dzienne logarytmiczne zwroty spółek
    rets_log = returns(prices, log=True)
    # obliczamy macierz kowariancji
    Sigma = shrink_cov(rets_log)
    # rozwiązujemy problem optymalizacyjny - każdy składnik ma mieć podobny udział w ryzyku
    w_rp = risk_parity_weights(Sigma, w_min=0.0, w_max=w_max)

    # WYZNACZAMY WAGI DO BLACK–LITTERMAN
    # liczymy roczną macierz kowariancji (dzienna * liczba sesji)
    Sigma_ann = Sigma * trading_days
    # priory z RP, które przycinamy, żeby nie było wartości ujemnych
    w_mkt = np.maximum(w_rp, 0)
    # normalizujemy aby suma wag była równa 1
    w_mkt = w_mkt / w_mkt.sum()

    bl_weights = None
    bl_weights_box = None

    # wyznaczmy wagi
    if valuation_path and Path(valuation_path).exists():
        try:
            val = load_valuation_sheet(valuation_path)
            val = val[val["Ticker"].isin(prices.columns)].copy()
            required = {"Ticker", "Views", "Confidence"}
            if not val.empty and required.issubset(val.columns):
                idx_map = {t: i for i, t in enumerate(prices.columns)}
                pick_idx = [idx_map[t] for t in val["Ticker"]]
                k, n = len(pick_idx), len(prices.columns)
                # macierz P - każdy wiersz to r-ty pogląd na spółkę j-tą
                # gdy mamy poglądy tylko na jedną spółkę bez relacji między nimi
                # to każdy wiersz ma jedynkę, reszta zera
                P = np.zeros((k, n))
                for r, j in enumerate(pick_idx):
                    P[r, j] = 1.0

                # wektor Q (views - upside) skorygowany od stopę wolną od ryzyka
                Q = val["Views"].astype(float).to_numpy() - r_f

                # confidence - jak bardzo ufamy danemu poglądowi
                conf = val["Confidence"].astype(float).clip(1e-6, 1.0).to_numpy()
                # bazowa wariancja dla każdego poglądu (wynikająca wyłącznie z:
                # zmienności i korelacji spółek, niepewności priory i macierzy P
                # „rynkowa” niepewność view
                base = np.diag(P @ (bl_tau * Sigma_ann) @ P.T)
                base = np.clip(base, 1e-12, None)
                # wariancja poglądów po uwzględnieniu pewności
                # „rynkowa niepewność × (nasza niepewność)”
                Omega = np.diag(base / (conf ** 2)) * float(bl_omega_scale)

                # wywołujemy funkcję BL
                # zwraca wagi w_bl
                bl_out = bl_minimal(
                    Sigma=Sigma_ann,
                    w_mkt=w_mkt,
                    delta=bl_delta,
                    tau=bl_tau,
                    P=P,
                    Q=Q,
                    Omega=Omega,
                    omega_scale=1.0
                )
                w_bl_raw = pd.Series(bl_out["w_bl"], index=prices.columns)

                # wagi ujemne przycinamy do 0
                # normalizujemy tak, aby suma wah była równa 1
                bl_weights = w_bl_raw.clip(lower=0).fillna(0.0)
                if bl_weights.sum() > 0:
                    bl_weights = bl_weights / bl_weights.sum()

                # wersja bl_box: rzutujemy wagi w_bl na „ograniczony sympleks”
                # suma wag = 1, a każda waga w przedziale [bl_box_lb, bl_box_ub].
                w_box = project_boxed_simplex(v=w_bl_raw.to_numpy(), lb=bl_box_lb, ub=bl_box_ub, s=1.0)
                bl_weights_box = pd.Series(w_box, index=prices.columns)
        except Exception as e:
            # Jeżeli coś pójdzie nie tak (np. problemy z danymi), po prostu pomijamy BL
            # i ograniczamy się do wag RP i bieżących wag portfela.
            print(f"[WARN] Pominięto Black–Litterman: {e}", file=sys.stderr)

    # EKSPORT
    # Funkcja export_report_xlsx:
    # - tworzy arkusz Summary z miarami ryzyka (NAV, zmienność, VaR, ES, MDD, gotówka),
    # - tworzy arkusz Weights (Now, RP, BL, BL_Box),
    # - zapisuje holdings, Prices_Tail i Config.
    export_report_xlsx(
        output_path=output_path,
        cfg_path=str(cfg_path),
        start_date=str(start_date),
        end_date=str(end_date),
        risk_emp=risk_emp,
        cash_balance=float(cash_balance),
        var_conf=var_conf,
        var_h=var_h,
        holdings=holdings,
        prices=prices,
        w_rp=pd.Series(w_rp, index=prices.columns),
        bl_weights=bl_weights,
        bl_weights_box=bl_weights_box,
        use_log=use_log,
        risk_window_days=risk_window_days,
        trading_days=trading_days,
        w_max=w_max,
        bl_tau=bl_tau,
        bl_delta=bl_delta,
        bl_omega_scale=bl_omega_scale,
        bl_box_lb=bl_box_lb,
        bl_box_ub=bl_box_ub,
        n_tickers=len(prices.columns),
    )

    print(f"OK. Zapisano raport do: {output_path}")


if __name__ == "__main__":
    main()
