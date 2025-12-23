"""
Run Portfolio Optimization
--------------------------

Ten skrypt wykonuje TYLKO:

1) wczytanie konfiguracji
2) pobranie cen
3) budowę holdings
4) filtr upside
5) walidację sektorów
6) Risk Parity + BL + BL_Box
7) wygenerowanie raportu Excel z PODSTAWOWYMI wskaźnikami ryzyka portfela

Karty ryzyka / historia będą w osobnych plikach:
- generate_risk_card_portfolio.py
- generate_history.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd

from analytics.risk_metrics import compute_empirical_risk
from analytics.risk_utils import returns
from analytics.risk_extra import compute_basic_risk_snapshot_main

from data.portfolio_loader import load_trades, build_holdings
from data.prices import get_prices
from data.valuation_loader import load_valuation_sheet, load_tickers_from_valuation

from optimization.risk_parity import shrink_cov, risk_parity_weights
from optimization.black_litterman import bl_minimal
from optimization.constraints import project_boxed_simplex

from reporting.exporter import export_report_xlsx

ROOT_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Ładowanie configu
# ---------------------------------------------------------

def read_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------
# Wybór tickerów
# ---------------------------------------------------------

def choose_tickers(cfg: dict, trades_df: pd.DataFrame) -> list[str]:
    # najpierw próbujemy wczytać tickery z arkusza wycen (portfolio2.xlsx)
    val_rel = cfg.get("valuation_excel_path")
    if val_rel:
        val_path = ROOT_DIR / val_rel
        if val_path.exists():
            try:
                return load_tickers_from_valuation(val_path)
            except Exception:
                pass

    # jeśli nie ma wycen – bierzemy tickery z transakcji
    if not trades_df.empty and "Ticker" in trades_df.columns:
        return (
            trades_df["Ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

    return []



# ---------------------------------------------------------
# Główna logika optymalizacji portfela
# ---------------------------------------------------------

def run():

    cfg_path = ROOT_DIR / "config.yaml"
    if not cfg_path.is_file():
        print("[ERROR] Brak pliku config.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"Używam konfiguracji: {cfg_path}")
    cfg = read_config(cfg_path)

    valuation_path = ROOT_DIR / cfg.get("valuation_excel_path", "input/portfolio2.xlsx")
    trades_path = ROOT_DIR / cfg.get("trades_excel_path", "input/portfolio.xlsx")
    output_path = ROOT_DIR / cfg.get("output_file", "output/portfolio_risk_report.xlsx")

    # Parametry ryzyka i modeli
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

    raw_min_upside = cfg.get("min_upside")
    min_tickers_after_filter = int(cfg.get("min_tickers_after_filter", 9))
    r_f = float(cfg.get("risk_free_rate", 0.0))

    # --- sektor constraints ---
    sec_cfg = cfg.get("sector_constraints", {}) or {}
    sector_enabled = bool(sec_cfg.get("enabled", False))
    max_sector_weight = float(sec_cfg.get("max_weight_per_sector")) if sec_cfg.get("max_weight_per_sector") else None

    # Daty
    start_date = cfg.get("start_date") or (datetime.today().date() - timedelta(days=730))
    end_date = cfg.get("end_date") or datetime.today().date()

    # ------------------------------
    # Transakcje → Holdings
    # ------------------------------
    try:
        trades_df, cash_balance = load_trades(trades_path)
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać transakcji: {e}")
        trades_df, cash_balance = pd.DataFrame(), 0.0

    if not trades_df.empty:
        holdings, _ = build_holdings(trades_df)
    else:
        holdings = pd.Series(dtype=float)

    # Tickery
    tickers = choose_tickers(cfg, trades_df)
    if not tickers:
        print("[ERROR] Brak tickerów do pobrania cen.", file=sys.stderr)
        sys.exit(2)

    # Wyceny (RAZ)
    val = pd.DataFrame()
    if valuation_path and valuation_path.exists():
        try:
            val = load_valuation_sheet(valuation_path)
        except Exception as e:
            print(f"[WARN] Nie udało się wczytać wycen: {e}")

    # Upside filter
    if raw_min_upside is not None:
        if val.empty:
            print("[ERROR] Ustawiono min_upside, ale brak wycen.", file=sys.stderr)
            sys.exit(2)

        sub = val[val["Ticker"].isin(tickers)].copy()
        tickers = sub.loc[sub["Views"] >= float(raw_min_upside), "Ticker"].dropna().tolist()

        if not tickers:
            print("[ERROR] Po filtrze upside nie zostały żadne tickery.", file=sys.stderr)
            sys.exit(2)

    if len(tickers) < min_tickers_after_filter:
        print(f"[ERROR] Za mało spółek po filtrze: {len(tickers)} < {min_tickers_after_filter}", file=sys.stderr)
        sys.exit(2)

    # --- WALIDACJA SEKTORÓW ---
    sector_map = None
    if sector_enabled:
        if val.empty:
            print("[ERROR] Sector constraints włączone, a brak wycen.", file=sys.stderr)
            sys.exit(2)

        val_sub = val[val["Ticker"].isin(tickers)].copy()
        missing = val_sub[val_sub["Sector"].isna()]["Ticker"].tolist()

        if missing:
            print(f"[ERROR] Brak sektora w portfolio2.xlsx dla tickerów: {missing}", file=sys.stderr)
            sys.exit(2)

        if max_sector_weight is None:
            print("[ERROR] sectors.enabled=True, ale brak max_sector_weight.", file=sys.stderr)
            sys.exit(2)

        sector_map = val_sub.set_index("Ticker")["Sector"].astype(str).to_dict()

    # ------------------------------
    # Pobranie cen
    # ------------------------------
    print(f"Pobieram ceny dla {len(tickers)} spółek...")
    prices = get_prices(tickers, start_date=start_date, end_date=end_date)

    if prices is None or prices.empty:
        print("[ERROR] Brak danych cenowych.", file=sys.stderr)
        sys.exit(3)

    prices = prices.reindex(columns=tickers)

    # Synchronizacja holdings
    if holdings.empty:
        holdings = pd.Series(0.0, index=prices.columns)
    else:
        holdings = holdings.reindex(prices.columns).fillna(0.0)

    # ------------------------------
    # Empiryczne metryki ryzyka (dziś)
    # ------------------------------
    risk_emp = compute_empirical_risk(
        prices=prices,
        holdings=holdings,
        horizon_days=var_h,
        trading_days=trading_days,
        risk_window_days=risk_window_days,
        use_log_returns=use_log,
        confidence=var_conf,
    )

    # ------------------------------
    # DODAJEMY PODSTAWOWE DODATKOWE WSKAŹNIKI RYZYKA
    # ------------------------------
    risk_extra = compute_basic_risk_snapshot_main(
        prices=prices,
        holdings=holdings,
        cash=cash_balance,
        risk_window_days=risk_window_days,
        var_conf=var_conf,
        var_h=var_h,
        trading_days=trading_days,
        r_f=r_f,
        benchmark=None   # brak benchmarku na ten moment
    )

    # scal w jeden słownik
    risk_emp.update(risk_extra)

    # ------------------------------
    # Risk Parity
    # ------------------------------
    rets_log = returns(prices, log=True)
    Sigma = shrink_cov(rets_log)

    try:
        w_rp = risk_parity_weights(
            Sigma,
            w_min=0.0,
            w_max=w_max,
            sector_map=sector_map,
            max_sector_weight=max_sector_weight if sector_enabled else None,
        )
    except Exception as e:
        print(f"[ERROR] Risk Parity failure: {e}", file=sys.stderr)
        sys.exit(4)

    # ------------------------------
    # Black–Litterman
    # ------------------------------
    Sigma_ann = Sigma * trading_days
    w_mkt = np.maximum(w_rp, 0)
    w_mkt = w_mkt / w_mkt.sum()

    bl_weights = None
    bl_weights_box = None

    if not val.empty:
        try:
            val_bl = val[val["Ticker"].isin(prices.columns)].copy()
            required = {"Ticker", "Views", "Confidence"}

            if required.issubset(val_bl.columns):
                idx_map = {t: i for i, t in enumerate(prices.columns)}
                pick_idx = [idx_map[t] for t in val_bl["Ticker"]]

                k = len(pick_idx)
                n = len(prices.columns)

                P = np.zeros((k, n))
                for r, j in enumerate(pick_idx):
                    P[r, j] = 1.0

                Q = val_bl["Views"].astype(float).to_numpy() - r_f
                conf = val_bl["Confidence"].astype(float).clip(1e-6, 1.0).to_numpy()

                base = np.diag(P @ (bl_tau * Sigma_ann) @ P.T)
                base = np.clip(base, 1e-12, None)
                Omega = np.diag(base / (conf ** 2)) * float(bl_omega_scale)

                bl_out = bl_minimal(
                    Sigma=Sigma_ann,
                    w_mkt=w_mkt,
                    delta=bl_delta,
                    tau=bl_tau,
                    P=P,
                    Q=Q,
                    Omega=Omega,
                    omega_scale=1.0,
                )

                w_bl_raw = pd.Series(bl_out["w_bl"], index=prices.columns)

                bl_weights = w_bl_raw.clip(lower=0).fillna(0.0)
                if bl_weights.sum() > 0:
                    bl_weights = bl_weights / bl_weights.sum()

                w_box = project_boxed_simplex(
                    v=w_bl_raw.to_numpy(),
                    lb=bl_box_lb,
                    ub=bl_box_ub,
                    s=1.0,
                )
                bl_weights_box = pd.Series(w_box, index=prices.columns)

        except Exception as e:
            print(f"[WARN] BL error: {e}", file=sys.stderr)

    # ------------------------------
    # Export Excel
    # ------------------------------
    export_report_xlsx(
        output_path=str(output_path),
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
        sector_map=sector_map,
        max_sector_weight=max_sector_weight if sector_enabled else None,
        Sigma=Sigma,
    )

    print(f"[OK] Raport wygenerowano: {output_path}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run()
