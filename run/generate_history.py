"""
Generate fund history
---------------------

Ten skrypt:

1) wczytuje konfigurację,
2) wczytuje transakcje i wyznacza historię holdings,
3) pobiera ceny dla wszystkich tickerów,
4) dla każdej daty liczy metryki ryzyka portfela (jak w run_portfolio_optimization),
5) zapisuje dzienną historię do CSV: history/fund_history.csv

Na razie w wersji prostej:
- gotówka (cash) jest stała w czasie = dzisiejszy cash_balance z load_trades,
- benchmark = None (bez TE / IR / bety w historii).
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import yaml
import numpy as np
import pandas as pd

from data.portfolio_loader import load_trades, build_holdings
from data.prices import get_prices
from data.valuation_loader import load_tickers_from_valuation

from analytics.risk_metrics import compute_empirical_risk
from analytics.risk_extra import compute_basic_risk_snapshot_main


ROOT_DIR = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def choose_tickers_for_history(cfg: dict, trades_df: pd.DataFrame) -> list[str]:
    """
    Bardzo podobne do choose_tickers z run_portfolio_optimization:
    najpierw bierzemy tickery z arkusza wycen, a jeśli się nie da – z transakcji.
    """
    val_rel = cfg.get("valuation_excel_path")
    if val_rel:
        val_path = ROOT_DIR / val_rel
        if val_path.exists():
            try:
                return load_tickers_from_valuation(val_path)
            except Exception:
                pass

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


def _compute_var95_1d_pln(
    prices_hist: pd.DataFrame,
    holdings_vec: pd.Series,
    cash: float,
    risk_window_days: int,
    trading_days: int,
) -> float:
    """
    Dodatkowe wyliczenie VaR95 1D w PLN (historyczny, empiryczny).
    Zrobione analogicznie do compute_basic_risk_snapshot_main.
    """
    rets = prices_hist.pct_change().dropna()
    if len(rets) == 0:
        return np.nan

    window = rets.tail(risk_window_days)

    last = prices_hist.ffill().iloc[-1]
    equity_value = float((holdings_vec * last).sum())
    if equity_value <= 0:
        return np.nan

    weights = (holdings_vec * last) / equity_value
    port_ret = window.mul(weights, axis=1).sum(axis=1)

    alpha95 = 0.05  # 1 - 0.95
    var95_ret = float(port_ret.quantile(alpha95))
    nav = equity_value + float(cash)

    return -var95_ret * nav  # w PLN


def main():
    cfg_path = ROOT_DIR / "config.yaml"
    if not cfg_path.is_file():
        print("[ERROR] Brak pliku config.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"Używam konfiguracji: {cfg_path}")
    cfg = read_config(cfg_path)

    valuation_path = ROOT_DIR / cfg.get("valuation_excel_path", "input/portfolio2.xlsx")
    trades_path = ROOT_DIR / cfg.get("trades_excel_path", "input/portfolio.xlsx")

    # Ścieżka wyjściowa CSV (na razie stała lub z configu, jeśli dodasz)
    history_rel = "output/fund_history.xlsx"
    history_path = ROOT_DIR / history_rel
    os.makedirs(history_path.parent, exist_ok=True)

    # Parametry ryzyka
    var_conf = float(cfg.get("var_confidence", 0.99))
    var_h = int(cfg.get("var_horizon_days", 20))
    use_log = bool(cfg.get("use_log_returns", True))
    risk_window_days = int(cfg.get("risk_window_days", 252))
    trading_days = int(cfg.get("trading_days", 252))
    r_f = float(cfg.get("risk_free_rate", 0.0))

    # --- Transakcje ---
    try:
        trades_df, cash_balance = load_trades(trades_path)
    except Exception as e:
        print(f"[ERROR] Nie udało się wczytać transakcji: {e}", file=sys.stderr)
        sys.exit(2)

    if trades_df.empty:
        print("[ERROR] Brak transakcji w pliku – nie można zbudować historii.", file=sys.stderr)
        sys.exit(2)

    # końcowe holdings + historia holdings po każdej transakcji
    holdings_final, holdings_hist = build_holdings(trades_df)

    # --- Tickery ---
    tickers = choose_tickers_for_history(cfg, trades_df)
    if not tickers:
        print("[ERROR] Brak tickerów do pobrania cen.", file=sys.stderr)
        sys.exit(3)

    # --- Ceny ---
    # zakres: jak w configu, a jeśli start_date = None → ostatnie 2 lata
    start_date = cfg.get("start_date") or (datetime.today().date() - timedelta(days=730))
    end_date = cfg.get("end_date") or datetime.today().date()

    print(f"Pobieram ceny dla historii: {len(tickers)} spółek, {start_date} – {end_date}")
    prices = get_prices(tickers, start_date=start_date, end_date=end_date)
    if prices is None or prices.empty:
        print("[ERROR] Brak danych cenowych z Yahoo Finance.", file=sys.stderr)
        sys.exit(3)

    prices = prices.reindex(columns=tickers)

    # --- Historia holdings na wszystkie dni z cenami ---
    # holdings_hist ma indeks = daty transakcji; może zawierać kilka wierszy na ten sam dzień.
    # Najpierw bierzemy ostatni stan z każdego dnia, żeby indeks był unikalny.
    holdings_hist = holdings_hist.sort_index()
    holdings_hist = holdings_hist[~holdings_hist.index.duplicated(keep="last")]

    # Teraz możemy bezpiecznie reindexować do kalendarza cen
    holdings_hist_full = (
        holdings_hist
        .reindex(prices.index)    # dopasowujemy do wszystkich dni, dla których mamy ceny
        .ffill()                  # przenosimy ostatni znany stan
        .fillna(0.0)
    )


    # Jeżeli w holdings_history jest mniej kolumn niż w prices (np. ticker w wycenach,
    # którego jeszcze nie kupiliśmy), dopełniamy zerami.
    for t in prices.columns:
        if t not in holdings_hist_full.columns:
            holdings_hist_full[t] = 0.0

    holdings_hist_full = holdings_hist_full[prices.columns]

    # Zaczynamy historię od pierwszego dnia, kiedy portfel MA jakiekolwiek akcje
    nonzero_mask = holdings_hist_full.ne(0).any(axis=1)
    if not nonzero_mask.any():
        print("[ERROR] Portfel nie ma żadnych akcji w historii (same zera).", file=sys.stderr)
        sys.exit(4)

    first_date = nonzero_mask.idxmax()  # pierwszy True
    prices = prices.loc[first_date:]
    holdings_hist_full = holdings_hist_full.loc[first_date:]

    rows = []

    for as_of_date in prices.index:
        # ceny do tej daty (dla risk_window_days itp.)
        prices_hist = prices.loc[:as_of_date]

        # holdings na ten dzień
        holdings_today = holdings_hist_full.loc[as_of_date].reindex(prices.columns).fillna(0.0)

        # --- ryzyko jak w głównym skrypcie: empiryczne + extra, scalone ---
        try:
            risk_emp = compute_empirical_risk(
                prices=prices_hist,
                holdings=holdings_today,
                horizon_days=var_h,
                trading_days=trading_days,
                risk_window_days=risk_window_days,
                use_log_returns=use_log,
                confidence=var_conf,
            )

            risk_extra = compute_basic_risk_snapshot_main(
                prices=prices_hist,
                holdings=holdings_today,
                cash=cash_balance,
                risk_window_days=risk_window_days,
                var_conf=var_conf,
                var_h=var_h,
                trading_days=trading_days,
                r_f=r_f,
                benchmark=None,
            )

            risk_emp.update(risk_extra)

        except Exception as e:
            # np. za mało danych – pomijamy taki dzień
            print(f"[WARN] Pomijam {as_of_date.date()}: nie udało się policzyć ryzyka ({e})")
            continue

        nav = float(risk_emp.get("nav", np.nan))
        cash = float(cash_balance)
        equity_value = nav - cash

        # VaR95 1D w PLN
        var95_1d_pln = _compute_var95_1d_pln(
            prices_hist=prices_hist,
            holdings_vec=holdings_today,
            cash=cash_balance,
            risk_window_days=risk_window_days,
            trading_days=trading_days,
        )

        rows.append(
            {
                "date": as_of_date.date().isoformat(),
                "NAV": nav,
                "cash": cash,
                "equity_value": equity_value,
                "daily_vol": risk_emp.get("daily_vol"),
                "annual_vol": risk_emp.get("annual_vol"),
                "VaR95_1D": var95_1d_pln,
                "VaR99_1D": risk_emp.get("var_1d"),
                "VaR99_20D": risk_emp.get("var_h"),
                "ES99_1D": risk_emp.get("es_1d"),
                "MDD": risk_emp.get("max_drawdown"),
                "Sharpe": risk_emp.get("sharpe"),
                "TrackingError": risk_emp.get("tracking_error"),
                "InformationRatio": risk_emp.get("information_ratio"),
                "Beta": risk_emp.get("beta"),
            }
        )

    if not rows:
        print("[ERROR] Nie udało się zbudować żadnego wiersza historii.", file=sys.stderr)
        sys.exit(5)

    history_df = pd.DataFrame(rows)
    history_df.set_index("date", inplace=False)

    # Liczymy dzienny zwrot z NAV (na końcu, żeby nie mieszać w pętli)
    history_df["daily_ret"] = history_df["NAV"].pct_change()

    # Przestawiamy kolejność kolumn tak, jak planowałaś
    cols_order = [
        "date",
        "NAV",
        "cash",
        "equity_value",
        "daily_ret",
        "daily_vol",
        "annual_vol",
        "VaR95_1D",
        "VaR99_1D",
        "VaR99_20D",
        "ES99_1D",
        "MDD",
        "Sharpe",
        "TrackingError",
        "InformationRatio",
        "Beta",
    ]
    # dodajemy tylko te kolumny, które faktycznie są w df
    cols_order = [c for c in cols_order if c in history_df.columns]
    history_df = history_df[cols_order]

    history_df.to_excel(history_path, index=False)
    print(f"[OK] Historia funduszu zapisana do: {history_path}")


if __name__ == "__main__":
    main()
