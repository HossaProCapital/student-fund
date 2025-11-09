import os
from typing import Optional
import pandas as pd

def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _to_sheet(writer, name, df, *, index):
    """Zapisz df do excela i automatycznie dopasuj szerokości kolumn."""
    df.to_excel(writer, sheet_name=name, index=index)
    ws = writer.sheets[name]

    # Pracujemy zawsze na danych z indeksem jako zwykłą kolumną 0..N
    printable = df.reset_index() if index else df.copy()
    headers = list(printable.columns)

    for col_idx, col in enumerate(headers):
        col_values = printable[col].astype(str).tolist()
        max_len = max([len(str(col))] + [len(v) for v in col_values]) if col_values else len(str(col))
        ws.set_column(col_idx, col_idx, min(100, max_len + 2))  # Mały margines i górny limit

def export_report_xlsx(
    *,
    output_path: str,
    cfg_path: str,
    start_date: str,
    end_date: str,
    # Summary
    risk_emp: dict,
    cash_balance: float,
    var_conf: float,
    var_h: int,
    # Arkusze szczegółowe
    holdings: pd.Series,
    prices: pd.DataFrame,
    w_rp: pd.Series,
    bl_weights: Optional[pd.Series] = None,
    bl_weights_box: Optional[pd.Series] = None,
    # Config
    use_log: bool = True,
    risk_window_days: int = 252,
    trading_days: int = 252,
    w_max: float = 0.20,
    bl_tau: float = 0.05,
    bl_delta: float = 2.5,
    bl_omega_scale: float = 1.0,
    bl_box_lb: float = 0.05,
    bl_box_ub: float = 0.12,
    n_tickers: int = 0,
):
    """Zapisuje wszystkie arkusze raportu do pliku excel."""
    _ensure_dir(output_path)

    # Podsumowanie
    summary = pd.DataFrame(
        {
            "Wartość portfela (NAV)": [risk_emp["nav"]],
            "Zmienność dzienna (σ)": [risk_emp["daily_vol"]],
            "Zmienność roczna (σ)": [risk_emp["annual_vol"]],
            f"VaR 1D @ {var_conf:.2%} (PLN)": [risk_emp["var_1d"]],
            f"ES  1D @ {var_conf:.2%} (PLN)": [risk_emp["es_1d"]],
            f"VaR √h, h={var_h} (PLN)": [risk_emp["var_h"]],
            f"ES  √h, h={var_h} (PLN)": [risk_emp["es_h"]],
            "Max Drawdown": [risk_emp["max_drawdown"]],
            "Gotówka (z arkusza transakcji)": [cash_balance],
        }
    ).T
    summary.columns = ["Wartość"]

    # Dane pomocnicze
    holdings_df = holdings.rename("Liczba akcji").to_frame()
    prices_tail = prices.tail(10)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Summary
        _to_sheet(writer, "Summary", summary, index=True)

        # Weights
        tickers = list(prices.columns)
        weights_df = pd.DataFrame(index=tickers)
        weights_df.index.name = "Ticker"

        weights_df["Now (%)"] = pd.Series(risk_emp.get("weights_map", {})).reindex(tickers).fillna(0.0) * 100.0
        weights_df["RP (%)"]  = pd.Series(w_rp).reindex(tickers).fillna(0.0) * 100.0

        if bl_weights is not None:
            weights_df["BL (%)"] = pd.Series(bl_weights).reindex(tickers).fillna(0.0) * 100.0
        if bl_weights_box is not None:
            weights_df["BL_Box (%)"] = pd.Series(bl_weights_box).reindex(tickers).fillna(0.0) * 100.0

        _to_sheet(writer, "Weights", weights_df.round(4), index=True)

        # Pozostałe arkusze
        _to_sheet(writer, "Holdings", holdings_df, index=True)
        _to_sheet(writer, "Prices_Tail", prices_tail, index=True)

        # Config
        config_df = pd.Series(
            {
                "config_path": os.path.abspath(cfg_path),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "var_confidence": var_conf,
                "var_horizon_days": var_h,
                "use_log_returns": use_log,
                "risk_window_days": risk_window_days,
                "trading_days": trading_days,
                "w_max": w_max,
                "bl_tau": bl_tau,
                "bl_delta": bl_delta,
                "bl_omega_scale": bl_omega_scale,
                "bl_box_lb": bl_box_lb,
                "bl_box_ub": bl_box_ub,
                "liczba_tickerów": n_tickers,
            }
        ).to_frame("config_value")
        _to_sheet(writer, "Config", config_df, index=True)
