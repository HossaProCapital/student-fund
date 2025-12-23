import os
from typing import Optional
import pandas as pd
import numpy as np


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _to_sheet(writer, name, df, *, index):
    """Zapisz df do excela i automatycznie dopasuj szerokości kolumn."""
    df.to_excel(writer, sheet_name=name, index=index)
    ws = writer.sheets[name]

    printable = df.reset_index() if index else df.copy()
    headers = list(printable.columns)

    for col_idx, col in enumerate(headers):
        col_values = printable[col].astype(str).tolist()
        max_len = max([len(str(col))] + [len(v) for v in col_values]) if col_values else len(str(col))
        ws.set_column(col_idx, col_idx, min(100, max_len + 2))


def export_report_xlsx(
    *,
    output_path: str,
    cfg_path: str,
    start_date: str,
    end_date: str,
    risk_emp: dict,
    cash_balance: float,
    var_conf: float,
    var_h: int,
    holdings: pd.Series,
    prices: pd.DataFrame,
    w_rp: pd.Series,
    bl_weights: pd.Series | None = None,
    bl_weights_box: pd.Series | None = None,
    use_log: bool = True,
    risk_window_days: int = 252,
    trading_days: int = 252,
    w_max: float = 0.2,
    bl_tau: float = 0.05,
    bl_delta: float = 2.5,
    bl_omega_scale: float = 1.0,
    bl_box_lb: float = 0.05,
    bl_box_ub: float = 0.12,
    n_tickers: int = 0,
    sector_map: dict | None = None,
    max_sector_weight: float | None = None,
    Sigma: pd.DataFrame | np.ndarray | None = None,   # <<< NOWY ARGUMENT
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
            "Sharpe": [risk_emp.get("sharpe")],
            "Tracking Error": [risk_emp.get("tracking_error")],
            "Information Ratio": [risk_emp.get("information_ratio")],
            "Beta": [risk_emp.get("beta")],
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

        # -----------------------------------------------------
        # Risk Contribution – udział w ryzyku dla różnych portfeli
        # -----------------------------------------------------
        try:
            # używamy tej samej macierzy kowariancji, co w Risk Parity,
            # jeśli została podana; w przeciwnym razie liczymy od nowa
            if Sigma is not None:
                if isinstance(Sigma, pd.DataFrame):
                    cov = Sigma.copy()
                else:
                    cov = pd.DataFrame(Sigma, index=prices.columns, columns=prices.columns)
            else:
                rets = prices.pct_change().dropna()
                if risk_window_days is not None and risk_window_days > 0:
                    rets = rets.tail(risk_window_days)
                cov = rets.cov()


            def _risk_contrib(weights_series: pd.Series | None):
                """Zwraca RC% (udział w ryzyku) jako Series, lub None jeśli nie da się policzyć."""
                if weights_series is None or cov is None or cov.empty:
                    return None

                w = weights_series.reindex(cov.index).fillna(0.0).to_numpy()
                Sigma = cov.to_numpy()

                marginal = Sigma @ w
                total_var = float(w.T @ marginal)
                if total_var <= 0:
                    return pd.Series(0.0, index=cov.index)

                rc = w * marginal
                rc_pct = rc / total_var   # suma ≈ 1.0
                return pd.Series(rc_pct, index=cov.index)

            # Wagi "Now" (z risk_emp["weights_map"])
            w_now = pd.Series(risk_emp.get("weights_map", {})).reindex(tickers).fillna(0.0)
            w_rp_full = pd.Series(w_rp).reindex(tickers).fillna(0.0)
            w_bl_full = pd.Series(bl_weights).reindex(tickers).fillna(0.0) if bl_weights is not None else None
            w_bl_box_full = pd.Series(bl_weights_box).reindex(tickers).fillna(0.0) if bl_weights_box is not None else None

            rc_now = _risk_contrib(w_now)
            rc_rp = _risk_contrib(w_rp_full)
            rc_bl = _risk_contrib(w_bl_full) if w_bl_full is not None else None
            rc_bl_box = _risk_contrib(w_bl_box_full) if w_bl_box_full is not None else None

            rc_df = pd.DataFrame(index=tickers)
            rc_df.index.name = "Ticker"

            if rc_now is not None:
                rc_df["Now RC (%)"] = rc_now.reindex(tickers).fillna(0.0) * 100.0
            if rc_rp is not None:
                rc_df["RP RC (%)"] = rc_rp.reindex(tickers).fillna(0.0) * 100.0
            if rc_bl is not None:
                rc_df["BL RC (%)"] = rc_bl.reindex(tickers).fillna(0.0) * 100.0
            if rc_bl_box is not None:
                rc_df["BL_Box RC (%)"] = rc_bl_box.reindex(tickers).fillna(0.0) * 100.0

            # zapisujemy tylko jeśli coś faktycznie policzyliśmy
            if not rc_df.empty and rc_df.shape[1] > 0:
                _to_sheet(writer, "Risk_Contribution", rc_df.round(4), index=True)

        except Exception as e:
            print(f"[WARN] Nie udało się policzyć Risk Contribution: {e}")




        # --- NOWE: ekspozycja sektorowa ---
        # Robimy to na bazie weights_df (w %), sumując wagi po sektorach.
        if sector_map is not None and isinstance(sector_map, dict) and len(sector_map) > 0:
            sector_s = pd.Series(sector_map, name="Sector").reindex(tickers)

            # Jeśli coś nie ma sektora, pokażemy jako UNKNOWN (nie wywalajmy raportu na etapie exportu)
            sector_s = sector_s.fillna("UNKNOWN").astype(str)

            tmp = weights_df.copy()
            tmp["Sector"] = sector_s.values

            # Groupby sektor -> suma wag w sektorze
            sector_exp = tmp.groupby("Sector")[weights_df.columns].sum()

            # Limit (jeśli podany)
            if max_sector_weight is not None:
                limit_pct = float(max_sector_weight) * 100.0
                sector_exp["Limit (%)"] = limit_pct

                # Status per portfel (OK/BREACH) dla każdej kolumny wagowej
                for col in ["Now (%)", "RP (%)", "BL (%)", "BL_Box (%)"]:
                    if col in sector_exp.columns:
                        sector_exp[f"{col} Status"] = sector_exp[col].apply(
                            lambda x: "BREACH" if x > limit_pct + 1e-9 else "OK"
                        )

            # Ładny porządek: największe sektory wg RP (jeśli jest), inaczej wg Now
            sort_col = "RP (%)" if "RP (%)" in sector_exp.columns else "Now (%)"
            sector_exp = sector_exp.sort_values(sort_col, ascending=False)

            _to_sheet(writer, "Sector_Exposure", sector_exp.round(4), index=True)

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
                # dopiszemy też info o sektorach
                "sector_constraints_enabled": bool(sector_map is not None),
                "max_weight_per_sector": max_sector_weight,
            }
        ).to_frame("config_value")
        _to_sheet(writer, "Config", config_df, index=True)
