import sys
from pathlib import Path
import yaml
import pandas as pd

from data.portfolio_loader import load_trades
from data.valuation_loader import load_valuation_sheet, load_tickers_from_valuation

ROOT_DIR = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def choose_tickers(cfg: dict, trades_df: pd.DataFrame) -> list[str]:
    # 1) najpierw próbujemy wczytać tickery z arkusza wycen (portfolio2.xlsx)
    val_rel = cfg.get("valuation_excel_path")
    if val_rel:
        val_path = ROOT_DIR / val_rel
        if val_path.exists():
            try:
                return load_tickers_from_valuation(val_path)
            except Exception:
                pass

    # 2) jeśli nie ma wycen – bierzemy tickery z transakcji
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


def main():
    # -------------------------
    # Config
    # -------------------------
    cfg_path = ROOT_DIR / "config.yaml"
    if not cfg_path.is_file():
        print("[ERROR] Brak pliku config.yaml", file=sys.stderr)
        sys.exit(1)

    cfg = read_config(cfg_path)

    valuation_path = ROOT_DIR / cfg.get("valuation_excel_path", "input/portfolio2.xlsx")
    trades_path = ROOT_DIR / cfg.get("trades_excel_path", "input/portfolio.xlsx")

    raw_min_upside = cfg.get("min_upside")

    # -------------------------
    # Transakcje (żeby mieć tickery bazowe)
    # -------------------------
    try:
        trades_df, _cash = load_trades(trades_path)
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać transakcji: {e}")
        trades_df = pd.DataFrame()

    tickers_base = choose_tickers(cfg, trades_df)
    if not tickers_base:
        print("[ERROR] Brak tickerów bazowych.", file=sys.stderr)
        sys.exit(2)

    # -------------------------
    # Wyceny
    # -------------------------
    if not valuation_path.exists():
        print(f"[ERROR] Brak pliku wycen: {valuation_path}", file=sys.stderr)
        sys.exit(3)

    try:
        val = load_valuation_sheet(valuation_path)
    except Exception as e:
        print(f"[ERROR] Nie udało się wczytać wycen: {e}", file=sys.stderr)
        sys.exit(3)

    if "Ticker" not in val.columns:
        print("[ERROR] Brak kolumny 'Ticker' w arkuszu wycen.", file=sys.stderr)
        sys.exit(3)

    val["Ticker"] = val["Ticker"].astype(str).str.upper()

    # -------------------------
    # Filtr upside
    # -------------------------
    if raw_min_upside is None:
        filtered_df = pd.DataFrame({"Ticker": tickers_base})
        filtered_tickers = tickers_base
        print("[INFO] min_upside nie ustawione – używam tickerów bazowych.")
    else:
        if "Views" not in val.columns:
            print("[ERROR] Ustawiono min_upside, ale brak kolumny 'Views' w wycenach.", file=sys.stderr)
            sys.exit(4)

        min_upside = float(raw_min_upside)

        sub = val[val["Ticker"].isin(tickers_base)].copy()
        sub["Views"] = pd.to_numeric(sub["Views"], errors="coerce")

        filtered_df = sub.loc[sub["Views"] >= min_upside, ["Ticker", "Views"]].dropna()

        if filtered_df.empty:
            print("[ERROR] Po filtrze upside nie zostały żadne tickery.", file=sys.stderr)
            sys.exit(4)

        filtered_df = filtered_df.sort_values("Views", ascending=False)
        filtered_tickers = filtered_df["Ticker"].tolist()

        print(f"\nTickery po filtrze upside (Views >= {min_upside}):")
        for _, row in filtered_df.iterrows():
            print(f"{row['Ticker']}\tViews={row['Views']}")

    # -------------------------
    # n oraz P = (n/15)*70%
    # -------------------------
    n = len(filtered_tickers)
    P = (n / 15.0) * 0.7  # część portfela (0..1)

    # -------------------------
    # OGRANICZENIA: 5% i 9% liczone z aktywnej części 70%
    # czyli globalnie w całym portfelu to: 0.70*0.05 i 0.70*0.09
    # -------------------------
    active_part = 0.70

    active_lb = 0.05  # 5% z aktywnej części
    active_ub = 0.09  # 9% z aktywnej części

    # przeliczenie na udział w CAŁYM portfelu
    global_lb = active_part * active_lb  # 0.035
    global_ub = active_part * active_ub  # 0.063

    if P <= 0:
        print("[ERROR] P <= 0 (brak części portfela do przeliczeń).", file=sys.stderr)
        sys.exit(5)

    # ograniczenia wewnątrz puli P
    lb_sub = global_lb / P
    ub_sub = global_ub / P

    # -------------------------
    # Podsumowanie + checki
    # -------------------------
    print("\n--- PODSUMOWANIE ---")
    print(f"Liczba tickerów (n): {n}")
    print(f"Część portfela P = (n/15)*70%: {P:.6f}  ({P*100:.2f}%)")

    print("\nOgraniczenia zdefiniowane jako % aktywnej części (70%):")
    print(f"  LB_active: {active_lb:.4f} ({active_lb*100:.2f}% aktywnej części)")
    print(f"  UB_active: {active_ub:.4f} ({active_ub*100:.2f}% aktywnej części)")

    print("\nTe same ograniczenia w ujęciu CAŁEGO portfela:")
    print(f"  LB_global = 0.70*0.05: {global_lb:.4f} ({global_lb*100:.2f}%)")
    print(f"  UB_global = 0.70*0.09: {global_ub:.4f} ({global_ub*100:.2f}%)")

    print("\nOgraniczenia w obrębie puli P:")
    print(f"  LB_sub = LB_global / P: {lb_sub:.6f} ({lb_sub*100:.2f}%)")
    print(f"  UB_sub = UB_global / P: {ub_sub:.6f} ({ub_sub*100:.2f}%)")

    # wykonalność: n * LB_sub <= 1
    if n * lb_sub > 1.0:
        print("\n[ERROR] Constrainty niewykonalne w obrębie puli P:")
        print(f"  n * LB_sub = {n*lb_sub:.4f} > 1.0 (suma minimów > 100% puli)")
    else:
        print("\n[OK] Dolne ograniczenie jest wykonalne w obrębie puli P (n * LB_sub <= 1).")

    if ub_sub < lb_sub:
        print("[ERROR] UB_sub < LB_sub – ograniczenia sprzeczne.")

    if ub_sub > 1.0:
        print("[WARN] UB_sub > 100% puli P – górne ograniczenie w obrębie puli praktycznie nie ogranicza.")


if __name__ == "__main__":
    main()
