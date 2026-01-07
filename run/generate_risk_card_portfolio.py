from pathlib import Path
from datetime import datetime, timedelta
import yaml

from data.prices import get_prices
from data.portfolio_loader import load_trades, build_holdings
from reporting.word_risk_card import PortfolioRiskCardGenerator


# katalog główny projektu (poziom wyżej niż folder run/)
ROOT_DIR = Path(__file__).resolve().parents[1]
cfg_path = ROOT_DIR / "config.yaml"


def main():
    # --- config ---
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # --- daty (jak w run_portfolio_optimization) ---
    today = datetime.today().date()
    start_date = cfg.get("start_date") or (today - timedelta(days=730))
    end_date = cfg.get("end_date") or today

    # --- transakcje → holdings + cash ---
    trades_path = ROOT_DIR / cfg.get("trades_excel_path", "input/portfolio.xlsx")
    trades_df, cash_balance = load_trades(trades_path)
    holdings, _ = build_holdings(trades_df)

    tickers = holdings[holdings != 0].index.tolist()
    if not tickers:
        print("[WARN] Brak aktywnych pozycji w portfelu – karta ryzyka portfela nie została wygenerowana.")
        return

    # --- ceny ---
    prices = get_prices(
        tickers,
        start_date=start_date,
        end_date=end_date,
    )
    holdings = holdings.reindex(prices.columns).fillna(0.0)

    # --- ścieżka do katalogu na karty ryzyka (POZIOM WYŻEJ) ---
    cards_dir = ROOT_DIR / "output" / "risk_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    out_file = cards_dir / f"portfolio_risk_card_{today}.docx"

    # --- generowanie karty ---
    gen = PortfolioRiskCardGenerator(
        prices=prices,
        holdings=holdings,
        cash=cash_balance,
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
    )
    gen.generate(output_path=str(out_file))

    print(f"[OK] Portfolio Risk Card zapisano do: {out_file}")


if __name__ == "__main__":
    main()
