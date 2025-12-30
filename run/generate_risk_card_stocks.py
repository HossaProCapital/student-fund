from pathlib import Path
from datetime import datetime, timedelta
import yaml

from data.prices import get_prices
from data.portfolio_loader import load_trades, build_holdings
from reporting.word_risk_card import StockRiskCardGenerator


ROOT_DIR = Path(__file__).resolve().parents[1]
cfg_path = ROOT_DIR / "config.yaml"


def main():
    # --- config ---
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # --- daty ---
    today = datetime.today().date()
    start_date = cfg.get("start_date") or (today - timedelta(days=730))
    end_date = cfg.get("end_date") or today

    # --- transakcje → holdings ---
    trades_path = ROOT_DIR / cfg.get("trades_excel_path", "input/portfolio.xlsx")
    trades_df, cash_balance = load_trades(trades_path)
    holdings, _ = build_holdings(trades_df)

    tickers = holdings[holdings > 0].index.tolist()
    if not tickers:
        print("[WARN] Brak aktywnych pozycji w portfelu – nie wygenerowano kart spółek.")
        return

    # --- ceny ---
    prices = get_prices(
        tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # katalog na karty spółek
    cards_dir = ROOT_DIR / "output" / "risk_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    gen = StockRiskCardGenerator(prices=prices, cfg=cfg)
    for t in tickers:
        out_file = cards_dir / f"stock_{t}_risk_card_{today}.docx"
        gen.generate_for_ticker(ticker=t, output_path=str(out_file))

    print(f"[OK] Wygenerowano karty ryzyka dla {len(tickers)} spółek w {cards_dir}")


if __name__ == "__main__":
    main()
