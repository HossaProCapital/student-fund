"""
Skrypt do generowania kart ryzyka dla pojedynczych spółek.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import pandas as pd

from analytics.risk_metrics_by_company import create_risk_cards_batch
from analytics.risk_utils import returns
from data.prices import get_prices
from data.valuation_loader import load_valuation_sheet
from optimization.risk_parity import shrink_cov, risk_parity_weights


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    cfg_path = Path("config.yaml")
    if not cfg_path.is_file():
        print("[ERROR] Nie znaleziono config.yaml", file=sys.stderr)
        sys.exit(1)

    cfg = read_config(cfg_path)

    valuation_path = cfg.get("valuation_excel_path", "input/portfolio2.xlsx")
    w_max = float(cfg.get("w_max", 0.20))
    min_upside = cfg.get("min_upside", 0.2)
    start_date = cfg.get("start_date") or (datetime.today().date() - timedelta(days=730))
    end_date = cfg.get("end_date") or datetime.today().date()

    # Wczytaj tickery
    val_df = load_valuation_sheet(valuation_path)
    val_df["Views"] = pd.to_numeric(val_df["Views"], errors="coerce")
    val_df = val_df[val_df["Views"] >= min_upside].copy()
    tickers = val_df["Ticker"].dropna().astype(str).str.upper().unique().tolist()

    # Pobierz ceny
    prices = get_prices(tickers, start_date=start_date, end_date=end_date)

    # Oblicz Risk Parity
    rets = returns(prices, log=True)
    Sigma = shrink_cov(rets)
    w_rp = risk_parity_weights(Sigma, w_min=0.0, w_max=w_max)

    risk_parity_dict = w_rp.to_dict()
    portfolio_prices = (prices * w_rp.values).sum(axis=1)
    valuation_dict = dict(zip(val_df["Ticker"], val_df["Views"]))

    # Generuj karty ryzyka
    risk_cards_df = create_risk_cards_batch(
        tickers=list(prices.columns),
        prices_df=prices,
        cfg=cfg,
        portfolio_prices=portfolio_prices,
        valuation_dict=valuation_dict,
        risk_parity_dict=risk_parity_dict
    )

    # Zapisz
    output_path = Path("output/risk_cards.xlsx")
    output_path.parent.mkdir(exist_ok=True)
    risk_cards_df.to_excel(output_path, index=False, sheet_name="Risk Cards")

    print(f"Zapisano: {output_path}")


if __name__ == "__main__":
    main()