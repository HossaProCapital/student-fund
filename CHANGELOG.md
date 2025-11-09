# CHANGELOG

All notable changes to **fundusz-hossa-procapital** are documented here.

---

## [Unreleased]
###Added - 09.11.2025 (@PawelSkrzypczak01)
- Dodano `analytics/risk_metrics_by_company.py` — funkcje do tworzenia kart ryzyka:
  - `calculate_volatility`, `calculate_downside_risk`, `calculate_sharpe_ratio`,
    `calculate_sortino_ratio`, `calculate_max_drawdown`, `calculate_beta_vs_portfolio`,
    `calculate_correlation_with_portfolio`, `create_risk_card`, `create_risk_cards_batch`.
  - bezpieczne obsługi edge-case'ów (Series/DataFrame, brak danych, usunięto bare-except).
- Dodano skrypt uruchomieniowy generujący karty ryzyka (plik main) — pobranie cen, risk parity, eksport `output/risk_cards.xlsx`.

### Planned / proposals
- Additional sheets (e.g., Allocation, Monitoring) and risk-limit alerts.

---

## [1.0.0] – 2025-11-09
### Added
- **analytics/risk_metrics.py**
  - `compute_empirical_risk(...)`: historical (non-parametric) portfolio risk:
    - daily and annual volatility (from log-returns),
    - historical VaR/ES on simple returns, √h scaling for horizon,
    - Max Drawdown on simple returns,
    - returns also: NAV, last-price weight map, and weight vector.
- **analytics/risk_utils.py**
  - `returns(...)` – log or simple returns, drops empty rows.
  - `to_simple(...)` – convert log → simple.
  - `portfolio_nav_and_weights(...)` – NAV from last prices and position weights.
- **data/portfolio_loader.py**
  - `load_trades(...)` – read trades from Excel, normalize fields, compute `cash_balance` (deposits + SELL − BUY).
  - `build_holdings(...)` – reconstruct final holdings and cumulative trade history.
- **data/prices.py**
  - `get_prices(...)` – download `Close` prices from **Yahoo Finance** (auto-adjust), drop no-trade days; normalize GPW tickers to `.WA`.
- **data/valuation_loader.py**
  - Robust PLN parsing (PL/EN formats with spaces/commas/“zł/PLN”).
  - `load_valuation_sheet(...)` – returns `Ticker, TargetPrice, PriceAtPublication, Confidence, Views`.
  - `load_tickers_from_valuation(...)` – ticker list from the sheet.
- **optimization/risk_parity.py**
  - `shrink_cov(...)` – returns covariance with a diagonal ridge.
  - `risk_parity_weights(...)` – SLSQP optimization for equal risk contributions with `sum(w)=1`, `w∈[w_min,w_max]`.
- **optimization/black_litterman.py**
  - `bl_minimal(...)` – Black–Litterman via PyPortfolioOpt:
    - prior: `π = δ Σ w_mkt`,
    - posterior: `μ_BL`,
    - Markowitz weights: `w_BL = (1/δ) Σ^{-1} μ_BL`.
- **optimization/constraints.py**
  - `project_boxed_simplex(...)` – projects weights onto a boxed simplex (`sum(w)=s`, `w∈[lb,ub]`) using bisection.
- **optimization/upside.py**
  - `filter_upside(...)` – filters stocks by `Views >= min_upside`, sorted descending.
- **reporting/exporter.py**
  - `export_report_xlsx(...)` – Excel report with sheets:
    - Summary (NAV, σ, VaR, ES, MDD, cash),
    - Weights (Now, RP, BL, BL_Box in %),
    - Holdings, Prices_Tail (last 10 days),
    - Config (run parameters).
  - Auto-fit column widths.
- **main.py**
  - Reads `config.yaml`, selects tickers (valuation sheet → trades fallback).
  - Upside filter with threshold and min count guard.
  - Fetches prices, rebuilds holdings, computes empirical risk.
  - Risk Parity on covariance of log-returns.
  - Black–Litterman:
    - annualized `Σ` (`* trading_days`),
    - `w_mkt` derived from Risk Parity (not market-cap),
    - builds `P` (absolute views), `Q = Views − r_f`, `Ω` ~ `diag(P τ Σ Pᵀ)/(conf²)`,
    - raw BL weights and boxed-simplex projection version.
  - Exports report to `output/portfolio_risk_report.xlsx`.
- **config.yaml**
  - Date range, I/O paths, risk params (`var_confidence`, `var_horizon_days`, `risk_window_days`, `trading_days`, `use_log_returns`),
  - Upside filter (`min_upside`, `min_tickers_after_filter`),
  - BL constraints (`bl_box_lb`, `bl_box_ub`), `risk_free_rate`.
- **requirements.txt**
  - `pandas`, `numpy`, `scipy`, `pypfopt`, `yfinance`, `PyYAML`, `openpyxl`, `xlsxwriter`, `python-dateutil`.

### Changed
- — (initial release: nothing to compare)

### Fixed
- — (initial release: none)

### Known limitations
- Report does not yet include monitoring sheets (rolling vol, correlations, limit alerts).

---

**Author:** Hossa ProCapital Risk Management Team  
**Release date:** 2025-11-09
