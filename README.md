# Fundusz Hossa Procapital - student fund

An analytical system for **portfolio risk management**, based on:

* **empirical risk measures (VaR, ES, Max Drawdown)**,
* **the Risk Parity model**,
* **the Black–Litterman (BL) model** with weight constraints.

The project integrates trade and valuation data from Excel, fetches historical prices from Yahoo Finance, computes portfolio risk metrics, and generates a comprehensive Excel report.

---

## Report contents

| Sheet           | Description                                                               |
| :-------------- | :------------------------------------------------------------------------ |
| **Summary**     | Portfolio value, volatility, VaR, ES, Max Drawdown, cash balance          |
| **Weights**     | Comparison of weights: current, Risk Parity, Black–Litterman, constrained |
| **Holdings**    | Current holdings of individual stocks                                     |
| **Prices_Tail** | Last 10 trading days of price data                                        |
| **Config**      | Configuration parameters used in the current session                      |

---

## Theory

### Empirical Risk

Risk measures are computed without assuming any specific return distribution:

* **VaR (Value-at-Risk)** – maximum expected loss not exceeded with probability `confidence`.
* **ES (Expected Shortfall)** – average loss in the tail beyond VaR.
* **Max Drawdown** – largest peak-to-trough decline in the portfolio’s value.

---

### Risk Parity

The goal is to achieve **equal risk contribution** for each asset in the portfolio:

$$
RC_i = w_i(\Sigma w)_i
$$

$$
RC_i = \frac{1}{n}, \quad i = 1, \dots, n
$$

The solution is obtained using **SLSQP optimization** with weight-sum and bound constraints
$w_{min}, w_{max}$.

---

### Black–Litterman (BL)

The model combines **equal risk contribution (Risk Parity)** with **subjective analyst views**:

$$
\mu_{\mathrm{BL}}
= \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
\left[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} Q\right]
$$

where:

* $π = δΣw_{PR}$ – equilibrium (prior) returns from the Risk Parity model,
* $P, Q$ – matrix and vector of analyst views (expected “upside”),
* $Ω$ – view error covariance (confidence-weighted),
* $τ$ – uncertainty of prior estimates,
* $δ$ – risk aversion coefficient.

Resulting optimal weights are:

$$
w_{\mathrm{BL}} = \frac{1}{\delta}\Sigma^{-1}\mu_{\mathrm{BL}}
$$

In this project, a **practical constrained version** with bounds `bl_box_lb, bl_box_ub` was implemented,
so that the resulting weights can be directly interpreted as realistic portfolio allocations.

---

## Project structure

```
fundusz-hossa-procapital/
│
├── analytics/
│   ├── risk_metrics.py       # Empirical portfolio risk (VaR, ES, MDD)
│   └── risk_utils.py         # Returns, NAV, conversions
│
├── data/
│   ├── portfolio_loader.py   # Load transactions and holdings
│   ├── prices.py             # Download prices from Yahoo Finance
│   └── valuation_loader.py   # Load company valuations
│
├── optimization/
│   ├── risk_parity.py        # Risk Parity (SLSQP)
│   ├── black_litterman.py    # Black–Litterman model (PyPortfolioOpt)
│   ├── constraints.py        # Weight projection onto a boxed simplex
│   └── upside.py             # Filter stocks by "upside"
│
├── reporting/
│   └── exporter.py           # Excel report generation
│
├── input/                    # Input files (trades, valuations)
├── output/                   # Output reports
├── config.yaml               # Configuration parameters
├── main.py                   # Main execution script
└── requirements.txt          # Python dependencies
```

