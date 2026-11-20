# Earnings forecasts and valuations - as of 2026-11-19 23:59:00-05:00

Announcements (8-K Item 2.02) processed: 18  
Panel training data through calendar quarter: 2026Q3  
10y Treasury 5.04%, Damodaran implied ERP 4.23%, real GDP growth (yoy, real-time vintage) 2.07%

## EPS forecasts (diluted, 10-year estimation window) and target prices

| Ticker | Name | FF48 | Last EPS | Release used | Firm 4Q | Firm+Ind 4Q | +Country 4Q | All 4Q | Price | Target (all) | V/P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAL | CALERES INC | 10 | 1.71 | no | 1.49 | 2.73 | 2.19 | 2.14 | 12.82 | 34.46 | 2.69 |
| POST | Post Holdings, Inc. | 2 | 1.68 | no | 5.91 | 5.61 | 4.64 | 5.39 | 79.28 | 124.04 | 1.56 |
| GAP | GAP INC | 42 | 0.86 | yes | 3.00 | 3.07 | 2.71 | 2.93 | 18.79 | 20.76 | 1.10 |
| ESE | ESCO TECHNOLOGIES INC | 36 | 1.26 | no | 11.62 | 7.74 | 7.39 | 8.91 | 223.02 | 106.41 | 0.48 |
| KEYS | Keysight Technologies, Inc. | 37 | 2.30 | no | 7.05 | 6.53 | 4.02 | 5.87 | 281.42 | 59.54 | 0.21 |
| INTU | INTUIT INC. | 34 | 2.59 | no | 18.23 | 10.80 | 11.41 | 13.48 | 310.02 | 147.68 | 0.48 |
| ESTC | Elastic N.V. | 34 | 0.00 | yes | 3.61 | 2.86 | 2.98 | 3.15 | 76.33 | 28.15 | 0.37 |
| NGVC | Natural Grocers by Vitamin Cot | 42 | 0.48 | yes | 2.05 | 1.81 | 1.33 | 1.73 | 27.15 | 25.44 | 0.94 |
| ROST | ROSS STORES, INC. | 42 | 2.66 | no | 8.50 | 7.93 | 6.84 | 7.76 | 219.88 | 70.29 | 0.32 |
| WSM | WILLIAMS SONOMA INC | 42 | 2.38 | yes | 9.68 | 8.15 | 7.52 | 8.45 | 220.66 | 59.12 | 0.27 |
| MMS | MAXIMUS, INC. | 34 | 1.82 | no | 7.01 | 5.52 | 5.79 | 6.11 | 62.00 | 87.94 | 1.42 |
| UGI | UGI CORP /PA/ | 31 | 0.04 | no | 2.84 | 2.53 | 2.08 | 2.48 | 35.10 | 30.80 | 0.88 |
| ROAD | Construction Partners, Inc. | 18 | 1.00 | yes | 2.58 | 2.75 | 1.81 | 2.38 | 72.24 | 36.56 | 0.51 |
| BBWI | Bath & Body Works, Inc. | 42 | 0.19 | no | 3.38 |  |  | 3.38 | 18.91 |  |  |
| VVV | VALVOLINE INC | 30 | 0.07 | no | 1.16 | 0.36 | 0.26 | 0.59 | 36.84 | 5.64 | 0.15 |
| LQDT | LIQUIDITY SERVICES INC | 34 | 0.36 | no | 1.18 | 1.07 | 1.13 | 1.12 | 33.03 | 13.51 | 0.41 |
| WMT | Walmart Inc. | 42 | 0.62 | yes | 2.55 | 2.25 | 1.87 | 2.23 | 126.04 | 40.18 | 0.32 |

## Not forecast

- Warner Music Group Corp. (CIK 1319161): insufficient EPS history in XBRL

## Notes

- Values for firms with negative book equity are not computed (ROE-based models undefined).
- Residual income book growth is capped at long-run growth (steady-state payout).
- 5-year-window panel models (especially the country model) are estimated on a short, post-2020 sample and can extrapolate macro effects; compare with the 10-year window and analyst consensus in firm_summary.csv.

## Sources

- SEC EDGAR XBRL companyfacts/frames/submissions, 8-K Item 2.02
- U.S. Treasury par yield curve
- Philadelphia Fed RTDSM (ROUTPUT, P)
- Damodaran implied ERP
- Kenneth French FF48 SIC definitions
- Nasdaq public quote API (daily close)

Files: `firm_summary.csv` (all variables), `eps_forecasts.csv` (by information set, window and horizon), `valuations.csv` (RIM, DCF, EP-RIM, composite, V/P), `run_meta.json`, `run_spec.json`.
