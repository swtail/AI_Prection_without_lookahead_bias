# Prompt record and methods: AI-assisted earnings forecasting and valuation without lookahead bias

This document records what was asked of the AI coding agent, in order, so the procedure can be reported and reproduced. Part A is the verbatim prompt log. Part B is one consolidated prompt that reproduces the setup in a new session. Part C is a methods summary with references. Part D is an AI-use disclosure template.

---

## A. Prompt log (verbatim, chronological)

The researcher typed these instructions into the agent's prompt window. Spelling is left as typed.

| # | Prompt |
|---|---|
| 1 | AI prediction without lookahead bias. |
| 2 | I need to develop a program predict future earnings , 1q, 2q, 3q and 4q (1year) earnings based on all available information today, which is techinically today. I would like to run this process everyday at 7 pm ET from monday through friday on trading day after a firm announce earnings. 1. use today issued earnings announcement and earnings announcement in the past 5 , and 10 years of data, predict 1q, 2q, 3q and 4q (1year) earnings. 2. Use in addition to firm specific information, you use industry information and predict the same thing. 3. now in addition to firm specific information, industry information, use country specific analysis to predict future earnings . 4. using well known valuation models (discounted cash flow , residual income models) and your own model you developed based on your deepresearch on valuation, you predict value of firms. you can use firm info only, industry info only, country info only, or all three information together. |
| 3 | follow information sources that are accpeted by accounting and finance literature. |
| 4 | What variables should I collect in addition to this predicted earnings and target value? target value should be stock price. |
| 5 | *(Agent question)* Do you have WRDS access (Compustat, CRSP, I/B/E/S)? **Answer:** "You need to use public information, not fee driven data provider". *(Agent question)* Where should the daily 7 pm ET results be saved? **Answer:** "Dropbox folder". |
| 6 | let's start from yesterday 11/19 prediction now. |
| 7 | Once you do analysis on firm, industry and country, global, then do not erase it. you keep it using for next day so that you don't have to do it again the same thing over and over. And record all analysis and specification for your analysis such as OPus 5.5 used with medium effort. |
| 8 | first, build up the instructions to input in the prompt windows so that I can write in the paper about what I did |
| 9 | So then you automatically analyze for firms that announce earnings on the day? So do the analysis for firms announced earnings 2026/11/19 |
| 10 | for dropbox folder, make new folder name __AI_Predicted_earnings |
| 11 | analyze firms announced earnings today 2026/11/19 |
| 12 | you run the analysis 11:59 pm ET. okay? it is scheduled analsyis |
| 13 | no need to do for all firms, how many firms announce earnings for firms in russell 3000 firms? |
| 14 | let's narrow down the sample to reduce workload and token. I want to test only for firms with analyst following. check yahoo finance or IBES and only when there is analyst at least one folllowing the firm, analyze those firms. |
| 15 | i have wrds access |
| 16 | should not use 8k filing because it can be later than actual earnings announcement date |
| 17 | I need to get today's earnings announcement dates for US firms |
| 18 | can you not ask me to approve and allow? |

**Design decisions taken by the agent in response, so they can be reported:**
- **Universe.** The official Russell 3000 list is proprietary, and iShares blocks automated downloads of its IWV holdings. The sample therefore uses a Russell 3000 *proxy*: the 3,000 largest U.S. SEC filers by public float as reported on the 10-K cover page (XBRL `dei:EntityPublicFloat`). Royalty trusts, ETFs and funds are excluded. XBRL scale errors are corrected when float exceeds 300× book equity.
- **Analyst following.** Coverage is taken from Nasdaq's public analyst API (estimates by Zacks), keeping firms with at least one estimate. Yahoo Finance has no official API. I/B/E/S via WRDS needs a PostgreSQL connection with two-factor authentication, which an unattended cloud run cannot provide. A WRDS (I/B/E/S) export can replace the Nasdaq source when supplied.
- **Cutoff.** The run uses 23:59 ET. EDGAR stops accepting filings at 22:00 ET, so every same-day announcement is included.
- **Announcement dates.** In response to prompt 16, announcement dates are taken from the earnings calendar (Nasdaq; data by Zacks), which gives the press-release date. An Item 2.02 8-K may be filed up to four business days after the release. The 8-K is used only as the source of press-release text, and only when it was accepted before the cutoff. Otherwise no preliminary EPS is used for that quarter.
- **Sample on 2026-11-19.**
  - Final (calendar dates): 78 announcements → 23 in the Russell 3000 proxy → 20 with at least one analyst estimate.
  - The earlier 8-K-based selection found only 37 → 19 → 18. It missed Matthews International and Blaize, among others.
- **Schedule.** A scheduled task runs `scripts/nightly.sh` every weekday at 23:59 ET; it does nothing on NYSE holidays. It restores the saved analysis state from the `analysis-state` branch, runs the pipeline, pushes the updated state, and uploads the outputs to Dropbox `/__AI_Predicted_earnings/<date>/`.

---

## B. Consolidated reproducible prompt

Paste the following into a new agent session that has this repository attached, to reproduce the setup.

> Build a Python program, free of lookahead bias, that runs every NYSE trading day at 23:59 US/Eastern. It covers firms that announced earnings that day (SEC 8-K Item 2.02 accepted by 23:59 ET) **and** are in a Russell 3000 proxy (top 3,000 U.S. filers by 10-K public float) **and** have at least one analyst estimate. For each such firm it uses only information that was public at the cutoff.
>
> **Data:** public, literature-accepted, non-fee sources only:
> - SEC EDGAR XBRL company facts, using filing dates as availability dates; restatements are kept as later vintages.
> - SEC XBRL frames for the cross-section.
> - The 8-K EX-99.1 press release for the just-announced quarter (diluted GAAP EPS, flagged preliminary).
> - Kenneth French's Fama-French 48 industry definitions.
> - The Philadelphia Fed Real-Time Data Set (real GDP and the GDP deflator as known at each vintage).
> - The U.S. Treasury daily par yield curve.
> - Damodaran's implied equity risk premium, available from 15 January after each year end.
> - Daily closing prices from the Nasdaq public quote API.
>
> **Forecasts:** diluted EPS for 1, 2, 3 and 4 quarters ahead, plus the 4-quarter sum, each estimated with 5-year and with 10-year windows, under three information sets:
> 1. **Firm only.** Equal-weight combination of the seasonal random walk, the seasonal random walk with drift, and the Brown-Rozeff (1,0,0)x(0,1,1)_4 model.
> 2. **Firm + industry.** Pooled OLS of ROE at t+k on own ROE, seasonal ROE, loss indicators, and FF48 industry median ROE, estimated on industry peers.
> 3. **Firm + industry + country.** Model 2 plus country median ROE and its yearly change, real-time real GDP growth, the term spread, and the 10-year yield.
>
> Training uses only labels realized by the forecast date.
>
> **Valuation:** the target is the per-share stock price. Use:
> - Residual income (Gebhardt, Lee and Swaminathan 2001 fade to target ROE by year 12)
> - FCFE discounted cash flow (5-year growth fade to long-run nominal growth)
> - An Empirical-Persistence RIM, with ROE deviations from the information set's long-run ROE decaying at an estimated AR(1) rate
>
> Run these under firm-only, industry-only, country-only and all-information parameter sets. Cost of equity = 10-year Treasury + Blume-adjusted 60-month CAPM beta × implied ERP. Report value-to-price ratios.
>
> **Variables:** record, per firm and run, every input and output variable (see the README variable list), plus the full run specification: code version, data sources, cutoff time, AI model and effort level.
>
> **Persistence:** keep all firm, industry, country and global analyses and reuse them on later runs (refresh only quarters still being filed; never reuse state saved after a run's cutoff).
>
> **Output:** daily results (CSV, markdown report, specification) go to a Dropbox folder. The analysis state goes to an `analysis-state` git branch.
>
> Start with a run for 2026-11-19, 19:00 ET.

---

## C. Methods summary

**Point-in-time design.** Every datum carries two dates: the period it describes (event time) and the date it became public (availability time). A forecast made at time T uses only data with availability time ≤ T. Revised numbers enter as new vintages and never overwrite what was known earlier. Training samples for horizon k are limited to outcomes realized by T. Automated audits check this:
- A future-invariance test scrambles all post-T data and requires identical predictions.
- A feature-causality test re-computes features on truncated histories.

**Earnings announcements.** Same-day announcements are identified from 8-K filings with Item 2.02 ("Results of Operations and Financial Condition") accepted by 19:00 ET. Diluted GAAP EPS for the announced quarter is extracted from the EX-99.1 press release. Non-GAAP, "adjusted" and guidance figures are excluded, as are values that are implausible relative to the firm's last eight quarters. The 10-Q/10-K XBRL value supersedes the press-release value when filed.

**Quarterly data.** Quarterly values come from XBRL. Three-month values are used directly. Otherwise a quarter is the difference between year-to-date values with the same fiscal-year start; fiscal Q4 = annual − nine months.

**Earnings models.**
- *Firm:* Foster (1977) seasonal random walk with and without drift, and Brown and Rozeff (1979) SARIMA, combined with equal weights (Clemen 1989).
- *Industry:* pooled cross-sectional regressions of future ROE in the spirit of Fama and French (2000), Hou, van Dijk and Zhang (2012), and Li and Mohanram (2014). Industries are Fama-French 48.
- *Country:* aggregate earnings (Kothari, Lewellen and Warner 2006) and real-time macroeconomic data (Croushore and Stark 2001).

**Valuation.**
- Residual income: Ohlson (1995), as implemented by Gebhardt, Lee and Swaminathan (2001).
- Discounted cash flow: FCFE via the value-driver identity (Penman 2013).
- Empirical-Persistence RIM: linear information dynamics (Ohlson 1995; Dechow, Hutton and Sloan 1999), with ROE mean reversion toward the information set's long-run ROE (Fama and French 2000; Nissim and Penman 2001). Losses are allowed to be less persistent (Hayn 1995).

The composite value is the median of the three models. The cost of equity is the CAPM, with a Blume (1971)-adjusted beta from 60 monthly returns and Damodaran's implied ERP.

**Variables recorded per firm and run.**
- Identifiers: CIK, ticker, name, SIC, FF48, country, state, fiscal year end.
- Announcement: accession number, acceptance time, press-release EPS and snippet, whether it was used.
- Fundamentals:
  - EPS: latest quarter and trailing four quarters.
  - Equity, book value per share and diluted shares.
  - Trailing-four-quarter revenue, net income, operating cash flow, capital expenditure and dividends per share.
  - Assets, debt, cash, payout ratio and latest ROE.
- Market: price and price date, market cap, book-to-market, earnings yield, raw and Blume beta.
- Rates and macro: 10-year and 3-month yields, term spread, ERP, real GDP growth (year-over-year, annualized quarter-over-quarter, 10-year), inflation, long-run nominal growth.
- Forecasts: every forecast (information set × window × horizon) with its model components and regression diagnostics.
- Valuation: per-model values, V/P ratios and parameters (target ROE, persistence ω, long-run g, r).

**Variables to add for evaluation** (collected later from the same sources):
- Realized EPS for each forecast quarter, giving forecast errors scaled by price.
- Stock returns over 1, 3, 6 and 12 months after the forecast, to test whether V/P predicts returns (Frankel and Lee 1998).
- Three-day announcement returns.
- The earnings surprise relative to the seasonal random walk (SUE).

### References
- Blume, M. (1971). On the assessment of risk. *Journal of Finance*.
- Brown, L., and Rozeff, M. (1979). Univariate time-series models of quarterly accounting earnings per share. *Journal of Accounting Research*.
- Clemen, R. (1989). Combining forecasts: A review and annotated bibliography. *International Journal of Forecasting*.
- Croushore, D., and Stark, T. (2001). A real-time data set for macroeconomists. *Journal of Econometrics*.
- Dechow, P., Hutton, A., and Sloan, R. (1999). An empirical assessment of the residual income valuation model. *Journal of Accounting and Economics*.
- Fama, E., and French, K. (2000). Forecasting profitability and earnings. *Journal of Business*.
- Foster, G. (1977). Quarterly accounting data: Time-series properties and predictive-ability results. *The Accounting Review*.
- Frankel, R., and Lee, C. (1998). Accounting valuation, market expectation, and cross-sectional stock returns. *Journal of Accounting and Economics*.
- Gebhardt, W., Lee, C., and Swaminathan, B. (2001). Toward an implied cost of capital. *Journal of Accounting Research*.
- Hayn, C. (1995). The information content of losses. *Journal of Accounting and Economics*.
- Hou, K., van Dijk, M., and Zhang, Y. (2012). The implied cost of capital: A new approach. *Journal of Accounting and Economics*.
- Kothari, S.P., Lewellen, J., and Warner, J. (2006). Stock returns, aggregate earnings surprises, and behavioral finance. *Journal of Financial Economics*.
- Li, K., and Mohanram, P. (2014). Evaluating cross-sectional forecasting models for implied cost of capital. *Review of Accounting Studies*.
- Nissim, D., and Penman, S. (2001). Ratio analysis and equity valuation. *Review of Accounting Studies*.
- Ohlson, J. (1995). Earnings, book values, and dividends in equity valuation. *Contemporary Accounting Research*.
- Penman, S. (2013). *Financial Statement Analysis and Security Valuation*, 5th ed. McGraw-Hill.

---

## D. AI-use disclosure (template)

> The software was developed with an AI coding agent (Claude Code, Anthropic; model: [MODEL], reasoning effort: [EFFORT]) running in a cloud container with network access limited to the data sources listed above. The researcher specified the task in the prompts listed in Part A. The agent wrote the code, its unit tests and the point-in-time audits. All data came from public sources. Each daily run stores its full specification in `run_spec.json`: code commit, data sources, cutoff time, model and effort.

Each run's `run_spec.json` records the model and effort for that run.
