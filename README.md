# analysis-state

Persistent analysis state for the nightly earnings run (code: branch `claude/pensive-wright-yjb7a3`).
Each run restores `state/`, reuses closed quarters, industry peers and macro rows, and appends to
`state/history/` and `state/firms/<cik>/history.csv`. `outputs/<date>/` keeps every day's full results.
