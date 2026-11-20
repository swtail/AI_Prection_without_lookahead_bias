#!/usr/bin/env bash
# Nightly earnings run (scheduled 23:59 US/Eastern, Mon-Fri).
#
#   scripts/nightly.sh [YYYY-MM-DD]      (default: today in US/Eastern)
#
# 1. restores analysis state from the `analysis-state` branch
# 2. runs the pipeline for the date (exits quietly on NYSE holidays)
# 3. writes slim copies for Dropbox to ./upload/<date>/
# 4. commits state + full outputs back to `analysis-state` and pushes
#
# Prints a final line  RESULT <status> <date> <upload dir>  for the caller.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DAY="${1:-$(TZ=America/New_York date +%F)}"
WORK="${NIGHTLY_WORKDIR:-$REPO/.nightly}"
STATE_WT="$WORK/state-branch"
UPLOAD="$REPO/upload/$DAY"

cd "$REPO"
if ! python3 -c "import sys; from earnings.calendar import is_trading_day; sys.exit(0 if is_trading_day('$DAY') else 1)"; then
  echo "RESULT holiday $DAY -"
  exit 0
fi

: "${EARNINGS_USER_AGENT:?set EARNINGS_USER_AGENT (SEC requires a contact)}"

mkdir -p "$WORK"
git fetch -q origin analysis-state
rm -rf "$STATE_WT"
git worktree prune
git worktree add -q -B analysis-state "$STATE_WT" origin/analysis-state

python3 -W ignore -m earnings.pipeline --date "$DAY" --out "$STATE_WT/outputs" --state "$STATE_WT/state"

if [ ! -f "$STATE_WT/outputs/$DAY/firm_summary.csv" ]; then
  echo "RESULT no-output $DAY -"
  exit 1
fi

rm -rf "$UPLOAD"
python3 -W ignore -m earnings.export "$STATE_WT/outputs/$DAY" "$UPLOAD"

cd "$STATE_WT"
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "State and outputs for run as of $DAY 23:59 ET (code $(git -C "$REPO" rev-parse --short HEAD))"
  for i in 1 2 3 4; do
    if git push -q origin analysis-state; then break; fi
    sleep $((2 ** i))
  done
fi

echo "RESULT ok $DAY $UPLOAD"
