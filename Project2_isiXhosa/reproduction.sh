#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command reproduction of every result in the extended abstract.
#
#   bash reproduction.sh                 # full run (~65 min on 2 CPU cores)
#   bash reproduction.sh --fast          # skip fastText + grid check (~35 min)
#   bash reproduction.sh --smoke         # 30% sample, code check only (~10 min)
#
# Other options:
#   --zips DIR      folder holding the two course zips (searched automatically
#                   in ., .., /content and /content/drive/MyDrive otherwise)
#   --no-install    do not pip-install requirements
#
# Data: needs data/TxtFiles and data/XhTxt.  If they are missing, the script
# unzips them from the course archives
#   1._TxtFiles1870-1894_TB.zip   and   TxtLangFiles1870-1894_TB.zip
# (any file names containing "TxtFiles1870-1894" / "TxtLangFiles1870-1894").
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

MODE="full"; INSTALL=1; ZIPDIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) MODE="fast" ;;
    --smoke) MODE="smoke" ;;
    --no-install) INSTALL=0 ;;
    --zips) ZIPDIR="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

say() { echo "[$(date +%H:%M:%S)] $*"; }
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || PY=python

# --------------------------------------------------------------- 1. packages
if [[ $INSTALL -eq 1 ]]; then
  say "Installing requirements"
  "$PY" -m pip install -q -r requirements.txt 2>/dev/null \
    || "$PY" -m pip install -q --break-system-packages -r requirements.txt
fi

# --------------------------------------------------------------- 2. data
count() { ls "$1" 2>/dev/null | grep -c '\.txt$' || true; }
if [[ $(count data/XhTxt) -lt 3000 || $(count data/TxtFiles) -lt 3000 ]]; then
  say "Setting up data/ from the course zips"
  all_zips() {
    local d
    for d in $ZIPDIR . .. /content /content/drive/MyDrive; do
      [[ -d "$d" ]] && find "$d" -maxdepth 2 -type f -name "*.zip" 2>/dev/null
    done
  }
  ZLANG=$(all_zips | grep "TxtLangFiles1870-1894" | head -1 || true)
  ZTXT=$(all_zips | grep "TxtFiles1870-1894" | grep -v "TxtLang" | head -1 || true)
  if [[ -z "$ZTXT" || -z "$ZLANG" ]]; then
    echo "ERROR: course zips not found. Put 1._TxtFiles1870-1894_TB.zip and"
    echo "TxtLangFiles1870-1894_TB.zip next to this script (or use --zips DIR)."
    exit 1
  fi
  say "  using $ZTXT"
  say "  using $ZLANG"
  rm -rf data/_raw && mkdir -p data/_raw/txt data/_raw/lang
  unzip -q -o "$ZTXT" -d data/_raw/txt
  unzip -q -o "$ZLANG" -d data/_raw/lang
  SRC_X=$(find data/_raw/lang -type d -name XhTxt -not -path "*MACOSX*" | head -1)
  SRC_T=$(dirname "$(find data/_raw/txt -type f -name 'IMVO_1884-11-03_p001.txt' -not -path '*MACOSX*' | head -1)")
  rm -rf data/XhTxt data/TxtFiles
  cp -r "$SRC_X" data/XhTxt
  cp -r "$SRC_T" data/TxtFiles
  rm -rf data/_raw
fi
say "data/XhTxt: $(count data/XhTxt) files, data/TxtFiles: $(count data/TxtFiles) files"

# --------------------------------------------------------------- 3. experiments
if [[ "$MODE" == "smoke" ]]; then
  OUT=outputs_smoke; mkdir -p "$OUT"
  say "Smoke run (30% of issues, no fastText) -> $OUT"
  "$PY" -u src/run_experiments.py --data data --out "$OUT" --seeds 13 --fast --smoke \
    > "$OUT/log_smoke.txt" 2>&1 || { tail -20 "$OUT/log_smoke.txt"; exit 1; }
else
  OUT=outputs; mkdir -p "$OUT"
  FLAG=""; [[ "$MODE" == "fast" ]] && FLAG="--fast"
  say "Running seed 13 (all analyses) and seeds 21+42 (ladder) in parallel -> $OUT"
  "$PY" -u src/run_experiments.py --data data --out "$OUT" --seeds 13 $FLAG \
    > "$OUT/log_seed13.txt" 2>&1 & P1=$!
  "$PY" -u src/run_experiments.py --data data --out "$OUT" --seeds 21 42 --e1-only $FLAG \
    > "$OUT/log_seeds21_42.txt" 2>&1 & P2=$!
  while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
    sleep 60
    say "  progress: $(tail -1 "$OUT/log_seed13.txt" | cut -c1-70) | $(tail -1 "$OUT/log_seeds21_42.txt" | cut -c1-50)"
  done
  wait $P1 || { echo "Seed-13 run failed:"; tail -20 "$OUT/log_seed13.txt"; exit 1; }
  wait $P2 || { echo "Seeds 21/42 run failed:"; tail -20 "$OUT/log_seeds21_42.txt"; exit 1; }
  if [[ "$MODE" == "full" ]]; then
    say "Grid-edge check"
    "$PY" -u src/check_grid_edge.py --data data --out "$OUT" > "$OUT/log_grid_edge.txt" 2>&1
  fi
fi

# --------------------------------------------------------------- 4. figures/tables
say "Building figures and tables"
"$PY" src/make_figures.py --out "$OUT" > "$OUT/log_figures.txt" 2>&1 \
  || { tail -20 "$OUT/log_figures.txt"; exit 1; }

say "Done. Results in $ROOT/$OUT (results.csv, stats.json, figures/)."
cat "$OUT/table_main.tex" 2>/dev/null | sed 's/\\\\//' | head -20
