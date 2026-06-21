#!/usr/bin/env bash
#
# End-to-end demo: generate ground truth, then run the anticipation pipeline
# on one ADT sequence.
#
# Usage:
#   ./run_demo.sh /path/to/Apartment_release_clean_seq150_M1292
#   ./run_demo.sh /path/to/seq --use_llm     # also query the LLM (needs OPENAI_API_KEY)
#
# Without --use_llm it runs perception-only (no API key, no cost).

set -euo pipefail

SEQ="${1:-}"
shift || true
EXTRA_ARGS="$*"   # e.g. --use_llm

if [[ -z "$SEQ" ]]; then
  echo "usage: $0 /path/to/ADT_sequence_folder [--use_llm]" >&2
  exit 1
fi
if [[ ! -d "$SEQ" ]]; then
  echo "error: sequence folder not found: $SEQ" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"

# Prefer the uv-managed .venv, then $PYTHON, then system python.
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="${PYTHON:-python}"
fi

cd "$REPO/src"

echo "==> [1/2] Generating ground truth for: $SEQ"
"$PY" gt.py --sequence_path "$SEQ"

echo "==> [2/2] Running anticipation pipeline ${EXTRA_ARGS:+($EXTRA_ARGS) }"
"$PY" main.py --sequence_path "$SEQ" $EXTRA_ARGS

echo "==> Done. Predictions under src/data/predictions/<sequence>/<params>/"
