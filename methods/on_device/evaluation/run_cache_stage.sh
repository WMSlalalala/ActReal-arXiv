#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FEWSHOT_SESSION_ZIP WORK_DIR USER_ID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEWSHOT_ZIP="$1"
WORK_DIR="$2"
USER_ID="$3"

python "$SCRIPT_DIR/prepare_fewshot_refs.py" \
  "$FEWSHOT_ZIP" \
  --out "$WORK_DIR/refs"

python "$SCRIPT_DIR/build_frozen_personal_cache.py" \
  --refs "$WORK_DIR/refs" \
  --out "$WORK_DIR/cache" \
  --user-id "$USER_ID" \
  --device cuda
