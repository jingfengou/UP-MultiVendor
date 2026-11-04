#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then echo "Usage: bash scripts/evaluate.sh GPU_COUNT CONFIG [--opts KEY VALUE ...]" >&2; exit 2; fi
count=$1; config=$2; shift 2
exec python -m up train --evaluate --ng "$count" --launch pytorch --config "$config" "$@"
