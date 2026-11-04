#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then echo "Usage: bash scripts/train.sh GPU_COUNT CONFIG [--opts KEY VALUE ...]" >&2; exit 2; fi
count=$1; config=$2; shift 2
exec python -m up train --ng "$count" --launch pytorch --config "$config" "$@"
