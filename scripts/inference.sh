#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then echo "Usage: bash scripts/inference.sh CONFIG CHECKPOINT IMAGE_OR_DIR [--opts KEY VALUE ...]" >&2; exit 2; fi
config=$1; checkpoint=$2; images=$3; shift 3
exec python -m up inference --ng 1 --launch pytorch --config "$config" --ckpt "$checkpoint" --img_path "$images" "$@"
