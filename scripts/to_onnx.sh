#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then echo "Usage: bash scripts/to_onnx.sh CONFIG OUTPUT.onnx [--ckpt PATH] [--input_size 3x224x224] [--verify]" >&2; exit 2; fi
config=$1; output=$2; shift 2
exec python -m up to_onnx --config "$config" --save_prefix "$output" "$@"
