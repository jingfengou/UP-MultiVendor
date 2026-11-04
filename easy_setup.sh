#!/usr/bin/env bash
set -euo pipefail
# Activate the vendor environment and install its Torch/TorchVision first.
case "${UP_BUILD_PLATFORM:-none}" in
  none|nvidia) ;;
  *) echo "UP_BUILD_PLATFORM must be none or nvidia" >&2; exit 2 ;;
esac
python -m pip install -r requirements.txt -r requirements/export.txt
python -m pip install --no-deps -r requirements/yolo11.txt
UP_BUILD_PLATFORM=none python -m pip install --no-deps -e .
if [ "${UP_BUILD_PLATFORM:-none}" = nvidia ]; then
  python setup.py build_ext --inplace
fi
