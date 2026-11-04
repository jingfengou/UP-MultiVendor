"""Script form of `python -m up yolo26`, suitable for torchrun."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from up.commands.yolo26 import add_arguments  # noqa: E402


if __name__ == '__main__':
    arguments = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    arguments.run(arguments)
