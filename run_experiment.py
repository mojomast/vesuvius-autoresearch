#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Vesuvius ink-detection experiment")
    parser.add_argument("--config", required=True, help="YAML/JSON experiment config")
    args = parser.parse_args()
    result = run_experiment(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
