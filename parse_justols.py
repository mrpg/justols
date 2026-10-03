#!/usr/bin/env -S uv run --no-project
"""Parse justols output: uv run parse_justols.py [output.tsv]."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Fit:
    """Coefficient fields match Rust's Coefficient; diagnostic names match CLI rows."""

    coefficients: dict[str, dict[str, float]]
    diagnostics: dict[str, float | int]


def parse_output(text: str) -> Fit:
    coefficients = {}
    diagnostics = {}
    counts = {"n", "df-model", "df-resid", "n-clusters"}

    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split("\t")
        if len(fields) == 5:
            name, estimate, std_error, t_stat, p_value = fields
            coefficients[name] = {
                "estimate": float(estimate),
                "std_error": float(std_error),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
        elif len(fields) == 2:
            name, value = fields
            diagnostics[name] = int(value) if name in counts else float(value)
        else:
            raise ValueError(
                f"line {line_number}: expected 2 or 5 tab-separated fields"
            )

    return Fit(coefficients, diagnostics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file", nargs="?", type=Path, help="justols output file (default: stdin)"
    )
    args = parser.parse_args()
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    print(parse_output(text))
