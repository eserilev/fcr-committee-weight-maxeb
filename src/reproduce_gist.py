"""Reproduce the published table from the 2023 calibration notebook.

This confirms the harness is right. The notebook gave every validator unit weight. It
swept `x` and `y` over 1..32. Per validator count, it reported the largest
inflation needed to cover the actual committee weight in 99.99% of trials.

Published result (notebook cell 7 output):

    validators   % increase
         9,375          1.9
        18,750          1.3
        37,500          1.0
        75,000          0.7
       150,000          0.5
       300,000          0.4
       600,000          0.3

Run: .venv/bin/python src/reproduce_gist.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

import model  # noqa: E402

CONFIDENCE = 0.9999
Z = float(norm.ppf(CONFIDENCE))

# The notebook swept 600_000 // [1, 2, 4, 8, 16, 32, 64].
NOTEBOOK_COUNTS = [600_000 // d for d in (64, 32, 16, 8, 4, 2, 1)]

PUBLISHED = {
    9_375: 1.9,
    18_750: 1.3,
    37_500: 1.0,
    75_000: 0.7,
    150_000: 0.5,
    300_000: 0.4,
    600_000: 0.3,
}

# The notebook scanned `np.arange(0.001, 0.5, 0.001)` and took the first value
# that met the bar, so its answers sit on a 0.1% grid, rounded up.
GRID = 0.001


def to_grid(margin: float) -> float:
    """Round up to the notebook's 0.1% grid, in percent."""
    return math.ceil(margin / GRID) * GRID * 100


def main() -> None:
    rows = []
    for count in NOTEBOOK_COUNTS:
        population = model.uniform_population(count)
        worst = 0.0
        worst_split = (0, 0)
        for x, y in model.notebook_slot_splits():
            margin = model.required_margin_analytic(population, x, y, Z)
            if margin > worst:
                worst = margin
                worst_split = (x, y)
        rows.append(
            {
                "validators": count,
                "worst_x": worst_split[0],
                "worst_y": worst_split[1],
                "analytic_pct": round(worst * 100, 4),
                "analytic_on_grid_pct": to_grid(worst),
                "published_pct": PUBLISHED[count],
            }
        )

    df = pd.DataFrame(rows)
    df["matches_published"] = np.isclose(
        df["analytic_on_grid_pct"], df["published_pct"], atol=1e-9
    )

    print("Unit weights, 99.99% confidence, notebook sweep x,y in 1..32")
    print(df.to_string(index=False))
    print()

    matched = int(df["matches_published"].sum())
    print(f"grid values matching the published table: {matched}/{len(df)}")

    # Spot-check the normal approximation against exact sampling at the two ends.
    rng = np.random.default_rng(20260910)
    print()
    print("Exact-sampling spot check (200k trials):")
    for count in (9_375, 600_000):
        population = model.uniform_population(count)
        row = df[df["validators"] == count].iloc[0]
        x, y = int(row["worst_x"]), int(row["worst_y"])
        mc, prob_safe = model.required_margin_mc(
            population, x, y, CONFIDENCE, 200_000, rng
        )
        print(
            f"  N={count:>7,} x={x:>2} y={y:>2}  "
            f"analytic={row['analytic_pct']:.3f}%  mc={mc * 100:.3f}%  "
            f"P(estimate covers actual)={prob_safe:.3f}"
        )

    out = Path(__file__).parent.parent / "results" / "reproduce_gist.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
