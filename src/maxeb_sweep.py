"""Required safety margin for the FCR committee weight estimate, post-Electra.

The 2023 calibration gave every validator 32 ETH. EIP-7251 raised the ceiling to
2048 ETH, so effective balances are now heterogeneous. Consolidation also cuts
the validator count at constant total stake.

This script holds total stake fixed and sweeps the consolidated fraction. For
each point it reports three values:

  - the validator count
  - the coefficient of variation of the effective balance
  - the inflation the estimate needs for 99.99% coverage

It also splits the required margin into two parts:

  - "count" is the margin from `Var(M)`, the number of distinct validators. This
    is the only term the 2023 model had.
  - "spread" is the margin from the variance of effective balances. This term is
    new.

The two add in quadrature, because they are the two terms of the law of total
variance.

Run: .venv/bin/python src/maxeb_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

import model  # noqa: E402

CONFIDENCE = 0.9999
Z = float(norm.ppf(CONFIDENCE))
SPEC_MARGIN_PCT = 0.5  # COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR = 5 per mille

# 19.2M ETH is 600,000 validators at 32 ETH. That is the notebook's largest row,
# so the zero-consolidation point reproduces its 0.3% figure exactly.
NOTEBOOK_STAKE_ETH = 600_000 * 32

# A rounded figure for the current mainnet deposit base, used as a second anchor.
MAINNET_STAKE_ETH = 34_000_000

CONSOLIDATED_FRACTIONS = [
    0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0
]


def margin_components(
    population: model.Population, x: int, y: int
) -> tuple[float, float, float]:
    """Return (total, count-only, spread-only) required margins.

    The two components come from the two terms of

        Var(actual) = mu^2 Var(M) + sigma^2 E[M (N - M)] / (N - 1)
    """
    n = population.n
    mu = population.mean
    sigma_sq = population.variance

    mean_m, var_m = model.union_size_moments(population, x, y)
    e_m_sq = var_m + mean_m**2
    e_m_times_n_minus_m = n * mean_m - e_m_sq

    var_count = mu**2 * var_m
    var_spread = sigma_sq * e_m_times_n_minus_m / (n - 1)
    estimate = model.pro_rata_estimate_gwei(population, x, y)

    return (
        Z * float(np.sqrt(var_count + var_spread)) / estimate,
        Z * float(np.sqrt(var_count)) / estimate,
        Z * float(np.sqrt(var_spread)) / estimate,
    )


def worst_case(population: model.Population) -> dict:
    """Largest required margin over the slot splits the spec branch reaches."""
    worst = (-1.0, 0.0, 0.0, (0, 0))
    for x, y in model.reachable_slot_splits():
        total, count, spread = margin_components(population, x, y)
        if total > worst[0]:
            worst = (total, count, spread, (x, y))
    total, count, spread, (x, y) = worst
    return {
        "worst_x": x,
        "worst_y": y,
        "required_pct": round(total * 100, 4),
        "count_component_pct": round(count * 100, 4),
        "spread_component_pct": round(spread * 100, 4),
    }


def sweep(total_stake_eth: float, label: str) -> pd.DataFrame:
    rows = []
    for fraction in CONSOLIDATED_FRACTIONS:
        population = model.two_point_population(total_stake_eth, fraction)
        row = {
            "scenario": label,
            "consolidated_stake_fraction": fraction,
            "validators": population.n,
            "cv_effective_balance": round(population.cv, 4),
        }
        row.update(worst_case(population))
        row["exceeds_spec_margin"] = row["required_pct"] > SPEC_MARGIN_PCT
        row["shortfall_x"] = round(row["required_pct"] / SPEC_MARGIN_PCT, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_against_sampling(total_stake_eth: float) -> None:
    """Check the normal approximation against exact multivariate-hypergeometric draws."""
    rng = np.random.default_rng(20260910)
    print("Exact-sampling check (300k trials, worst split for each point):")
    print(f"{'frac':>6} {'N':>9} {'cv':>7} {'analytic %':>11} {'sampled %':>10} {'P(covers)':>10}")
    for fraction in (0.0, 0.2, 0.5, 0.8, 0.95):
        population = model.two_point_population(total_stake_eth, fraction)
        worst = worst_case(population)
        x, y = worst["worst_x"], worst["worst_y"]
        mc, prob_safe = model.required_margin_mc(
            population, x, y, CONFIDENCE, 300_000, rng
        )
        print(
            f"{fraction:>6.2f} {population.n:>9,} {population.cv:>7.3f} "
            f"{worst['required_pct']:>11.3f} {mc * 100:>10.3f} {prob_safe:>10.3f}"
        )


def main() -> None:
    frames = [
        sweep(NOTEBOOK_STAKE_ETH, "19.2M ETH (notebook baseline)"),
        sweep(MAINNET_STAKE_ETH, "34M ETH (mainnet scale)"),
    ]
    df = pd.concat(frames, ignore_index=True)

    pd.set_option("display.width", 200)
    for label, frame in df.groupby("scenario", sort=False):
        print(f"=== {label} ===")
        print(
            frame.drop(columns=["scenario"]).to_string(
                index=False,
                formatters={"consolidated_stake_fraction": "{:.2f}".format},
            )
        )
        print()

    print(f"Spec margin is {SPEC_MARGIN_PCT}% "
          f"(COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR = 5 per mille).")
    print(f"Rows above that margin: {int(df['exceeds_spec_margin'].sum())}/{len(df)}")
    print()

    validate_against_sampling(NOTEBOOK_STAKE_ETH)

    out = Path(__file__).parent.parent / "results" / "maxeb_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
