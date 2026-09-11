"""The same-epoch branch of `estimate_committee_weight_between_slots`.

The 2023 notebook only studied ranges that span an epoch boundary. When the range
sits inside one epoch the spec takes a different branch:

    committee_weight = total_active_balance // SLOTS_PER_EPOCH
    return committee_weight * (end_slot - start_slot + 1)

There is no pro-rata term, because committees inside one epoch do not overlap.
There is no safety adjustment either. The estimate is `k` mean committee weights
for `k` slots.

Under unit weights that is nearly exact. Committees inside an epoch differ in size
by at most one validator, so the error is at most `k` validators.

Under MaxEB the committees hold a random mix of 32 ETH and 2048 ETH validators. So
the weight of `k` specific committees deviates from `k` mean committee weights by
a percentage.

This script measures that deviation. It also checks a constraint that blocks the
obvious fix.

`compute_honest_ffg_support_for_current_target` does

    ffg_weight_till_now = estimate_committee_weight_between_slots(
        total_active_balance, compute_start_slot_at_epoch(current_epoch), current_slot - 1
    )
    remaining_ffg_weight = total_active_balance - ffg_weight_till_now

That subtraction needs the estimate to stay at or below `total_active_balance`.
The range always sits inside the current epoch, so `k` reaches `SLOTS_PER_EPOCH - 1`.
At that point the estimate is already `31/32` of the total, which leaves only
about 3.2% of headroom. If the margin the estimate needs is larger than the
headroom, padding this branch cannot work. The estimate has to be clamped instead.

Run: .venv/bin/python src/same_epoch_branch.py
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
SPE = model.SLOTS_PER_EPOCH
NOTEBOOK_STAKE_ETH = 600_000 * 32


def same_epoch_estimate_gwei(population: model.Population, k: int) -> int:
    """The spec's same-epoch branch, with its floor."""
    return (population.total_active_balance // SPE) * k


def same_epoch_moments(population: model.Population, k: int) -> tuple[float, float]:
    """Mean and variance of the true weight of `k` committees inside one epoch.

    Together the `k` committees are a uniform random sample without replacement,
    of size `k * vpc`. This holds because the shuffle of one epoch partitions the
    validator set.
    """
    n = population.n
    n_sample = k * population.validators_per_committee(SPE)
    mu = population.mean
    sigma_sq = population.variance

    mean = n_sample * mu
    variance = n_sample * sigma_sq * (n - n_sample) / (n - 1)
    return mean, variance


def required_margin(population: model.Population, k: int) -> float:
    mean, variance = same_epoch_moments(population, k)
    estimate = same_epoch_estimate_gwei(population, k)
    return Z * float(np.sqrt(variance)) / estimate


def headroom_fraction(population: model.Population, k: int) -> float:
    """How much the estimate can grow before it passes `total_active_balance`."""
    estimate = same_epoch_estimate_gwei(population, k)
    return population.total_active_balance / estimate - 1.0


def main() -> None:
    print("Same-epoch branch: required margin by range length")
    print("Total stake 19.2M ETH. 'k' is the number of slots in the range.\n")

    rows = []
    for fraction in (0.0, 0.2, 0.5, 0.8):
        population = model.two_point_population(NOTEBOOK_STAKE_ETH, fraction)
        for k in (1, 2, 4, 8, 16, 24, 31):
            rows.append(
                {
                    "consolidated_stake_fraction": fraction,
                    "validators": population.n,
                    "cv": round(population.cv, 3),
                    "k_slots": k,
                    "required_pct": round(required_margin(population, k) * 100, 4),
                    "headroom_to_total_pct": round(headroom_fraction(population, k) * 100, 3),
                }
            )

    df = pd.DataFrame(rows)
    df["padding_fits_in_headroom"] = df["required_pct"] <= df["headroom_to_total_pct"]

    pd.set_option("display.width", 200)
    print(
        df.to_string(
            index=False, formatters={"consolidated_stake_fraction": "{:.2f}".format}
        )
    )
    print()

    print("Reading:")
    print("  'required_pct' is the inflation needed for 99.99% coverage.")
    print("  'headroom_to_total_pct' is how much the estimate can grow before it")
    print("  passes total_active_balance and breaks the subtraction in")
    print("  compute_honest_ffg_support_for_current_target.")
    print()

    conflicts = df[~df["padding_fits_in_headroom"]]
    print(
        f"rows where the needed padding does not fit in the headroom: "
        f"{len(conflicts)}/{len(df)}"
    )
    if len(conflicts) > 0:
        worst = conflicts.sort_values("k_slots", ascending=False).iloc[0]
        print(
            f"  example: consolidated={worst['consolidated_stake_fraction']:.2f}, "
            f"k={int(worst['k_slots'])} slots needs {worst['required_pct']:.2f}% "
            f"but only {worst['headroom_to_total_pct']:.2f}% is available"
        )

    # Exact-sampling check at one point.
    print("\nExact-sampling check (200k trials, k = 1):")
    rng = np.random.default_rng(20260910)
    for fraction in (0.0, 0.5):
        population = model.two_point_population(NOTEBOOK_STAKE_ETH, fraction)
        n_sample = population.validators_per_committee(SPE)
        drawn = rng.multivariate_hypergeometric(
            population.counts.astype(np.int64), n_sample, size=200_000
        )
        actual = drawn @ population.values_gwei.astype(np.float64)
        estimate = same_epoch_estimate_gwei(population, 1)
        sampled = float(np.quantile(actual, CONFIDENCE)) / estimate - 1.0
        print(
            f"  consolidated={fraction:.2f}  analytic={required_margin(population, 1) * 100:.3f}%"
            f"  sampled={sampled * 100:.3f}%"
            f"  P(estimate covers actual)={float(np.mean(actual <= estimate)):.3f}"
        )

    out = Path(__file__).parent.parent / "results" / "same_epoch_branch.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
