"""How often the estimate fails to cover the true committee weight today.

`analyse_mainnet.py` asks one question: how much inflation does the estimate need
to hit its 99.99% target? This script asks the reverse. The spec is unchanged, so
what coverage does the current padding deliver?

The target comes from the 2023 notebook, which set `MIN_PROB_SAFETY = 0.9999`. The
padded estimate must fail to cover the true weight at most 1 time in 10,000.

The current padding is 0.5% on the cross-boundary branch and nothing on the
same-epoch branch.

Run: .venv/bin/python src/coverage_today.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

import analyse_mainnet  # noqa: E402
import model  # noqa: E402
import same_epoch_branch as same_epoch  # noqa: E402

TARGET_FAILURE_RATE = 0.0001
Z = float(norm.ppf(1.0 - TARGET_FAILURE_RATE))
SPE = model.SLOTS_PER_EPOCH
CROSS_BOUNDARY_PADDING = 0.005
SAME_EPOCH_PADDING = 0.0
RESULTS = Path(__file__).parent.parent / "results"


def failure_rate(required_margin: float, padding: float) -> tuple[float, float]:
    """Convert a padding into a failure rate.

    The required margin for the target is `Z` standard deviations. So one standard
    deviation is `required_margin / Z`, and the padding covers `padding / sd` of
    them. The failure rate is the normal tail beyond that point.
    """
    sd = required_margin / Z
    standard_deviations = padding / sd
    return standard_deviations, float(1.0 - norm.cdf(standard_deviations))


def closed_form_table(population: model.Population) -> pd.DataFrame:
    rows = []
    for k in (1, 2, 4, 8, 16, 31):
        required = same_epoch.required_margin(population, k)
        sds, fails = failure_rate(required, SAME_EPOCH_PADDING)
        rows.append(
            {
                "branch": "same-epoch",
                "range": f"k={k}",
                "needs_pct": round(required * 100, 3),
                "gets_pct": SAME_EPOCH_PADDING * 100,
                "padding_in_sd": round(sds, 2),
                "fails_pct": round(fails * 100, 2),
            }
        )
    for x, y in ((1, 1), (3, 5), (8, 8), (16, 16), (31, 31)):
        required = model.required_margin_analytic(population, x, y, Z)
        sds, fails = failure_rate(required, CROSS_BOUNDARY_PADDING)
        rows.append(
            {
                "branch": "cross-boundary",
                "range": f"x={x} y={y}",
                "needs_pct": round(required * 100, 3),
                "gets_pct": CROSS_BOUNDARY_PADDING * 100,
                "padding_in_sd": round(sds, 2),
                "fails_pct": round(fails * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def sampled_check(population: model.Population, trials: int = 60_000) -> None:
    """Confirm the closed form with exact draws at a few points."""
    rng = np.random.default_rng(11)
    print(f"Exact-sampling check ({trials // 1000}k trials):", flush=True)

    for k in (1, 4, 31):
        n_sample = k * population.validators_per_committee(SPE)
        drawn = rng.multivariate_hypergeometric(
            population.counts.astype(np.int64), n_sample, size=trials
        )
        actual = drawn @ population.values_gwei.astype(np.float64)
        estimate = same_epoch.same_epoch_estimate_gwei(population, k)
        fails = float(np.mean(actual > estimate))
        print(f"  same-epoch     k={k:<2}      fails {fails * 100:>5.1f}%", flush=True)

    for x, y in ((1, 1), (3, 5)):
        actual = model.sample_actual_weight(population, x, y, trials, rng, SPE)
        estimate = model.pro_rata_estimate_gwei(population, x, y, SPE) * (
            1.0 + CROSS_BOUNDARY_PADDING
        )
        fails = float(np.mean(actual > estimate))
        print(f"  cross-boundary x={x} y={y}    fails {fails * 100:>5.1f}%", flush=True)


def main() -> None:
    population, payload = analyse_mainnet.load_population()

    print("Spec unchanged. Mainnet balances from", payload["fetched_utc"])
    print(f"Active validators: {population.n:,}")
    print(
        f"Target: the padded estimate fails to cover the true weight at most "
        f"{TARGET_FAILURE_RATE * 100:.2f}% of the time.\n"
    )

    df = closed_form_table(population)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    print("Reading: 'padding_in_sd' is the padding measured in standard deviations")
    print("of the true committee weight. The target needs", f"{Z:.2f}.")
    print()

    sampled_check(population)

    out = RESULTS / "coverage_today.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
