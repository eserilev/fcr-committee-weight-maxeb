"""Required safety margin for the FCR committee weight estimate, on real mainnet data.

Reads the effective balance histogram that `fetch_mainnet_balances.py` collects,
then reports the inflation the estimate needs for 99.99% coverage, for both
branches of `estimate_committee_weight_between_slots`.

It also names the concrete FCR call sites that reach each branch, so the numbers
map onto real behaviour rather than an abstract sweep.

Run: .venv/bin/python src/analyse_mainnet.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

import model  # noqa: E402
import same_epoch_branch as same_epoch  # noqa: E402

CONFIDENCE = 0.9999
Z = float(norm.ppf(CONFIDENCE))
SPE = model.SLOTS_PER_EPOCH
GWEI_PER_ETH = model.GWEI_PER_ETH
SPEC_CROSS_BOUNDARY_MARGIN_PCT = 0.5  # the 5 per mille adjustment
SPEC_SAME_EPOCH_MARGIN_PCT = 0.0  # that branch applies no adjustment

DATA = Path(__file__).parent.parent / "results" / "mainnet_effective_balances.json"
RESULTS = Path(__file__).parent.parent / "results"


def load_population() -> tuple[model.Population, dict]:
    payload = json.loads(DATA.read_text())
    histogram = {
        int(balance): count
        for balance, count in payload["effective_balance_gwei_to_count"].items()
    }
    items = sorted(histogram.items())
    population = model.Population(
        values_gwei=np.array([b for b, _ in items], dtype=np.int64),
        counts=np.array([c for _, c in items], dtype=np.int64),
    )
    return population, payload


def describe(population: model.Population, payload: dict) -> None:
    print("=== Mainnet active validator set ===")
    print(f"source            : {payload['source_url']}")
    print(f"fetched           : {payload['fetched_utc']}")
    print(f"active validators : {population.n:,}")
    print(
        f"total active bal  : {population.total_active_balance / GWEI_PER_ETH:,.0f} ETH"
    )
    print(f"mean eff. balance : {population.mean / GWEI_PER_ETH:,.2f} ETH")
    print(
        f"sd eff. balance   : {np.sqrt(population.variance) / GWEI_PER_ETH:,.2f} ETH"
    )
    print(f"cv eff. balance   : {population.cv:.4f}")
    print(f"distinct balances : {len(population.values_gwei):,}")

    order = np.argsort(-population.counts)[:8]
    print("\nmost common effective balances:")
    for i in order:
        balance = population.values_gwei[i] / GWEI_PER_ETH
        count = population.counts[i]
        share = 100 * count / population.n
        print(f"  {balance:>9,.2f} ETH  {count:>9,} validators  {share:>5.2f}%")

    at_32 = int(
        population.counts[population.values_gwei == 32 * GWEI_PER_ETH].sum()
    )
    above_32 = population.n - at_32
    stake_above = int(
        (
            population.values_gwei[population.values_gwei > 32 * GWEI_PER_ETH]
            * population.counts[population.values_gwei > 32 * GWEI_PER_ETH]
        ).sum()
    )
    print(
        f"\nvalidators above 32 ETH: {above_32:,} "
        f"({100 * above_32 / population.n:.2f}% of validators, "
        f"{100 * stake_above / population.total_active_balance:.2f}% of stake)"
    )
    print(f"largest effective balance: {population.values_gwei.max() / GWEI_PER_ETH:,.0f} ETH")
    print()


def cross_boundary_table(population: model.Population) -> pd.DataFrame:
    """Required margin for the pro-rata branch, over the reachable slot splits."""
    rows = []
    for x, y in model.reachable_slot_splits():
        margin = model.required_margin_analytic(population, x, y, Z)
        rows.append({"x_end_epoch_slots": x, "y_start_epoch_slots": y,
                     "required_pct": margin * 100})
    return pd.DataFrame(rows)


def same_epoch_table(population: model.Population) -> pd.DataFrame:
    rows = []
    for k in range(1, SPE):
        rows.append(
            {
                "k_slots": k,
                "required_pct": same_epoch.required_margin(population, k) * 100,
                "headroom_to_total_pct": same_epoch.headroom_fraction(population, k) * 100,
            }
        )
    return pd.DataFrame(rows)


def call_sites(population: model.Population) -> None:
    """Concrete FCR situations, with the branch and margin each one reaches.

    `compute_safety_threshold` calls
        estimate_committee_weight_between_slots(total, parent_slot + 1, current_slot - 1)
    so the range depends on the gap to the parent and on how long ago the block
    arrived. `compute_honest_ffg_support_for_current_target` calls it with
        (compute_start_slot_at_epoch(current_epoch), current_slot - 1)
    which never crosses a boundary.
    """
    print("=== Concrete FCR situations ===")
    print("SLOTS_PER_EPOCH is 32. Epoch 1 starts at slot 32.\n")

    scenarios = [
        ("safety threshold, no missed slots, mid-epoch",
         "parent 40, block 41, current 42", "same_epoch", 1, None),
        ("safety threshold, block 2 slots old",
         "parent 40, block 41, current 43", "same_epoch", 2, None),
        ("safety threshold, 4 missed slots before the block",
         "parent 36, block 41, current 42", "same_epoch", 5, None),
        ("safety threshold, block at the last slot of an epoch",
         "parent 30, block 31, current 33", "cross_boundary", None, (1, 1)),
        ("safety threshold, block at the last slot, 3 slots old",
         "parent 30, block 31, current 35", "cross_boundary", None, (3, 1)),
        ("safety threshold, missed slots across the boundary",
         "parent 26, block 33, current 35", "cross_boundary", None, (3, 5)),
        ("honest FFG support, 8 slots into the epoch",
         "range 32..39, current 40", "same_epoch", 8, None),
        ("honest FFG support, late in the epoch",
         "range 32..62, current 63", "same_epoch", 31, None),
    ]

    rows = []
    for label, slots, branch, k, split in scenarios:
        if branch == "same_epoch":
            margin = same_epoch.required_margin(population, k)
            padding = SPEC_SAME_EPOCH_MARGIN_PCT
            detail = f"k={k}"
        else:
            x, y = split
            margin = model.required_margin_analytic(population, x, y, Z)
            padding = SPEC_CROSS_BOUNDARY_MARGIN_PCT
            detail = f"x={x}, y={y}"
        rows.append(
            {
                "situation": label,
                "slots": slots,
                "branch": branch,
                "range": detail,
                "required_pct": round(margin * 100, 3),
                "spec_padding_pct": padding,
                "shortfall_x": ("inf" if padding == 0
                                else round(margin * 100 / padding, 1)),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))
    print()
    df.to_csv(RESULTS / "mainnet_call_sites.csv", index=False)


def validate(population: model.Population, trials: int = 100_000) -> None:
    """Compare the normal approximation against exact multivariate-hypergeometric draws.

    An exact draw costs `O(distinct balances)`, and mainnet carries over a thousand
    distinct effective balances. So this checks a few representative splits rather
    than the whole grid. The parametric sweep in `maxeb_sweep.py` checks the
    approximation more widely on coarser populations.
    """
    print(f"=== Exact-sampling check ({trials // 1000}k trials) ===", flush=True)
    rng = np.random.default_rng(20260910)

    print("cross-boundary branch:", flush=True)
    for x, y in ((1, 1), (3, 5)):
        analytic = model.required_margin_analytic(population, x, y, Z) * 100
        sampled, prob_safe = model.required_margin_mc(
            population, x, y, CONFIDENCE, trials, rng
        )
        print(
            f"  x={x:>2} y={y:>2}  analytic={analytic:>7.3f}%  "
            f"sampled={sampled * 100:>7.3f}%  P(covers)={prob_safe:.3f}",
            flush=True,
        )

    print("same-epoch branch:", flush=True)
    for k in (1, 4):
        analytic = same_epoch.required_margin(population, k) * 100
        n_sample = k * population.validators_per_committee(SPE)
        drawn = rng.multivariate_hypergeometric(
            population.counts.astype(np.int64), n_sample, size=trials
        )
        actual = drawn @ population.values_gwei.astype(np.float64)
        estimate = same_epoch.same_epoch_estimate_gwei(population, k)
        sampled = float(np.quantile(actual, CONFIDENCE)) / estimate - 1.0
        prob_safe = float(np.mean(actual <= estimate))
        print(
            f"  k={k:>2}        analytic={analytic:>7.3f}%  "
            f"sampled={sampled * 100:>7.3f}%  P(covers)={prob_safe:.3f}",
            flush=True,
        )
    print()


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            f"{DATA} is missing. Run src/fetch_mainnet_balances.py first."
        )

    population, payload = load_population()
    describe(population, payload)

    cross = cross_boundary_table(population)
    same = same_epoch_table(population)
    cross.to_csv(RESULTS / "mainnet_cross_boundary.csv", index=False)
    same.to_csv(RESULTS / "mainnet_same_epoch.csv", index=False)

    worst_cross = cross.loc[cross["required_pct"].idxmax()]
    worst_same = same.loc[same["required_pct"].idxmax()]

    print("=== Required margin for 99.99% coverage ===")
    print(
        f"cross-boundary branch: worst {worst_cross['required_pct']:.3f}% at "
        f"x={int(worst_cross['x_end_epoch_slots'])}, "
        f"y={int(worst_cross['y_start_epoch_slots'])}  "
        f"(spec pads {SPEC_CROSS_BOUNDARY_MARGIN_PCT}%, "
        f"short by {worst_cross['required_pct'] / SPEC_CROSS_BOUNDARY_MARGIN_PCT:.1f}x)"
    )
    print(
        f"cross-boundary branch: median {cross['required_pct'].median():.3f}%, "
        f"best {cross['required_pct'].min():.3f}%"
    )
    print(
        f"same-epoch branch    : worst {worst_same['required_pct']:.3f}% at "
        f"k={int(worst_same['k_slots'])}  (spec pads nothing)"
    )
    print(
        f"same-epoch branch    : at k=31 needs "
        f"{same[same['k_slots'] == 31]['required_pct'].iloc[0]:.3f}% with "
        f"{same[same['k_slots'] == 31]['headroom_to_total_pct'].iloc[0]:.3f}% headroom "
        f"before it passes total_active_balance"
    )
    print()

    call_sites(population)
    validate(population)

    print(f"wrote {RESULTS}/mainnet_cross_boundary.csv")
    print(f"wrote {RESULTS}/mainnet_same_epoch.csv")
    print(f"wrote {RESULTS}/mainnet_call_sites.csv")


if __name__ == "__main__":
    main()
