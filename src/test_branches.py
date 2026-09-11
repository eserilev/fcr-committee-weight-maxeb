"""Checks on the premises the analysis rests on.

1. Which branch of `estimate_committee_weight_between_slots` a slot range takes,
   and which `(x, y)` splits the pro-rata branch actually reaches. The analysis
   sweeps `reachable_slot_splits()`. This function brute-forces every slot range
   in a multi-epoch window and confirms that set.

2. That the estimate is the mean of the modelled actual weight, so the estimator
   is unbiased and needs inflation to become an upper bound.

3. That `spec_pro_rata_estimate_gwei`, which mirrors the spec's integer
   arithmetic, agrees with the exact rational form to within the floors.

Run: .venv/bin/python src/test_branches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import model  # noqa: E402

SPE = model.SLOTS_PER_EPOCH


def is_full_validator_set_covered(start_slot: int, end_slot: int) -> bool:
    """Spec: `is_full_validator_set_covered`."""
    start_full_epoch = (start_slot + SPE - 1) // SPE
    end_full_epoch = (end_slot + 1) // SPE
    return start_full_epoch < end_full_epoch


def classify(start_slot: int, end_slot: int):
    """Return the branch the spec takes, and the derived slot counts."""
    if start_slot > end_slot:
        return "empty", None
    if is_full_validator_set_covered(start_slot, end_slot):
        return "full_epoch", None

    start_epoch = start_slot // SPE
    end_epoch = end_slot // SPE
    if start_epoch == end_epoch:
        return "same_epoch", end_slot - start_slot + 1

    num_slots_in_end_epoch = (end_slot % SPE) + 1
    num_slots_in_start_epoch = SPE - (start_slot % SPE)
    return "cross_boundary", (num_slots_in_end_epoch, num_slots_in_start_epoch)


def check_reachable_splits() -> None:
    found = set()
    same_epoch_k = set()
    # Four epochs is enough. Ranges longer than that always cover a full epoch.
    horizon = 4 * SPE
    for start in range(horizon):
        for end in range(start, horizon):
            branch, detail = classify(start, end)
            if branch == "cross_boundary":
                found.add(detail)
            elif branch == "same_epoch":
                same_epoch_k.add(detail)

    expected = set(model.reachable_slot_splits())
    assert found == expected, (
        f"reachable splits disagree: "
        f"missing {sorted(expected - found)[:5]}, extra {sorted(found - expected)[:5]}"
    )
    print(f"OK  cross-boundary branch reaches exactly {len(found)} (x, y) splits")
    print(f"    x in 1..{max(x for x, _ in found)}, y in 1..{max(y for _, y in found)}")

    assert same_epoch_k == set(range(1, SPE)), f"same-epoch k set: {sorted(same_epoch_k)}"
    print(f"OK  same-epoch branch reaches k in 1..{max(same_epoch_k)}")


def check_named_scenarios() -> None:
    """The concrete situations quoted in the write-up."""
    cases = [
        ((41, 41), "same_epoch", 1),
        ((41, 42), "same_epoch", 2),
        ((37, 41), "same_epoch", 5),
        ((31, 32), "cross_boundary", (1, 1)),
        ((31, 34), "cross_boundary", (3, 1)),
        ((27, 34), "cross_boundary", (3, 5)),
        ((32, 39), "same_epoch", 8),
        ((32, 62), "same_epoch", 31),
        ((0, 31), "full_epoch", None),
        # 32 slots long and still covers no full epoch, so it takes the
        # pro-rata branch rather than the full-epoch short circuit.
        ((5, 36), "cross_boundary", (5, 27)),
    ]
    for (start, end), want_branch, want_detail in cases:
        branch, detail = classify(start, end)
        assert branch == want_branch, f"[{start},{end}] branch {branch} != {want_branch}"
        if want_detail is not None:
            assert detail == want_detail, f"[{start},{end}] detail {detail} != {want_detail}"
    print(f"OK  {len(cases)} named slot ranges classify as expected")


def check_unbiased() -> None:
    """The estimate is the mean of the actual weight, so it covers it about half the time.

    Two versions. With sample sizes `x * N / spe` the identity is exact:

        mu * E[M] = (T/N) * (N/spe) * (x + y*(spe-x)/spe) = estimate

    The notebook, and `model.sample_sizes`, use `vpc = N // spe`. That floor makes
    the sample sizes slightly small, which biases the mean down by around 1e-5
    relative. That is four orders of magnitude below the margins under study, so
    the analysis keeps the notebook's convention for comparability.
    """
    population = model.two_point_population(600_000 * 32, 0.4)
    n = population.n
    mu = population.mean

    worst_exact = 0.0
    worst_floored = 0.0
    for x, y in model.reachable_slot_splits():
        estimate = model.pro_rata_estimate_gwei(population, x, y)

        # Exact sample sizes, no floor.
        n_b = x * n / SPE
        n_s = y * n / SPE
        mean_m_exact = n_b + n_s * (n - n_b) / n
        worst_exact = max(worst_exact, abs(mu * mean_m_exact / estimate - 1.0))

        mean, _ = model.actual_weight_moments(population, x, y)
        worst_floored = max(worst_floored, abs(mean / estimate - 1.0))

    assert worst_exact < 1e-12, f"exact-size estimator is biased by {worst_exact}"
    assert worst_floored < 1e-4, f"floored estimator is biased by {worst_floored}"
    print(
        f"OK  estimate equals the mean actual weight "
        f"(exact sizes: {worst_exact:.2e}, notebook vpc floor: {worst_floored:.2e})"
    )


def check_spec_integer_form() -> None:
    population = model.two_point_population(600_000 * 32, 0.4)
    worst = 0.0
    for x, y in model.reachable_slot_splits():
        exact = model.pro_rata_estimate_gwei(population, x, y)
        spec = model.spec_pro_rata_estimate_gwei(population, x, y)
        worst = max(worst, abs(spec / exact - 1.0))
    assert worst < 1e-6, f"spec integer form differs by {worst}"
    print(f"OK  spec integer arithmetic matches the rational form (max gap {worst:.2e})")


def check_adjust() -> None:
    """`adjust_committee_weight_estimate_to_ensure_safety` delivers at least 0.5%."""
    for estimate in (1, 999, 1000, 1001, 10**9, 10**18, 12_345_678_901):
        adjusted = model.spec_adjust(estimate)
        assert adjusted >= estimate * 1005 / 1000, (
            f"adjust({estimate}) = {adjusted} is below the 0.5% margin"
        )
    print("OK  spec_adjust delivers at least the 0.5% margin on sampled inputs")


def main() -> None:
    check_reachable_splits()
    check_named_scenarios()
    check_unbiased()
    check_spec_integer_form()
    check_adjust()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
