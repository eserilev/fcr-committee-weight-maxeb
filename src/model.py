"""Model of the FCR committee weight estimate and its error.

Reference: `estimate_committee_weight_between_slots` in
consensus-specs/specs/phase0/fast-confirmation.md, and the calibration notebook
at https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20

The estimate covers a slot range that spans an epoch boundary and covers no full
epoch. Say the range holds `x` slots in the end (current) epoch and `y` slots in
the start (previous) epoch. The spec estimates the weight as

    committee_weight * (x + y * (SLOTS_PER_EPOCH - x) / SLOTS_PER_EPOCH)

where `committee_weight = total_active_balance // SLOTS_PER_EPOCH`. The pro-rata
term is there because the previous epoch's committees overlap the current
epoch's committees. In expectation `x / SLOTS_PER_EPOCH` of any previous-epoch
committee is already counted.

The estimate covers the total effective balance of the *union* of two sets:

  - B, the validators in the `x` end-epoch committees
  - S, the validators in the `y` start-epoch committees

Committees inside one epoch partition the active set, and the two epochs shuffle
independently. So B and S are independent simple random samples without
replacement, of sizes `x * vpc` and `y * vpc`.

The 2023 notebook gave every validator unit weight. Then `weight(B)` is the
constant `x * vpc`, and all the randomness sits in `|S \\ B|`, which is
hypergeometric. Post-Electra effective balances range from 32 to 2048 ETH, so
`weight(B)` is a random sum as well. This module models the general case.

Two facts make the general case cheap to compute.

1. `|B union S| = nB + K` with `K ~ Hypergeometric(N - nB, nB, nS)`. Given its
   size M, the union is a uniform random subset of size M. The joint law of
   (B, S) does not change if validators are relabelled. So the law of the union
   does not change either. A relabelling-invariant law on size-M subsets is
   uniform.

2. The population is a histogram over a few distinct balances. The composition
   of a uniform size-M subset is multivariate hypergeometric. That samples
   exactly, and fast, with no permutation of the validator set.

The estimate equals the mean of the actual weight exactly:

    mu * E[M] = (T/N) * (N/spe) * (x + y*(spe-x)/spe) = (T/spe) * (...)

So the estimate is unbiased, and it is at or above the actual weight only about
half the time. `COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR` exists to inflate
it to a high-probability upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOTS_PER_EPOCH = 32
GWEI_PER_ETH = 10**9

# The per mille value in the spec: ceil(estimate/1000) * (1000 + 5).
SPEC_ADJUSTMENT_FACTOR = 5


@dataclass(frozen=True)
class Population:
    """A validator set as a histogram of effective balances.

    `values_gwei` holds the distinct effective balances. `counts` holds how many
    validators carry each. Order does not matter.
    """

    values_gwei: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        if self.values_gwei.shape != self.counts.shape:
            raise ValueError("values and counts must have the same shape")
        if np.any(self.counts < 0):
            raise ValueError("counts must not be negative")
        if np.any(self.values_gwei <= 0):
            raise ValueError("balances must be positive")

    @property
    def n(self) -> int:
        """Number of validators."""
        return int(self.counts.sum())

    @property
    def total_active_balance(self) -> int:
        """Total effective balance in Gwei."""
        return int((self.values_gwei.astype(object) * self.counts.astype(object)).sum())

    @property
    def mean(self) -> float:
        return self.total_active_balance / self.n

    @property
    def variance(self) -> float:
        """Population variance of the effective balance, with an N divisor."""
        mu = self.mean
        return float(np.sum(self.counts * (self.values_gwei - mu) ** 2) / self.n)

    @property
    def cv(self) -> float:
        """Coefficient of variation of the effective balance."""
        return float(np.sqrt(self.variance) / self.mean)

    def validators_per_committee(self, spe: int = SLOTS_PER_EPOCH) -> int:
        """`vpc` as the notebook computes it: floor division of the validator count."""
        return self.n // spe


def uniform_population(num_validators: int, balance_eth: float = 32.0) -> Population:
    """Every validator carries the same effective balance. Reproduces the 2023 model."""
    return Population(
        values_gwei=np.array([int(balance_eth * GWEI_PER_ETH)], dtype=np.int64),
        counts=np.array([num_validators], dtype=np.int64),
    )


def two_point_population(
    total_stake_eth: float,
    consolidated_stake_fraction: float,
    large_balance_eth: float = 2048.0,
    small_balance_eth: float = 32.0,
) -> Population:
    """Total stake is fixed. A fraction of it sits in large validators.

    This holds total stake constant while consolidation reduces the validator
    count. Both effects matter, so varying them together is the point.
    """
    if not 0.0 <= consolidated_stake_fraction <= 1.0:
        raise ValueError("consolidated_stake_fraction must lie in [0, 1]")

    large_stake = total_stake_eth * consolidated_stake_fraction
    small_stake = total_stake_eth - large_stake
    num_large = int(round(large_stake / large_balance_eth))
    num_small = int(round(small_stake / small_balance_eth))

    values = []
    counts = []
    if num_small > 0:
        values.append(int(small_balance_eth * GWEI_PER_ETH))
        counts.append(num_small)
    if num_large > 0:
        values.append(int(large_balance_eth * GWEI_PER_ETH))
        counts.append(num_large)

    return Population(
        values_gwei=np.array(values, dtype=np.int64),
        counts=np.array(counts, dtype=np.int64),
    )


def histogram_population(balance_eth_to_count: dict[float, int]) -> Population:
    """Build a population from an explicit effective balance histogram.

    Use this with real mainnet data. See README.md for how to collect it.
    """
    items = sorted(balance_eth_to_count.items())
    return Population(
        values_gwei=np.array([int(b * GWEI_PER_ETH) for b, _ in items], dtype=np.int64),
        counts=np.array([c for _, c in items], dtype=np.int64),
    )


def sample_sizes(
    population: Population, x: int, y: int, spe: int = SLOTS_PER_EPOCH
) -> tuple[int, int]:
    """Sizes of the two committee samples, following the notebook's `vpc` convention."""
    vpc = population.validators_per_committee(spe)
    return x * vpc, y * vpc


def pro_rata_estimate_gwei(
    population: Population, x: int, y: int, spe: int = SLOTS_PER_EPOCH
) -> float:
    """The spec estimate before the safety adjustment, as an exact rational value.

    `spec_pro_rata_estimate_gwei` mirrors the spec's integer arithmetic instead.
    """
    committee_weight = population.total_active_balance / spe
    return committee_weight * (x + y * (spe - x) / spe)


def spec_pro_rata_estimate_gwei(
    population: Population, x: int, y: int, spe: int = SLOTS_PER_EPOCH
) -> int:
    """The spec estimate before the safety adjustment, with the spec's floors.

    Mirrors the `else` branch of `estimate_committee_weight_between_slots`, where
    `x` is `num_slots_in_end_epoch` and `y` is `num_slots_in_start_epoch`.
    """
    committee_weight = population.total_active_balance // spe
    start_epoch_weight = committee_weight * y
    end_epoch_weight = committee_weight * x
    start_epoch_weight_pro_rated = start_epoch_weight // spe * (spe - x)
    return start_epoch_weight_pro_rated + end_epoch_weight


def spec_adjust(estimate: int, factor: int = SPEC_ADJUSTMENT_FACTOR) -> int:
    """`adjust_committee_weight_estimate_to_ensure_safety` from the spec."""
    ceil = (estimate + 999) // 1000
    return ceil * (1000 + factor)


def union_size_moments(
    population: Population, x: int, y: int, spe: int = SLOTS_PER_EPOCH
) -> tuple[float, float]:
    """Mean and variance of `M = |B union S|`, the number of distinct validators."""
    n = population.n
    n_b, n_s = sample_sizes(population, x, y, spe)
    good = n - n_b

    mean_k = n_s * good / n
    var_k = n_s * (good / n) * (n_b / n) * (n - n_s) / (n - 1)
    return n_b + mean_k, var_k


def actual_weight_moments(
    population: Population, x: int, y: int, spe: int = SLOTS_PER_EPOCH
) -> tuple[float, float]:
    """Mean and variance of the actual union weight.

    Law of total variance over `M`:

        Var = mu^2 * Var(M) + sigma^2 * E[M (N - M)] / (N - 1)

    The first term is the only one the 2023 unit-weight model had. The second
    term is what heterogeneous effective balances add.
    """
    n = population.n
    mu = population.mean
    sigma_sq = population.variance

    mean_m, var_m = union_size_moments(population, x, y, spe)
    e_m_sq = var_m + mean_m**2
    e_m_times_n_minus_m = n * mean_m - e_m_sq

    mean = mu * mean_m
    variance = mu**2 * var_m + sigma_sq * e_m_times_n_minus_m / (n - 1)
    return mean, variance


def required_margin_analytic(
    population: Population,
    x: int,
    y: int,
    z: float,
    spe: int = SLOTS_PER_EPOCH,
) -> float:
    """Relative inflation needed so the estimate covers the actual weight.

    Normal approximation: `r = z * SD(actual) / estimate`. `z` is the one-sided
    normal quantile for the target confidence. Use `validate.py` to check the
    approximation against exact sampling.
    """
    mean, variance = actual_weight_moments(population, x, y, spe)
    estimate = pro_rata_estimate_gwei(population, x, y, spe)
    return z * float(np.sqrt(variance)) / estimate


def sample_actual_weight(
    population: Population,
    x: int,
    y: int,
    num_trials: int,
    rng: np.random.Generator,
    spe: int = SLOTS_PER_EPOCH,
) -> np.ndarray:
    """Draw exact samples of the actual union weight, in Gwei.

    Two stages, following the module docstring. First draw the union size `M`.
    Then draw the balance composition of a uniform size-M subset, which is
    multivariate hypergeometric. Trials are grouped by `M` so each distinct size
    needs one vectorised call.
    """
    n = population.n
    n_b, n_s = sample_sizes(population, x, y, spe)
    good = n - n_b

    k = rng.hypergeometric(ngood=good, nbad=n_b, nsample=n_s, size=num_trials)
    m = n_b + k

    out = np.empty(num_trials, dtype=np.float64)
    counts = population.counts.astype(np.int64)
    values = population.values_gwei.astype(np.float64)

    for size in np.unique(m):
        idx = np.flatnonzero(m == size)
        drawn = rng.multivariate_hypergeometric(counts, int(size), size=idx.size)
        out[idx] = drawn @ values

    return out


def required_margin_mc(
    population: Population,
    x: int,
    y: int,
    confidence: float,
    num_trials: int,
    rng: np.random.Generator,
    spe: int = SLOTS_PER_EPOCH,
) -> tuple[float, float]:
    """Relative inflation needed, from exact sampling.

    Returns the required margin and the fraction of trials where the unadjusted
    estimate already covered the actual weight. The second value lands near 0.5,
    because the estimate is unbiased.
    """
    actual = sample_actual_weight(population, x, y, num_trials, rng, spe)
    estimate = pro_rata_estimate_gwei(population, x, y, spe)
    quantile = float(np.quantile(actual, confidence))
    prob_safe = float(np.mean(actual <= estimate))
    return max(0.0, quantile / estimate - 1.0), prob_safe


def reachable_slot_splits(spe: int = SLOTS_PER_EPOCH) -> list[tuple[int, int]]:
    """The `(x, y)` splits that reach the pro-rata branch of the spec function.

    The range spans an epoch boundary and covers no full epoch. `y = spe` covers
    the whole start epoch, and `x = spe` covers the whole end epoch, so
    `is_full_validator_set_covered` short-circuits both. That leaves
    `1 <= x <= spe - 1` and `1 <= y <= spe - 1`.
    """
    return [(x, y) for x in range(1, spe) for y in range(1, spe)]


def notebook_slot_splits(spe: int = SLOTS_PER_EPOCH) -> list[tuple[int, int]]:
    """The splits the 2023 notebook swept, `1..spe` on both axes.

    This includes splits the spec function never reaches. Kept so the
    reproduction matches the published table.
    """
    return [(x, y) for x in range(1, spe + 1) for y in range(1, spe + 1)]
