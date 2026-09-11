# Method

This file explains the model and the derivation. `README.md` holds the question,
the results, and the conclusions.

## The function under study

`estimate_committee_weight_between_slots` in
[`specs/phase0/fast-confirmation.md`](https://github.com/ethereum/consensus-specs/blob/dev/specs/phase0/fast-confirmation.md).
It estimates the total effective balance of the committees in a slot range. It has
four branches:

| Condition | Return |
|---|---|
| `start_slot > end_slot` | `0` |
| the range covers a full epoch | `total_active_balance` |
| the range sits inside one epoch | `committee_weight * k` for `k` slots |
| the range crosses an epoch boundary and covers no full epoch | a pro-rata formula, then a safety adjustment |

`committee_weight` is `total_active_balance // SLOTS_PER_EPOCH`. That is the mean
weight of the committee for one slot.

Only the fourth branch gets a safety adjustment:

```python
def adjust_committee_weight_estimate_to_ensure_safety(estimate: Gwei) -> Gwei:
    ceil = (estimate + 999) // 1000
    return ceil * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR)
```

`COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR` is 5, so the adjustment adds 0.5%.

## Why the estimate needs a margin at all

Write `x` for the number of slots the range holds in the end (later) epoch, and
`y` for the number in the start (earlier) epoch. The pro-rata estimate is

```
committee_weight * (x + y * (SLOTS_PER_EPOCH - x) / SLOTS_PER_EPOCH)
```

The quantity it estimates is the total effective balance of the **union** of two
validator sets:

- `B`, the validators in the `x` end-epoch committees
- `S`, the validators in the `y` start-epoch committees

The two sets overlap, because the same validators are shuffled again each epoch.
The pro-rata term corrects for that. In expectation, `x / SLOTS_PER_EPOCH` of any
start-epoch committee already sits in `B`, so each start-epoch committee adds only
`(SLOTS_PER_EPOCH - x) / SLOTS_PER_EPOCH` of new weight.

The correction is exact **in expectation** and nothing more. The union is random,
so the estimate lands above the actual weight roughly half the time. FCR needs an
upper bound, not a mean. That gap is what the adjustment factor fills.

## The sampling model

Beacon committees are not balance-weighted. `get_beacon_committee` shuffles the
active validator indices and slices the result. So the committees of one epoch
partition the active set. Each committee holds about `N / 32` validators, chosen
without regard to balance. Two epochs shuffle independently.

So:

- `B` is a simple random sample without replacement, size `nB = x * vpc`
- `S` is an independent simple random sample without replacement, size `nS = y * vpc`
- `vpc = N // SLOTS_PER_EPOCH`, following the 2023 notebook

The actual weight is `W(B union S)`.

## Two facts that make it cheap to compute

**1. The union size is hypergeometric, and the union is uniform given its size.**

`|B union S| = nB + K` where `K = |S \ B|` is hypergeometric with `N - nB` good
items, `nB` bad items, and `nS` draws.

Given `|B union S| = M`, the union is a uniform random subset of size `M`. The
joint law of `(B, S)` does not change if validators are relabelled. So the law of
the union does not change either. A relabelling-invariant law on size-`M` subsets
is the uniform one.

That removes the need to simulate two shuffles. One hypergeometric draw plus one
uniform subset gives an exact sample.

**2. The subset composition is multivariate hypergeometric.**

The population is a histogram over a few distinct effective balances. Drawing `M`
validators without replacement gives per-balance counts that follow a multivariate
hypergeometric law. `numpy` samples that directly, so no permutation of a
million-element array is needed. `model.sample_actual_weight` groups trials by `M`
so each distinct size costs one vectorised call.

## The closed form

Let `mu` and `sigma^2` be the mean and variance of the effective balance across
the population, with an `N` divisor. By the law of total variance:

```
E[actual]   = mu * E[M]
Var[actual] = mu^2 * Var(M)  +  sigma^2 * E[M (N - M)] / (N - 1)
```

The first term is the only one the 2023 model had, because unit weights make
`sigma = 0`. The second term is what heterogeneous effective balances add.

`Var(sum) = n * sigma^2 * (N - n) / (N - 1)` is the standard variance of a
without-replacement sample total.

The required relative margin for confidence `p` is then

```
r = z_p * sqrt(Var[actual]) / estimate
```

with `z_p` the one-sided normal quantile. `E[actual]` equals the estimate exactly
when the sample sizes are `x * N / 32`, which is the unbiasedness noted above.

The normal approximation is a convenience. `model.required_margin_mc` confirms
every headline number against exact sampling.

On heterogeneous populations the approximation runs slightly **low**, because the
true upper tail is right-skewed. The sampled figures are the ones to quote.

On equal-weight populations with few validators it runs **high**. The actual
weight is then an integer count with a small standard deviation. A 99.99% quantile
sits about 3.7 standard deviations out. Hypergeometric tails are far lighter than
normal that far out. This shows up only in the reproduction of the 2023 table,
where it is conservative.

## Same-epoch branch

Inside one epoch the committees do not overlap, so there is no pro-rata term. The
`k` committees together are a uniform random sample of size `k * vpc`. Then

```
E[actual]   = k * vpc * mu
Var[actual] = k * vpc * sigma^2 * (N - k * vpc) / (N - 1)
```

The estimate is `committee_weight * k`. Under unit weights the two match almost
exactly, which is why the 2023 work never looked at this branch. Under MaxEB they
do not.

## A constraint on any fix

`compute_honest_ffg_support_for_current_target` does

```python
ffg_weight_till_now = estimate_committee_weight_between_slots(
    total_active_balance, compute_start_slot_at_epoch(current_epoch), current_slot - 1
)
remaining_ffg_weight = total_active_balance - ffg_weight_till_now
```

That subtraction needs the estimate to stay at or below `total_active_balance`.
The range always sits inside the current epoch, so it always takes the same-epoch
branch, and `k` reaches 31. At `k = 31` the estimate is already `31/32` of the
total, which leaves about 3.2% of headroom.

`src/same_epoch_branch.py` reports the required margin next to that headroom. So
you can compare any proposed padding against it.

## Idealisations

- Committee sizes are modelled as exactly `vpc`. Real committee sizes differ by at
  most one validator, from the integer division in `compute_committee`. That is
  about 0.005% of a committee and it is dwarfed by the effects measured here.
- `vpc = N // 32` biases the modelled mean low by around 1e-5 relative, four
  orders of magnitude below the margins under study. `src/test_branches.py`
  measures this and shows the bias vanishes with exact sample sizes.
- Consecutive epoch shuffles are treated as independent. They use different seeds,
  which is the intent of the design.
