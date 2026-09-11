# FCR committee weight estimate: safety margin after MaxEB

## The question

The Fast Confirmation Rule estimates the weight of the committees in a slot range.
The estimate is unbiased, so it covers the true weight only about half the time.
The spec inflates it by 0.5% to turn it into a high-probability upper bound:

```python
# specs/phase0/fast-confirmation.md
def adjust_committee_weight_estimate_to_ensure_safety(estimate: Gwei) -> Gwei:
    ceil = (estimate + 999) // 1000
    return ceil * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR)  # factor = 5
```

That 0.5% comes from a [calibration notebook](https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20)
written in July 2023. The notebook gives every validator the same weight:

```python
validators_per_committee = total_validators // 32
nbad = x * validators_per_committee
```

EIP-7251 raised the maximum effective balance to 2048 ETH. Effective balances are
no longer equal, so committee **weight** now varies far more than committee
**size**. The notebook has two revisions, both from 2023-07-22, and no later ones.
`COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR` appears in exactly one commit in
consensus-specs, the one that merged the rule in April 2026.

**Is 0.5% still enough?**

## The answer

No. The estimate misses its own 99.99% calibration target. The table below uses
the mainnet validator set of 2026-09-10. On the padded branch the shortfall
reaches 10x. The branch with no padding needs 7.25%.

| Branch | Worst required margin | Spec padding | Short by |
|---|---|---|---|
| crosses an epoch boundary | **5.09%** (at `x=1, y=1`) | 0.5% | 10.2x |
| sits inside one epoch | **7.25%** (at `k=1`) | none | — |

The median over all reachable cross-boundary splits is 0.637%. That is above 0.5%
too, so the shortfall is not confined to a rare corner.

The largest gap sits on the most ordinary path. `compute_safety_threshold` calls
the estimator with `(parent_slot + 1, current_slot - 1)`. For a block with no
missed slot before it, evaluated one slot after it arrives, that range is a single
slot inside one epoch. It needs 7.25% and gets nothing.

```
                                            situation                           slots         branch    range  required  padded
         safety threshold, no missed slots, mid-epoch parent 40, block 41, current 42     same_epoch      k=1     7.252%    0.0%
                  safety threshold, block 2 slots old parent 40, block 41, current 43     same_epoch      k=2     5.045%    0.0%
    safety threshold, 4 missed slots before the block parent 36, block 41, current 42     same_epoch      k=5     3.027%    0.0%
 safety threshold, block at the last slot of an epoch parent 30, block 31, current 33 cross_boundary x=1, y=1     5.091%    0.5%
safety threshold, block at the last slot, 3 slots old parent 30, block 31, current 35 cross_boundary x=3, y=1     3.497%    0.5%
   safety threshold, missed slots across the boundary parent 26, block 33, current 35 cross_boundary x=3, y=5     2.354%    0.5%
           honest FFG support, 8 slots into the epoch        range 32..39, current 40     same_epoch      k=8     2.256%    0.0%
                honest FFG support, late in the epoch        range 32..62, current 63     same_epoch     k=31     0.234%    0.0%
```

## What the margin is today

The section above asks how much padding the estimate needs. This one asks the
reverse. The spec is unchanged, so what does the current padding deliver?

The target is the one the 2023 notebook set. The padded estimate must come out too
low at most 1 time in 10,000.

| branch | padding | comes out too low |
|---|---|---|
| same-epoch, any `k` | none | **~50% of the time** |
| cross-boundary, `x=1 y=1` | 0.5% | **36%** |
| cross-boundary, `x=3 y=5` | 0.5% | 21% |
| cross-boundary, `x=8 y=8` | 0.5% | 11% |
| cross-boundary, `x=16 y=16` | 0.5% | 0.7% |

The same-epoch branch has no padding, so it is a coin flip at every range length.
It was always a coin flip. Under equal weights that cost nothing, because the
estimate was almost exact. The misses are percent-sized now.

On the cross-boundary branch, 0.5% is only **0.37 standard deviations** of the true
committee weight at `x=1 y=1`. The target needs 3.72. Long ranges still land near
target, because they sample most of the validator set and the variance collapses.

So the designed margin is effectively gone.

### How large the misses are

Frequency alone overstates the problem. The size matters, and the misses are small.

| | same-epoch `k=1` | cross-boundary `x=1 y=1` |
|---|---|---|
| comes out too low | 49.8% | 35.5% |
| median miss | 1.33% | 0.76% |
| too low by more than 2% | 15.4% | 3.4% |
| too low by more than 5% | 0.6% | 0.006% |
| worst in 200,000 draws | 9.59% | 6.12% |

Read those two tables together. The estimate comes out too low often, and it comes
out too low by about 1%.

### What this does and does not mean

**The engineered buffer is gone.** The design allowed one miss beyond the padding
in 10,000. Today it misses one time in two or three.

**The confirmations are not shown to be wrong.** A 1% error in the estimate moves
the safety threshold by about 1%. Whether that flips a confirmation depends on the
slack between a block's real support and the threshold. A block that FCR confirms
normally carries a large attesting supermajority, well clear of the bar. This work
does not measure how often it sits close.

FCR runs with roughly no safety buffer on this term, where the design gave it a
large one. That is a regression. It is not evidence that a confirmed block was
wrong.

Both tables come from `src/coverage_today.py`, and exact sampling confirms them.

## Why

Consolidation is concentrated, not widespread. On the snapshot:

| | |
|---|---|
| active validators | 909,682 |
| total active balance | 43,036,901 ETH |
| mean effective balance | 47.31 ETH |
| sd of effective balance | 158.03 ETH |
| coefficient of variation | **3.34** |
| distinct effective balances | 1,414 |
| validators still at exactly 32 ETH | 98.45% |
| validators above 32 ETH | 1.55% of validators, **33.38% of stake** |

A committee holds about 28,427 validators. About 440 of them are above 32 ETH.
Those 440 carry a third of the weight of the committee. The count of large
validators in a given committee fluctuates by about 5%. That alone moves the
weight of the committee by percent. The spec pads half a percent.

Under equal weights this effect does not exist. The count of validators in a
committee is fixed by the shuffle, so the weight is fixed too. That is why the
2023 model saw a 0.3% requirement at 600,000 validators, and why it no longer
applies.

## What is in this repository

```
src/model.py                   the model: population, estimate, moments, exact sampler
src/test_branches.py           checks on the premises (run this first)
src/reproduce_gist.py          reproduces the 2023 published table
src/maxeb_sweep.py             parametric sweep over consolidation, fixed total stake
src/same_epoch_branch.py       the unpadded branch, and the headroom constraint
src/fetch_mainnet_balances.py  collects the real effective balance histogram
src/analyse_mainnet.py         the headline result, on real data
METHOD.md                      the model and its derivation
results/                       CSV output and run logs
```

## Validation

**The code reproduces the 2023 table.** Six of the seven published rows come
out exactly, and the seventh differs by one grid step:

| validators | published | this code |
|---|---|---|
| 9,375 | 1.9% | 1.9% |
| 18,750 | 1.3% | 1.4% |
| 37,500 | 1.0% | 1.0% |
| 75,000 | 0.7% | 0.7% |
| 150,000 | 0.5% | 0.5% |
| 300,000 | 0.4% | 0.4% |
| 600,000 | 0.3% | 0.3% |

The 18,750 row sits on a rounding boundary. The notebook ran 100,000 trials
against a 99.99% target. That leaves about ten exceedances to place the quantile.
So the published value carries visible noise there.

**The closed form agrees with exact sampling.** Multivariate-hypergeometric draws
confirm every headline number:

```
cross-boundary branch:
  x= 1 y= 1  analytic=  5.091%  sampled=  5.266%  P(covers)=0.505
  x= 3 y= 5  analytic=  2.354%  sampled=  2.318%  P(covers)=0.501
same-epoch branch:
  k= 1        analytic=  7.252%  sampled=  7.471%  P(covers)=0.508
  k= 4        analytic=  3.446%  sampled=  3.417%  P(covers)=0.500
```

`P(covers)` is the fraction of trials where the unadjusted estimate already
covered the actual weight. It lands at 0.50 on real data, which confirms the
estimator is unbiased and needs the inflation to become an upper bound.

The closed form runs slightly **low** against sampling on the MaxEB populations,
because the true upper tail is right-skewed. So the sampled figures are the
conservative ones, and they are the ones quoted above.

The approximation errs the other way on equal-weight populations with few
validators. At 9,375 validators the closed form asks for 1.88%. Sampling asks for
1.25%. The actual weight there is a validator count, so it is an integer with a
standard deviation near 3. A 99.99% quantile sits 3.7 standard deviations out.
Hypergeometric tails are much lighter than normal that far out, so the normal
quantile overshoots.

This affects only the reproduction of the 2023 table, where it is conservative.
It does not affect the MaxEB results. There the balances are heterogeneous and the
distribution is smooth.

**The premises are confirmed, not assumed.** `src/test_branches.py` brute-forces
every slot range over four epochs. It confirms three things. The pro-rata branch
reaches exactly the 961 splits with `x` and `y` in 1..31. The same-epoch branch
reaches `k` in 1..31. The estimate equals the mean actual weight.

## Consolidation sweep

Holding total stake at 19.2M ETH and varying how much of it is consolidated
(`src/maxeb_sweep.py`). "count" is the variance the 2023 model captured. "spread"
is what heterogeneous balances add.

| consolidated stake | validators | cv | required | count part | spread part |
|---|---|---|---|---|---|
| 0% | 600,000 | 0.00 | 0.24% | 0.24% | 0.00% |
| 5% | 570,469 | 1.72 | 3.31% | 0.24% | 3.30% |
| 20% | 481,875 | 3.15 | 6.60% | 0.26% | 6.59% |
| 50% | 304,688 | 3.94 | 10.37% | 0.33% | 10.36% |
| 80% | 127,500 | 3.15 | 12.82% | 0.51% | 12.81% |
| 100% | 9,375 | 0.00 | 1.88% | 1.88% | 0.00% |

The spread term dominates as soon as consolidation starts. Five percent of stake
in large validators is enough to push the requirement past 3%.

The two ends of the sweep are both equal-weight populations, so their spread term
is zero. The 100% row reproduces the notebook's 9,375-validator figure of 1.9%,
because 19.2M ETH at 2048 ETH each is 9,375 validators.

## What this does and does not show

**It shows** that `estimate_committee_weight_between_slots` no longer meets its
99.99% coverage target. On the padded branch it falls short by a factor of about
10, on real mainnet data.

**It does not show** that a confirmed block gets reorged. That needs the
protocol-level analysis in the confirmation rule paper, with an adversary and a
network model. This work stops at the arithmetic.

The link between the two runs through `compute_safety_threshold`:

```python
threshold = (maximum_support + proposer_score + 2 * adversarial_weight - support_discount) // 2
```

`maximum_support` is the estimate. `adversarial_weight` derives from the same
estimator, and enters with a factor of two. Under-estimating both lowers the
threshold, so `is_one_confirmed` confirms blocks at a lower bar than the rule
intends. The erosion is roughly proportional to the under-estimate.

Note also that one estimate violation is not one reorg. The rule runs every slot,
and a 99.99% per-evaluation target already permits several violations a day at
7,200 slots. The finding is that the designed margin has eroded, not that a
failure has occurred.

## A fix fits, on the branch that has none

Padding the same-epoch branch is possible. `compute_honest_ffg_support_for_current_target`
subtracts the output of that branch from `total_active_balance`, so the estimate must
not pass the total. At `k = 31`, where the headroom is tightest, the branch needs
0.234% and 3.226% is available. See `src/same_epoch_branch.py`.

The cross-boundary branch has no such constraint, so its factor can rise.

Choosing new values is a spec question, not one this repository settles. The
requirement moves with the consolidation level, so any new constant needs a
target consolidation scenario, not just today's snapshot.

## Reproducing

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python src/test_branches.py        # premises
.venv/bin/python src/reproduce_gist.py       # 2023 table
.venv/bin/python src/maxeb_sweep.py          # consolidation sweep
.venv/bin/python src/same_epoch_branch.py    # unpadded branch

.venv/bin/python src/fetch_mainnet_balances.py   # about 9 minutes, 473 requests
.venv/bin/python src/analyse_mainnet.py          # the headline result
.venv/bin/python src/coverage_today.py            # what the padding delivers today
```

`fetch_mainnet_balances.py` reads a public beacon API and keeps only the
effective balance histogram. Point it at another node with `--url`. This
repository holds the snapshot used here, at
`results/mainnet_effective_balances.json`. So the analysis reruns with no network
access.

## Limitations

- The mainnet figures are one snapshot, taken 2026-09-10. Consolidation continues,
  so the requirement moves.
- Committee sizes are modelled as exactly `N // 32`. Real sizes differ by at most
  one validator, which is about 0.005% of a committee.
- Consecutive epoch shuffles are treated as independent.
- The 99.99% target is inherited from the 2023 notebook. Whether that is the right
  target is a separate question.
- `METHOD.md` lists the derivation and the idealisations in full.

## References

- [`specs/phase0/fast-confirmation.md`](https://github.com/ethereum/consensus-specs/blob/dev/specs/phase0/fast-confirmation.md)
- [consensus-specs PR #4747](https://github.com/ethereum/consensus-specs/pull/4747), which merged the rule
- [The 2023 calibration notebook](https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20)
- [EIP-7251](https://eips.ethereum.org/EIPS/eip-7251)
