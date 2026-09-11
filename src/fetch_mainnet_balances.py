"""Collect the mainnet effective balance histogram from a public beacon API.

The analysis needs two things: the number of active validators, and the histogram
of their effective balances. This script fetches both and writes a small JSON
file. It does not keep the raw validator records.

Committee membership follows `is_active_at`, which reads activation and exit
epochs and ignores balance. Slashed validators stay in committees until they
exit, so every status that starts with "active_" counts.

Run: .venv/bin/python src/fetch_mainnet_balances.py [--url URL] [--batch N]
Output: results/mainnet_effective_balances.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_URL = "https://ethereum-beacon-api.publicnode.com"
GWEI_PER_ETH = 10**9


def post_validators(base_url: str, ids: list[int], retries: int = 4) -> list[dict]:
    """POST /eth/v1/beacon/states/head/validators for a batch of indices."""
    url = f"{base_url}/eth/v1/beacon/states/head/validators"
    body = json.dumps({"ids": [str(i) for i in ids]}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            # The default urllib agent gets a 403 from the CDN in front of the API.
            "User-Agent": "fcr-committee-weight-maxeb/1.0",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())["data"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} after {error} (sleep {wait}s)")
            time.sleep(wait)
    return []


def find_validator_count(base_url: str) -> int:
    """Find the number of validator records by probing indices."""
    low, high = 0, 1
    while post_validators(base_url, [high]):
        low = high
        high *= 2
        if high > 1 << 26:
            raise RuntimeError("validator index probe ran away")
    while high - low > 1:
        mid = (low + high) // 2
        if post_validators(base_url, [mid]):
            low = mid
        else:
            high = mid
    return low + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    print(f"probing validator count at {args.url}")
    total = find_validator_count(args.url)
    print(f"validator records: {total:,}")

    active = Counter()
    status_counts = Counter()
    fetched = 0
    start = time.time()

    for begin in range(0, total, args.batch):
        ids = list(range(begin, min(begin + args.batch, total)))
        records = post_validators(args.url, ids)
        fetched += len(records)
        for record in records:
            status = record["status"]
            status_counts[status] += 1
            if status.startswith("active_"):
                active[int(record["validator"]["effective_balance"])] += 1
        if begin % (args.batch * 20) == 0:
            elapsed = time.time() - start
            print(
                f"  {fetched:>9,}/{total:,} records  "
                f"{len(active)} distinct active balances  {elapsed:.0f}s"
            )
        time.sleep(args.sleep)

    num_active = sum(active.values())
    total_balance = sum(b * c for b, c in active.items())
    print(f"\nactive validators: {num_active:,}")
    print(f"total active balance: {total_balance / GWEI_PER_ETH:,.0f} ETH")
    print(f"distinct effective balances: {len(active)}")

    out = Path(__file__).parent.parent / "results" / "mainnet_effective_balances.json"
    out.write_text(
        json.dumps(
            {
                "source_url": args.url,
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "validator_records": total,
                "active_validators": num_active,
                "total_active_balance_gwei": total_balance,
                "status_counts": dict(sorted(status_counts.items())),
                "effective_balance_gwei_to_count": {
                    str(b): c for b, c in sorted(active.items())
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
