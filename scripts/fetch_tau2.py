"""Download the tau2-bench airline, retail and telecom trajectories from Sierra's public S3 bucket."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
BUCKET = "https://sierra-tau-bench-public.s3.amazonaws.com"
DOMAINS = ("airline", "retail", "telecom")

TRIALS_RE = re.compile(r"_(" + "|".join(DOMAINS) + r")_.*\d+trials?\.json$")
RESULTS_RE = re.compile(r"/(" + "|".join(DOMAINS) + r")_results\.json$")


def enumerate_bucket() -> list[tuple[str, int]]:
    token, keys, pages = None, [], 0
    while True:
        url = f"{BUCKET}/?list-type=2&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url).read())
        for entry in root.findall("s3:Contents", NS):
            keys.append((entry.find("s3:Key", NS).text, int(entry.find("s3:Size", NS).text)))
        pages += 1
        print(f"  page {pages:>3}  keys {len(keys):>6}", flush=True)
        if root.find("s3:IsTruncated", NS).text != "true":
            return keys
        token = root.find("s3:NextContinuationToken", NS).text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/tau2"))
    parser.add_argument("--refresh-keys", action="store_true", help="re-enumerate instead of using the cache")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "_bucket_keys.json"
    if cache.exists() and not args.refresh_keys:
        keys = [tuple(k) for k in json.loads(cache.read_text(encoding="utf-8"))]
        print(f"loaded cached key list: {len(keys)} objects", flush=True)
    else:
        print("enumerating bucket", flush=True)
        keys = enumerate_bucket()
        cache.write_text(json.dumps(keys), encoding="utf-8")
        print(f"saved key list to {cache} ({len(keys)} objects)", flush=True)

    wanted = [
        (key, size)
        for key, size in keys
        if "/trajectories/" in key and (TRIALS_RE.search(key) or RESULTS_RE.search(key))
    ]
    print(f"\nmatched {len(wanted)} files, {sum(s for _, s in wanted) / 1e9:.2f} GB\n", flush=True)

    for i, (key, size) in enumerate(sorted(wanted), 1):
        submission, name = key.split("/")[1], key.rsplit("/", 1)[-1]
        destination = args.out / submission / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == size:
            print(f"[{i:>2}/{len(wanted)}] cached   {size / 1e6:>8.1f} MB  {submission}/{name}", flush=True)
            continue
        print(f"[{i:>2}/{len(wanted)}] fetching {size / 1e6:>8.1f} MB  {submission}/{name}", flush=True)
        urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", destination)
        # A short read leaves a file that parses as JSON but silently drops simulations, so the
        # size is compared against the listing rather than trusted.
        written = destination.stat().st_size
        if written != size:
            sys.exit(f"{key}: expected {size} bytes, wrote {written}")

    print(f"\ncomplete: {len(wanted)} files in {args.out}", flush=True)


if __name__ == "__main__":
    main()
