#!/usr/bin/env python
"""W2: merge per-seed / per-variant ladder shards into one file.

The ladder is run as one single-threaded process per seed (and per robustness
row) so it fits the 10-core allocation without oversubscribing a shared box.
Each shard writes results/y3_w2/<tag>_s<seed>.json holding {seed: {variant: ...}}.
This unions them by seed and by variant, refusing to merge shards whose resolved
configuration disagrees on anything except the variant list.

  python scripts/y3_w2_merge.py --shards ladder k64 --out ladder
"""

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import y3_w2_lib as L                                           # noqa: E402

_META = ("_config", "_testset_counts", "_calibset_counts", "_recorded_reference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=["ladder"])
    ap.add_argument("--out", default="ladder")
    a = ap.parse_args()

    merged, seen = {}, []
    for pre in a.shards:
        for path in sorted(glob.glob(os.path.join(L.OUT, "%s_s*.json" % pre))):
            if path.endswith("_summary.json"):
                continue
            shard = json.load(open(path))
            seen.append(os.path.basename(path))
            for seed, rows in shard.items():
                dst = merged.setdefault(seed, {})
                for k, v in rows.items():
                    if k in _META:
                        dst.setdefault(k, v)
                        continue
                    if k in dst:
                        raise SystemExit("variant %s for seed %s appears in two "
                                         "shards" % (k, seed))
                    dst[k] = v
                # the resolved config must agree everywhere except the variant
                c0, c1 = dst["_config"], rows["_config"]
                bad = sorted(x for x in set(c0) | set(c1)
                             if x != "variant" and c0.get(x) != c1.get(x))
                if bad:
                    raise SystemExit("shard %s config disagrees on %r" % (path, bad))

    if not merged:
        raise SystemExit("no shards matched %r" % a.shards)
    out = os.path.join(L.OUT, "%s.json" % a.out)
    with open(out + ".tmp", "w") as fh:
        json.dump(merged, fh, indent=1)
    os.replace(out + ".tmp", out)
    for seed in sorted(merged, key=int):
        vs = [k for k in merged[seed] if not k.startswith("_")]
        print("seed %s: %s" % (seed, ", ".join(sorted(vs))))
    print("\nmerged %d shards (%s) -> %s" % (len(seen), ", ".join(seen), out))


if __name__ == "__main__":
    sys.exit(main())
