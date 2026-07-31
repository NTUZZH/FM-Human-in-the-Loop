#!/usr/bin/env python
"""W2 orientation probe (throwaway diagnostics, writes nothing but a JSON).

Answers three questions before any W2 code is written:
  1. Does ``Supervisor.review``'s log entry expose the feasible set Q? (No -- see
     the report; this script measures what a wrapping subclass recovers.)
  2. What is the queue-size distribution at REVIEWED decisions at the headline
     cell? (Sets the choice-set width K for the conditional logit.)
  3. How many overrides / confirmations does one headline-cell DAgger run make?

Read-only w.r.t. the repo: imports the shipped modules, edits nothing.
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

torch.set_num_threads(4)

from fmwos.env import DispatchEnv                                # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402
from fmwos.hitl import augmented_rule as AR                      # noqa: E402
from fmwos.hitl import deciders as dec                           # noqa: E402
from fmwos.hitl.supervisor import Supervisor                     # noqa: E402
from fmwos.hitl.choice_estimator import QueueLoggingSupervisor   # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_w2")


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def main():
    import glob
    files = sorted(glob.glob(os.path.join(
        _INST, "c09", "storm2", "w80", "c09_storm2_w80_u100_*.json")))
    train = [_load(p) for p in files[:16]]
    overlay = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL",
                                          master_seed=12345,
                                          channel="full_class_shift"))
    # One RULE+SUP episode per training instance at the headline cell.
    base_keys, qsz, n_rev, n_over, n_conf = None, [], 0, 0, 0
    for inst in train[:4]:
        applied = overlay.apply(inst)
        sup = QueueLoggingSupervisor(overlay, inst, rho=0.25, epsilon=0.0,
                                     theta=1.0, mechanism="targeted", seed=301,
                                     applied=applied)
        _s, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=301)
        for e in log:
            if not e.get("reviewed"):
                continue
            if base_keys is None:
                base_keys = sorted(e.keys())
            qsz.append(len(e["cand_ids"]))
            n_rev += 1
            n_over += bool(e["override"])
            n_conf += bool(e["confirmation"])
    q = np.asarray(qsz)
    out = {
        "log_entry_keys_with_wrapper": base_keys,
        "n_instances_probed": 4,
        "n_reviewed": int(n_rev), "n_overrides": int(n_over),
        "n_confirmations": int(n_conf),
        "queue_size_at_reviewed": {
            "mean": float(q.mean()), "median": float(np.median(q)),
            "p90": float(np.percentile(q, 90)), "p99": float(np.percentile(q, 99)),
            "max": int(q.max()), "min": int(q.min()),
            "frac_gt_64": float((q > 64).mean()),
        },
    }
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "probe.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
