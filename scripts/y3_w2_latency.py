#!/usr/bin/env python
"""W2: per-decision deployment cost of the queue-conditioned decider.

The manuscript quotes a per-decision cost for the augmented rule. A
queue-conditioned estimator cannot use a static per-instance hat_s map, so it
runs one forward pass per dispatch decision; this measures what that costs.

The box is shared, so the ABSOLUTE milliseconds are an upper bound and are not a
measurement. The RATIO is the reportable quantity: both deciders are timed in
the same process, on the same instances, in the same interleaved order, so
whatever contention exists applies to both. What else was running is recorded in
the output file.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import y3_w2_lib as L                                           # noqa: E402
from fmwos.env import DispatchEnv                               # noqa: E402
from fmwos.hitl import augmented_rule as AR                     # noqa: E402
from fmwos.hitl import choice_estimator as CE                   # noqa: E402


def _time_decider(make, instances, reps=1):
    """Total wall time and decision count of running a decider to completion."""
    n, secs = 0, 0.0
    for _ in range(reps):
        for inst in instances:
            d = make(inst)
            cnt = {"n": 0}

            def _counted(queue, t, rng, _d=d, _c=cnt):
                _c["n"] += 1
                return _d(queue, t, rng)

            t0 = time.perf_counter()
            DispatchEnv(inst).run_supervised(_counted, supervisor=None,
                                             method="lat", seed=301)
            secs += time.perf_counter() - t0
            n += cnt["n"]
    return secs, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-instances", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cell = L.build_cell(seed=301)
    insts = cell["eval"][:a.n_instances]
    torch.manual_seed(301)
    per_order = CE.ChoiceModel(use_queue=False)
    queued = CE.ChoiceModel(use_queue=True)
    tables = {i["meta"]["id"]: CE.instance_tables(i) for i in insts}

    def mk_plain(inst):
        return AR.augmented_atc_decider(per_order.est.core, inst,
                                        channel=cell["channel"])

    def mk_queue(inst):
        return CE.queue_conditioned_atc_decider(
            queued, inst, channel=cell["channel"],
            table=tables[inst["meta"]["id"]])

    # interleave so any drift in machine load hits both equally
    out = {}
    for name, mk in (("per_order", mk_plain), ("queue", mk_queue),
                     ("per_order2", mk_plain), ("queue2", mk_queue)):
        s, n = _time_decider(mk, insts, reps=a.reps)
        out[name] = {"secs": s, "n_decisions": n, "ms_per_decision": 1000.0 * s / n}
        print("%-11s %8.2f s over %6d decisions -> %.4f ms/decision"
              % (name, s, n, 1000.0 * s / n))

    po = min(out["per_order"]["ms_per_decision"], out["per_order2"]["ms_per_decision"])
    qu = min(out["queue"]["ms_per_decision"], out["queue2"]["ms_per_decision"])
    try:
        load = open("/proc/loadavg").read().strip()
        top = subprocess.run(
            ["ps", "-eo", "pcpu,args", "--sort=-pcpu"],
            capture_output=True, text=True).stdout.splitlines()[1:4]
    except Exception:
        load, top = "?", []
    out["summary"] = {
        "ms_per_decision_per_order_best": po,
        "ms_per_decision_queue_best": qu,
        "ratio_queue_over_per_order": qu / po,
        "n_instances": len(insts), "reps": a.reps, "threads": a.threads,
        "loadavg_at_end": load, "top_processes": top,
        "caveat": "shared box; absolute ms is an UPPER BOUND, not a measurement. "
                  "Both deciders were timed in the same process, interleaved, so "
                  "the RATIO is the reportable quantity. Estimated from the "
                  "FASTEST of two passes each."}
    with open(os.path.join(L.OUT, "latency.json") + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(os.path.join(L.OUT, "latency.json") + ".tmp",
               os.path.join(L.OUT, "latency.json"))
    print("\nbest-of-two: per-order %.4f ms, queue-conditioned %.4f ms, "
          "ratio %.1fx" % (po, qu, qu / po))
    print("loadavg %s" % load)


if __name__ == "__main__":
    sys.exit(main())
