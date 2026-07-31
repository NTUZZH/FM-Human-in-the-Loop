"""E0 regression anchor (Paper Y3, Phase P1).

With the supervisor DISABLED the modified environment must be behaviourally
byte-identical to the unmodified Y1 dispatcher (proposal Sec.6, E0). This script
checks, fast (< 5 min) and self-contained, exiting NONZERO on any mismatch:

  (i)  RULE 3-way exactness. For all 4 verdict campuses (5, 9, 10, 12) and both
       sizes (150, 400), the first replay instance, all 6 rules: the freshly-run
       unmodified baseline ``pdrs.dispatch`` == the untouched ``env.run_policy``
       == the supervisor-path ``env.run_supervised(supervisor=None)``,
       assignment-for-assignment (wo -> tech, start, end exact), and all pass the
       independent validator.
  (ii) POLICY replay. The Y1 MLP checkpoint (seed 301) replayed greedily through
       the env.step path reproduces itself bit-for-bit across two runs and yields
       a feasible schedule (the env.step / driver code is unchanged by P1).
  (iii)SHAPED-REWARD telescoping via the existing test suite (tests/test_env.py):
       reused unchanged; its pass is required.

Run:  PYTHONPATH=src python scripts/y3_e0_anchor.py
"""

import glob
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import pdrs, validator                    # noqa: E402
from fmwos.env import DispatchEnv                     # noqa: E402
from fmwos.hitl import deciders as dec                # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_CKPT = os.path.join(_ROOT, "results", "p3_train", "seed301", "best.pt")

CAMPUSES = ["c05", "c09", "c10", "c12"]
SIZES = ["150", "400"]
RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]


def _first(campus, size):
    return sorted(glob.glob(os.path.join(_INST, campus, "replay", size, "*.json")))[0]


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _bywo(sched):
    return {a["wo"]: (a["tech"], a["start_bh"], a["end_bh"])
            for a in sched["assignments"]}


def _exact(a, b):
    ma, mb = _bywo(a), _bywo(b)
    if set(ma) != set(mb):
        return False
    return all(ma[w][0] == mb[w][0]
               and abs(ma[w][1] - mb[w][1]) <= 1e-12
               and abs(ma[w][2] - mb[w][2]) <= 1e-12 for w in ma)


def check_rules(failures):
    print("(i) RULE 3-way exactness: pdrs.dispatch == run_policy == run_supervised(None)")
    n = 0
    for campus in CAMPUSES:
        for size in SIZES:
            inst = _load(_first(campus, size))
            for rule in RULES:
                ref = pdrs.dispatch(inst, rule)
                rp = DispatchEnv(inst).run_policy(pdrs.get_rule(rule), method=rule)
                rs = dec.run_rule(DispatchEnv(inst), rule)
                ok = (_exact(ref, rp) and _exact(ref, rs)
                      and validator.validate(inst, rs)["feasible"]
                      and len(rs["assignments"]) == len(inst["work_orders"]))
                n += 1
                if not ok:
                    failures.append("RULE MISMATCH %s/%s %s" % (campus, size, rule))
    print("    checked %d (campus x size x rule) triples: %s"
          % (n, "OK" if not failures else "FAIL"))


def check_policy(failures):
    print("(ii) POLICY replay determinism + feasibility (Y1 MLP seed 301)")
    if not os.path.exists(_CKPT):
        print("    checkpoint %s missing -- SKIPPED (non-fatal)" % _CKPT)
        return
    import torch
    from fmwos.policy import DispatchPolicy
    torch.set_num_threads(1)
    pol = DispatchPolicy.load(_CKPT, map_location="cpu")
    pol.eval()

    def rollout(inst):
        env = DispatchEnv(inst)
        obs = env.reset()
        done = False
        while not done:
            a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
            obs, _r, done, _i = env.step(a)
        return env.to_schedule("rl301", seed=301)

    for campus in CAMPUSES:
        inst = _load(_first(campus, "150"))
        s1, s2 = rollout(inst), rollout(inst)
        ok = _exact(s1, s2) and validator.validate(inst, s1)["feasible"]
        if not ok:
            failures.append("POLICY MISMATCH %s/150" % campus)
    print("    replayed 4 campuses twice each: %s"
          % ("OK" if not failures else "FAIL"))


def check_latent_gate(failures):
    """(iv) M1 latent-head gate=0 bit-exactness (Paper Y3 P2, Sec.6).

    Loading the Y1 checkpoint into the M1 policy with the shift head gated OFF
    reproduces the Y1 forward pass (logits AND value) to the bit, with and
    without latent features supplied."""
    print("(iv) M1 latent-head gate=0 bit-exactness vs Y1 forward")
    if not os.path.exists(_CKPT):
        print("    checkpoint %s missing -- SKIPPED (non-fatal)" % _CKPT)
        return
    import torch
    from fmwos.policy import DispatchPolicy
    from fmwos.hitl.latent_head import LatentDispatchPolicy, LAT_DIM
    from fmwos.env import K_CAND, F_JOB, F_CTX
    torch.set_num_threads(1)
    y1 = DispatchPolicy.load(_CKPT, map_location="cpu").eval()
    m1 = LatentDispatchPolicy.from_y1_checkpoint(_CKPT, gate=0.0).eval()
    # Fair-M1 variant (in-network deadline head): gate=0 must ALSO be bit-exact.
    m1d = LatentDispatchPolicy.from_y1_checkpoint(_CKPT, gate=0.0,
                                                  deadline_head=True).eval()
    g = torch.Generator().manual_seed(0)
    ok = True
    for _ in range(4):
        b = 8
        cand = torch.randn(b, K_CAND, F_JOB, generator=g)
        mask = torch.zeros(b, K_CAND, dtype=torch.bool)
        for i in range(b):
            n = int(torch.randint(1, K_CAND + 1, (1,), generator=g).item())
            mask[i, :n] = True
        ctx = torch.randn(b, F_CTX, generator=g)
        lat = torch.randn(b, K_CAND, LAT_DIM, generator=g)
        with torch.no_grad():
            lg1, v1 = y1(cand, mask, ctx)
            lg2, v2 = m1(cand, mask, ctx, lat)      # gate=0: latfeat must be inert
            lg3, v3 = m1(cand, mask, ctx, None)
            lg4, v4 = m1d(cand, mask, ctx, lat)     # gate=0 + deadline head: inert
        ok = ok and torch.equal(lg1, lg2) and torch.equal(v1, v2) \
            and torch.equal(lg1, lg3) and torch.equal(v1, v3) \
            and torch.equal(lg1, lg4) and torch.equal(v1, v4)
    if not ok:
        failures.append("M1 gate=0 NOT bit-exact with Y1 forward")
    print("    4 random batches (base + deadline-head variants): %s"
          % ("OK" if ok else "FAIL"))


def check_telescoping(failures):
    print("(iii) SHAPED-REWARD telescoping via tests/test_env.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(_ROOT, "src")
    r = subprocess.run([sys.executable, os.path.join(_ROOT, "tests", "test_env.py")],
                       capture_output=True, text=True, env=env)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    print("    test_env.py exit=%d  last: %s" % (r.returncode, tail))
    if r.returncode != 0:
        failures.append("tests/test_env.py FAILED (exit %d)" % r.returncode)


def main():
    failures = []
    check_rules(failures)
    check_policy(failures)
    check_latent_gate(failures)
    check_telescoping(failures)
    print()
    if failures:
        print("E0 ANCHOR: FAIL")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("E0 ANCHOR: PASS")


if __name__ == "__main__":
    main()
