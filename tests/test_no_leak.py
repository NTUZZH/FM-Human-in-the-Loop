"""No-leak guarantee (Paper Y3, P2 RED LINE).

The latent must never touch the policy's observations or the RL reward. Two
checks:

* RUNTIME: with an overlay + supervisor attached at rho=0 (nothing reviewed,
  nothing overridden), the observation vectors AND the reward sequence of a
  fixed greedy decider are byte-identical to the same rollout with NO overlay
  attached. Any leak into obs/reward would perturb one of the two streams.
* GREP-LEVEL: the env's observation- and reward-building code contains no
  reference to any latent quantity (overlay, shift, w_star, c_star, xi, f_of).
"""

import glob
import os
import re
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv
from fmwos.policy import DispatchPolicy
from fmwos.hitl.overlay import Overlay, OverlayParams
from fmwos.hitl.supervisor import Supervisor

_INST = os.path.join(_ROOT, "data", "processed", "instances")


def _an_instance():
    import json
    path = sorted(glob.glob(os.path.join(_INST, "c09", "replay", "150", "*.json")))[0]
    with open(path) as fh:
        return json.load(fh)


def _rollout(inst, policy, overlay=None):
    """Greedy rollout; returns (list of obs copies, list of rewards). If overlay
    is given, a rho=0 supervisor is consulted at every decision (and must never
    change the executed pick)."""
    env = DispatchEnv(inst, reward_mode="shaped")
    sup = None
    if overlay is not None:
        sup = Supervisor(overlay, inst, rho=0.0, epsilon=0.0, theta=1.0,
                         mechanism="targeted", seed=0)
    obs = env.reset()
    obs_seq, rew_seq = [], []
    done = env._done
    while not done:
        obs_seq.append((obs["cand"].copy(), obs["mask"].copy(), obs["ctx"].copy()))
        a, _lp, _v, _e = policy.act(obs, greedy=True, device="cpu")
        exec_a = a
        if sup is not None:
            cands = env._candidates
            executed, _entry = sup.review(cands[a], cands, env._cur_now, 0.5)
            exec_a = cands.index(executed)
            assert exec_a == a, "rho=0 supervisor changed the pick (should be inert)"
        obs, r, done, _i = env.step(exec_a)
        rew_seq.append(float(r))
    return obs_seq, rew_seq


def test_runtime_no_leak():
    inst = _an_instance()
    torch.manual_seed(0)
    policy = DispatchPolicy()
    policy.eval()
    overlay = Overlay(OverlayParams(beta=1.0, family="F-NL"))   # max recoverable latent

    o_a, r_a = _rollout(inst, policy, overlay=None)
    o_b, r_b = _rollout(inst, policy, overlay=overlay)

    assert len(o_a) == len(o_b) == len(r_a) == len(r_b), (len(o_a), len(o_b))
    for k, ((ca, ma, xa), (cb, mb, xb)) in enumerate(zip(o_a, o_b)):
        assert np.array_equal(ca, cb), "cand differ at step %d" % k
        assert np.array_equal(ma, mb), "mask differ at step %d" % k
        assert np.array_equal(xa, xb), "ctx differ at step %d" % k
    assert r_a == r_b, "reward sequence differs"
    print("  runtime: %d decisions, obs & reward byte-identical with/without "
          "overlay (rho=0, beta=1) OK" % len(o_a))


def test_grep_no_leak():
    src = os.path.join(_ROOT, "src", "fmwos", "env.py")
    with open(src) as fh:
        text = fh.read()
    # isolate the observation- and reward-building methods
    banned = ["overlay", "w_star", "c_star", "shift", "_xi", "latent",
              "preferred_pick", "true_objective"]
    # scan the whole env module: the Y1 env must not import or mention the latent
    lowered = text.lower()
    hits = [b for b in banned if b.lower() in lowered]
    assert not hits, "env.py references latent tokens: %s" % hits
    # and it must not import the hitl subpackage
    assert not re.search(r"from\s+\.hitl|import\s+hitl|fmwos\.hitl", text), \
        "env.py imports the hitl subpackage"
    print("  grep: env.py obs/reward code free of latent tokens & hitl imports OK")


if __name__ == "__main__":
    test_runtime_no_leak()
    test_grep_no_leak()
    print("ALL NO-LEAK TESTS PASSED")
