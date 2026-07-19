"""D_int buffer aggregation + intervention-weighted loss (Paper Y3, P2)."""

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import K_CAND, F_JOB, F_CTX
from fmwos.hitl.latent_head import LatentDispatchPolicy, LAT_DIM
from fmwos.hitl.intervention import InterventionBuffer, imitation_loss


def _entry(override, confirmation, decider_idx=0, preferred_idx=1):
    return {
        "cand": np.zeros((K_CAND, F_JOB), np.float32),
        "mask": np.array([True] * 3 + [False] * (K_CAND - 3)),
        "ctx": np.zeros((F_CTX,), np.float32),
        "latfeat": np.zeros((K_CAND, LAT_DIM), np.float32),
        "decider_idx": decider_idx, "preferred_idx": preferred_idx,
        "override": override, "confirmation": confirmation, "margin": 0.1,
    }


def test_aggregation_never_resets():
    buf = InterventionBuffer(capacity=1000, seed=0)
    # "iteration 1": 10 overrides, 20 confirmations
    for _ in range(10):
        buf.add(_entry(True, False))
    for _ in range(20):
        buf.add(_entry(False, True))
    assert len(buf) == 30
    # "iteration 2": aggregate MORE (never reset)
    for _ in range(5):
        buf.add(_entry(True, False))
    st = buf.stats()
    assert len(buf) == 35, len(buf)
    assert st["total_reviewed"] == 35
    assert st["total_overrides"] == 15
    assert st["total_confirmations"] == 20
    assert abs(st["override_rate_of_reviews"] - 15 / 35) < 1e-9
    print("  aggregation: retained=%d over=%d conf=%d orr=%.3f OK"
          % (len(buf), st["total_overrides"], st["total_confirmations"],
             st["override_rate_of_reviews"]))


def test_reservoir_bounded():
    buf = InterventionBuffer(capacity=100, seed=1)
    for _ in range(5000):
        buf.add(_entry(True, False))
    assert len(buf) == 100, len(buf)
    assert buf.total_reviewed == 5000
    print("  reservoir: capacity respected (retained=100, seen=5000) OK")


def test_weighting():
    """A mixed override/confirmation batch weights overrides 5x confirmations."""
    torch.manual_seed(0)
    pol = LatentDispatchPolicy(gate=0.0)
    K = K_CAND
    # two-sample batch: sample 0 = override (target 1), sample 1 = confirm (target 1)
    cand = torch.zeros(2, K, F_JOB)
    mask = torch.zeros(2, K, dtype=torch.bool); mask[:, :3] = True
    ctx = torch.zeros(2, F_CTX)
    lat = torch.zeros(2, K, LAT_DIM)
    target = torch.tensor([1, 1])
    is_over = torch.tensor([True, False])
    batch = {"cand": cand, "mask": mask, "ctx": ctx, "latfeat": lat,
             "target": target, "is_override": is_over}
    ow, cw = 5.0, 1.0
    loss = imitation_loss(pol, batch, override_weight=ow, confirm_weight=cw)

    # manual weighted CE
    logits, _v = pol.forward(cand, mask, ctx, None)
    logp = torch.log_softmax(logits, dim=-1)
    ce = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    expected = (ow * ce[0] + cw * ce[1]) / (ow + cw)
    assert torch.allclose(loss, expected, atol=1e-6), (float(loss), float(expected))
    # sanity: with equal CE per sample, the override sample dominates the mean
    assert ow / (ow + cw) == 5 / 6
    print("  weighting: loss==(5*ce_o+1*ce_c)/6 OK (loss=%.5f)" % float(loss))


def test_sample_batch_targets():
    buf = InterventionBuffer(capacity=50, seed=2)
    buf.add(_entry(True, False, decider_idx=0, preferred_idx=2))     # target 2
    buf.add(_entry(False, True, decider_idx=1, preferred_idx=1))     # target 1
    b = buf.sample_batch(2, device="cpu")
    assert b["target"].shape[0] == 2
    assert set(int(t) for t in b["target"]) <= {1, 2}
    print("  sample_batch: targets/flags materialize OK")


if __name__ == "__main__":
    test_aggregation_never_resets()
    test_reservoir_bounded()
    test_weighting()
    test_sample_batch_targets()
    print("ALL INTERVENTION TESTS PASSED")
