"""M1 latent-head gating bit-exactness + shift-head plumbing (Paper Y3, P2).

gate == 0 must reproduce the Y1 forward pass to the bit. This is the E0 anchor
extended to the auxiliary head; a load of a Y1 checkpoint into the M1 policy with
the head gated off yields IDENTICAL logits and values.
"""

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import K_CAND, F_JOB, F_CTX
from fmwos.policy import DispatchPolicy
from fmwos.hitl.latent_head import LatentDispatchPolicy, LAT_DIM

_CKPT = os.path.join(_ROOT, "results", "p3_train", "seed301", "best.pt")


def _rand_batch(b=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    cand = torch.randn(b, K_CAND, F_JOB, generator=g)
    mask = torch.zeros(b, K_CAND, dtype=torch.bool)
    for i in range(b):
        n = int(torch.randint(1, K_CAND + 1, (1,), generator=g).item())
        mask[i, :n] = True
    ctx = torch.randn(b, F_CTX, generator=g)
    lat = torch.randn(b, K_CAND, LAT_DIM, generator=g)
    return cand, mask, ctx, lat


def test_gate_zero_bit_exact():
    if not os.path.exists(_CKPT):
        print("  checkpoint missing -- comparing two same-seeded fresh models")
        torch.manual_seed(7)
        y1 = DispatchPolicy()
        torch.manual_seed(7)
        m1 = LatentDispatchPolicy(gate=0.0)
        # copy backbone weights so both share the base path exactly
        sd = {k: v for k, v in y1.state_dict().items()}
        m1.load_state_dict(sd, strict=False)
    else:
        y1 = DispatchPolicy.load(_CKPT, map_location="cpu")
        m1 = LatentDispatchPolicy.from_y1_checkpoint(_CKPT, gate=0.0)
    y1.eval(); m1.eval()
    cand, mask, ctx, lat = _rand_batch(seed=1)
    with torch.no_grad():
        lg1, v1 = y1(cand, mask, ctx)
        # gate=0: with AND without latfeat must both equal the Y1 forward exactly
        lg2, v2 = m1(cand, mask, ctx, lat)
        lg3, v3 = m1(cand, mask, ctx, None)
    assert torch.equal(lg1, lg2), (lg1 - lg2).abs().max().item()
    assert torch.equal(v1, v2), (v1 - v2).abs().max().item()
    assert torch.equal(lg1, lg3) and torch.equal(v1, v3)
    print("  gate=0 bit-exact: logits & value IDENTICAL to Y1 (with/without latfeat) OK")


def test_gate_open_changes_logits():
    torch.manual_seed(3)
    m1 = LatentDispatchPolicy(gate=1.0)
    # make the head produce a non-trivial output
    for p in m1.shift_head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    m1.eval()
    cand, mask, ctx, lat = _rand_batch(seed=2)
    with torch.no_grad():
        lg_on, _ = m1(cand, mask, ctx, lat)
        m1.set_gate(0.0)
        lg_off, _ = m1(cand, mask, ctx, lat)
    # opening the gate must move the (valid) logits
    diff = (lg_on - lg_off)[mask].abs().max().item()
    assert diff > 1e-4, diff
    print("  gate>0 shifts logits by max |d|=%.4f OK" % diff)


def test_checkpoint_roundtrip():
    import tempfile
    m = LatentDispatchPolicy(gate=0.7)
    for p in m.shift_head.parameters():
        torch.nn.init.normal_(p, std=0.3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m1.pt")
        m.save(path)
        m2 = LatentDispatchPolicy.load(path)
    cand, mask, ctx, lat = _rand_batch(seed=5)
    with torch.no_grad():
        a, va = m(cand, mask, ctx, lat)
        b, vb = m2(cand, mask, ctx, lat)
    assert torch.equal(a, b) and torch.equal(va, vb)
    assert abs(float(m2.gate) - 0.7) < 1e-6
    print("  checkpoint save/load roundtrip exact (gate preserved) OK")


if __name__ == "__main__":
    test_gate_zero_bit_exact()
    test_gate_open_changes_logits()
    test_checkpoint_roundtrip()
    print("ALL LATENT-HEAD TESTS PASSED")
