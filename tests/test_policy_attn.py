"""Unit sanity for the attention dispatch policy (src/fmwos/policy_attn.py).

Plain-python test runner (matches tests/test_curriculum.py style):

  (i)  permutation equivariance -- permuting the candidate order permutes the
       scores identically and leaves the value unchanged (to 1e-5);
  (ii) mask correctness -- padded slots get -inf logits and no gradient flows
       through their input features;
  (iii) parity smoke -- one env episode runs end-to-end via act()/step().

Run:  PYTHONPATH=src python3 tests/test_policy_attn.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fmwos.policy_attn import AttnDispatchPolicy, _NEG_INF
from fmwos.env import DispatchEnv, F_JOB, F_CTX, K_CAND


def _rand_obs_batch(rng, b, k, n_valid):
    """A batch with a KNOWN number of leading valid candidates per row."""
    cand = rng.standard_normal((b, k, F_JOB)).astype(np.float32)
    mask = np.zeros((b, k), dtype=bool)
    for i in range(b):
        mask[i, : n_valid[i]] = True
        cand[i, n_valid[i]:] = 0.0            # padded rows all-zero (as in env)
    ctx = rng.standard_normal((b, F_CTX)).astype(np.float32)
    return cand, mask, ctx


# --------------------------------------------------------------------------- #
def test_permutation_equivariance(failures):
    """Permuting valid candidates permutes logits identically; value invariant."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pol = AttnDispatchPolicy().eval()

    n_valid = 7
    cand = rng.standard_normal((1, K_CAND, F_JOB)).astype(np.float32)
    cand[0, n_valid:] = 0.0
    mask = np.zeros((1, K_CAND), dtype=bool)
    mask[0, :n_valid] = True
    ctx = rng.standard_normal((1, F_CTX)).astype(np.float32)

    ct = torch.as_tensor(cand)
    mt = torch.as_tensor(mask)
    xt = torch.as_tensor(ctx)
    with torch.no_grad():
        logits0, val0 = pol(ct, mt, xt)

    # Permute ONLY the valid candidate block; padding stays padding.
    perm = np.arange(K_CAND)
    perm[:n_valid] = rng.permutation(n_valid)
    ct2 = ct[:, perm, :].contiguous()
    with torch.no_grad():
        logits1, val1 = pol(ct2, mt, xt)

    # logits1[valid positions] should equal logits0 permuted the same way.
    l0 = logits0[0, :n_valid].numpy()
    l1 = logits1[0, :n_valid].numpy()
    max_logit_err = float(np.max(np.abs(l1 - l0[perm[:n_valid]])))
    val_err = float(abs(val1.item() - val0.item()))
    print("    equivariance: max_logit_err=%.2e  val_err=%.2e" %
          (max_logit_err, val_err))
    if max_logit_err > 1e-5:
        failures.append("permutation equivariance: logit err %.2e > 1e-5"
                        % max_logit_err)
    if val_err > 1e-5:
        failures.append("permutation invariance (value): err %.2e > 1e-5"
                        % val_err)


def test_mask_correctness(failures):
    """Padded logits are -inf; grad wrt padded input features is exactly 0."""
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    pol = AttnDispatchPolicy().train()
    b, n_valid = 3, [5, 1, 12]
    cand, mask, ctx = _rand_obs_batch(rng, b, K_CAND, n_valid)

    ct = torch.tensor(cand, requires_grad=True)
    mt = torch.as_tensor(mask)
    xt = torch.as_tensor(ctx)
    logits, value = pol(ct, mt, xt)

    # (a) padded logits == _NEG_INF exactly.
    padded_ok = True
    for i in range(b):
        pad = logits[i, n_valid[i]:]
        if not torch.all(pad == _NEG_INF):
            padded_ok = False
    if not padded_ok:
        failures.append("mask: some padded logit != _NEG_INF")

    # (b) masked softmax puts ~0 prob on padded slots.
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()
    pad_prob = 0.0
    for i in range(b):
        pad_prob = max(pad_prob, float(probs[i, n_valid[i]:].sum().item()))
    if pad_prob > 1e-6:
        failures.append("mask: padded prob mass %.2e > 1e-6" % pad_prob)

    # (c) gradient wrt padded candidate features is exactly 0 (no flow).
    loss = logits[mask].sum() + value.sum()
    loss.backward()
    grad = ct.grad.detach().numpy()
    max_pad_grad = 0.0
    for i in range(b):
        if n_valid[i] < K_CAND:
            max_pad_grad = max(max_pad_grad,
                               float(np.max(np.abs(grad[i, n_valid[i]:]))))
    print("    mask: padded_logits_neg_inf=%s  pad_prob=%.2e  max_pad_grad=%.2e"
          % (padded_ok, pad_prob, max_pad_grad))
    if max_pad_grad != 0.0:
        failures.append("mask: gradient leaked to padded features (max=%.2e)"
                        % max_pad_grad)


def test_value_invariant_to_padding(failures):
    """Value + valid logits are invariant to how many padded slots exist."""
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    pol = AttnDispatchPolicy().eval()
    n_valid = 6
    cand = rng.standard_normal((1, K_CAND, F_JOB)).astype(np.float32)
    cand[0, n_valid:] = 0.0
    mask = np.zeros((1, K_CAND), dtype=bool)
    mask[0, :n_valid] = True
    ctx = rng.standard_normal((1, F_CTX)).astype(np.float32)

    with torch.no_grad():
        logits0, val0 = pol(torch.as_tensor(cand), torch.as_tensor(mask),
                            torch.as_tensor(ctx))
        # Perturb the padded rows: must not change valid logits or the value.
        cand2 = cand.copy()
        cand2[0, n_valid:] = rng.standard_normal((K_CAND - n_valid, F_JOB))
        logits1, val1 = pol(torch.as_tensor(cand2), torch.as_tensor(mask),
                            torch.as_tensor(ctx))
    dl = float(torch.max(torch.abs(
        logits1[0, :n_valid] - logits0[0, :n_valid])).item())
    dv = float(abs(val1.item() - val0.item()))
    print("    padding-invariance: max_valid_logit_delta=%.2e val_delta=%.2e"
          % (dl, dv))
    if dl > 1e-5 or dv > 1e-5:
        failures.append("padding invariance: logit=%.2e value=%.2e (>1e-5)"
                        % (dl, dv))


def test_parity_smoke(failures):
    """One full env episode runs end-to-end through act()/step()."""
    inst_path = _find_instance()
    if inst_path is None:
        print("    smoke: no instance found, SKIP")
        return
    import json
    with open(inst_path) as fh:
        inst = json.load(fh)
    torch.manual_seed(3)
    pol = AttnDispatchPolicy().eval()
    env = DispatchEnv(inst)
    obs = env.reset()
    steps = 0
    done = False
    while not done:
        a, logp, val, ent = pol.act(obs, greedy=True, device="cpu")
        if not (0 <= a < K_CAND) or not obs["mask"][a]:
            failures.append("smoke: policy picked invalid/padded action %d" % a)
            break
        obs, r, done, info = env.step(a)
        steps += 1
        if steps > 200000:
            failures.append("smoke: episode did not terminate")
            break
    sched = env.to_schedule("attn")
    n_assign = len(sched["assignments"])
    n_wo = len(inst["work_orders"])
    print("    smoke: steps=%d assignments=%d work_orders=%d"
          % (steps, n_assign, n_wo))
    if n_assign != n_wo:
        failures.append("smoke: assignments %d != work_orders %d"
                        % (n_assign, n_wo))


def _find_instance():
    root = os.path.join(os.path.dirname(__file__), "..",
                        "data", "processed", "instances")
    for c in ("c05", "c09", "c10", "c12"):
        d = os.path.join(root, c, "replay", "150")
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    return os.path.join(d, f)
    # fixture fallback
    fx = os.path.join(os.path.dirname(__file__), "fixtures")
    if os.path.isdir(fx):
        for f in sorted(os.listdir(fx)):
            if f.endswith(".json"):
                return os.path.join(fx, f)
    return None


def main():
    failures = []
    print("test_permutation_equivariance"); test_permutation_equivariance(failures)
    print("test_mask_correctness"); test_mask_correctness(failures)
    print("test_value_invariant_to_padding"); test_value_invariant_to_padding(failures)
    print("test_parity_smoke"); test_parity_smoke(failures)

    print()
    if failures:
        print("POLICY_ATTN TESTS FAILED (%d):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        sys.exit(1)
    print("ALL POLICY_ATTN TESTS PASSED")


if __name__ == "__main__":
    main()
