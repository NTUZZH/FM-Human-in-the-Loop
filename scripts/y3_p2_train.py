"""M1 outer DAgger loop runner (Paper Y3, Phase P2).

Per outer iteration (8 by default): roll out the CURRENT policy under the
simulated supervisor for a fixed slice of the fair-compute budget, log every
reviewed decision into the aggregated intervention buffer D_int, train the
backbone with the intervention-weighted objective (Y1 PPO on the OBSERVABLE
shaped reward + a supervised imitation term toward the supervisor's preferred
pick), and train the gated latent-shift head on the override stream as weak
labels. Metrics are written per iteration to a CSV.

FAIR-COMPUTE RULE (asserted in code): the TOTAL environment-step budget across
all outer iterations equals Y1's single-run PPO budget (read from
results/p3_train/<seed>/config.json: updates * n_envs * steps_per_env). At the
full budget the runner asserts equality; a reduced smoke run asserts internal
consistency only and logs the fraction.

Information structure (RED LINES, asserted):
* the RL reward is the env's OBSERVABLE shaped reward (recorded weights) --
  unchanged Y1 path; the latent never enters observations or the reward.
* the latent reaches the learner ONLY through the override/confirmation log
  (D_int and the weak-label stream).
* hat_s accuracy vs the true shift is computed by a clearly QUARANTINED eval
  function that is the sole place the overlay latent is read, for reporting only.

CLI
---
python scripts/y3_p2_train.py --beta 0.75 --rho 0.25 --eps 0 --seed 301 \
    --out train_log/y3_p2/m1_pilot            # full Tier-1 pilot
python scripts/y3_p2_train.py --smoke --out /tmp/.../smoke   # 2 iters, ~5%
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import validator                                    # noqa: E402
from fmwos.env import DispatchEnv, K_CAND                      # noqa: E402
from fmwos.train import InstanceSampler, load_dev_set, _stack_obs  # noqa: E402
from fmwos.hitl.overlay import Overlay, OverlayParams          # noqa: E402
from fmwos.hitl.supervisor import Supervisor                   # noqa: E402
from fmwos.hitl.latent_head import (                           # noqa: E402
    LatentDispatchPolicy, latfeat_for_candidates,
    weak_labels_from_entries, train_estimator)
from fmwos.hitl.intervention import (                          # noqa: E402
    InterventionBuffer, imitation_loss, OVERRIDE_WEIGHT, CONFIRM_WEIGHT)
from fmwos.hitl import augmented_rule                          # noqa: E402
from fmwos.hitl import true_objective                          # noqa: E402

CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]


# --------------------------------------------------------------------------- #
# Fair-compute budget                                                         #
# --------------------------------------------------------------------------- #
def y1_single_run_budget(seed=301):
    """Y1's single-run PPO env-step budget = updates * n_envs * steps_per_env."""
    cfg_path = os.path.join(_ROOT, "results", "p3_train", "seed%d" % seed, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    return int(cfg["updates"]) * int(cfg["n_envs"]) * int(cfg["steps_per_env"]), cfg


def resolve_schedule(total_budget, outer_iters, n_envs, steps_per_env, y1_budget,
                     budget_frac=1.0):
    """updates_per_iter such that iters*updates*n_envs*steps == total_budget."""
    per_update = n_envs * steps_per_env
    updates_per_iter = total_budget // (outer_iters * per_update)
    configured = updates_per_iter * outer_iters * per_update
    # fair-compute assertions
    if abs(budget_frac - 1.0) < 1e-12:
        assert configured == y1_budget, (
            "FAIR-COMPUTE VIOLATION: configured budget %d != Y1 budget %d"
            % (configured, y1_budget))
    assert configured == updates_per_iter * outer_iters * per_update
    return updates_per_iter, configured


# --------------------------------------------------------------------------- #
# Per-env slot: env + bound supervisor + latent-feature cache                 #
# --------------------------------------------------------------------------- #
class Slot:
    def __init__(self, instance, overlay, cell, seed):
        self.overlay = overlay
        self.cell = cell
        self.seed = seed
        self._bind(instance)

    def _bind(self, instance):
        self.instance = instance
        self.applied = self.overlay.apply(instance)
        self.env = DispatchEnv(instance, reward_mode="shaped")
        self.sup = Supervisor(self.overlay, instance, rho=self.cell["rho"],
                              epsilon=self.cell["eps"], theta=self.cell["theta"],
                              mechanism=self.cell["mechanism"], seed=self.seed,
                              applied=self.applied)
        self.feat_cache = {}
        self.obs = self.env.reset()
        self.done = self.env._done

    def rebind(self, instance):
        self._bind(instance)


def make_slot(sampler, overlay, cell, seed):
    return Slot(sampler.sample(), overlay, cell, seed)


# --------------------------------------------------------------------------- #
# Evaluation (policy ALONE): observable + true TWT, and hat_s recovery         #
# --------------------------------------------------------------------------- #
def eval_policy_alone(policy, instances, overlay, device="cpu"):
    """Greedy policy ALONE (no supervisor). Returns mean observable WWT and mean
    true TWT* over ``instances`` (gate is left as configured on the policy)."""
    policy.eval()
    obs_list, true_list = [], []
    for inst in instances:
        applied = overlay.apply(inst)
        env = DispatchEnv(inst)
        obs = env.reset()
        fc = {}
        done = env._done
        while not done:
            lat = latfeat_for_candidates(env._candidates, fc)
            a, _lp, _v, _m = policy.act_with_margin(obs, latfeat=lat, greedy=True,
                                                    device=device)
            obs, _r, done, _i = env.step(a)
        sched = env.to_schedule("m1")
        obs_list.append(validator.validate(inst, sched)["metrics"]["WWT"])
        true_list.append(true_objective.score_true(inst, sched, overlay, applied)["TWT_true"])
    return float(np.mean(obs_list)), float(np.mean(true_list))


# --------------------------------------------------------------------------- #
# Main training                                                               #
# --------------------------------------------------------------------------- #
def train_m1(cell, seed, out_dir, outer_iters=8, budget_frac=1.0, gate=1.0,
             override_weight=OVERRIDE_WEIGHT, confirm_weight=CONFIRM_WEIGHT,
             il_coef=1.0, buffer_capacity=60000, head_epochs=30,
             n_probe=16, device="cpu", smoke=False,
             sampler=None, probe=None, overlay=None, deadline_head=False,
             label_source="preferred", il_pure=False):
    """Train the M1 DAgger loop for one cell.

    ``sampler`` / ``probe`` / ``overlay`` are optional injection points (P1.5):
    the default replay-campus sampler and dev set are used when they are None, so
    the locked Tier-1 behaviour is unchanged; a caller (e.g. the storm2
    contention runner) can inject a storm2 sampler + probe set + a
    channel-specific overlay instead. The overlay defaults to the cell's
    full_class_shift overlay."""
    os.makedirs(out_dir, exist_ok=True)
    if label_source not in ("preferred", "executed"):
        raise ValueError("label_source must be 'preferred' or 'executed'")
    torch.manual_seed(seed); np.random.seed(seed)

    n_envs, steps_per_env = 16, 512
    gamma, lam, clip = 1.0, 0.98, 0.2
    epochs, ent_coef, val_coef, max_grad, lr, minibatch = 4, 0.01, 0.5, 0.5, 3e-4, 1024

    y1_budget, y1_cfg = y1_single_run_budget()
    total_budget = int(round(y1_budget * budget_frac))
    updates_per_iter, configured = resolve_schedule(
        total_budget, outer_iters, n_envs, steps_per_env, y1_budget, budget_frac)

    if overlay is None:
        overlay = Overlay(OverlayParams(beta=cell["beta"], family=cell["family"],
                                        master_seed=cell["master_seed"],
                                        channel=cell.get("channel", "full_class_shift")))
    if sampler is None:
        sampler = InstanceSampler(CAMPUSES, SIZES, seed, curriculum="v1")
    if probe is None:
        probe = load_dev_set(CAMPUSES, SIZES, n_probe, dict(sampler._cache))

    policy = LatentDispatchPolicy(gate=gate, deadline_head=deadline_head).to(device)
    # The weak-label shift head is trained separately (train_estimator); every
    # other parameter -- including the fair-M1 deadline head -- is trained by
    # PPO + the intervention-weighted imitation term.
    backbone_params = [p for n, p in policy.named_parameters()
                       if not n.startswith("shift_head")]
    optim = torch.optim.Adam(backbone_params, lr=lr)

    D_int = InterventionBuffer(capacity=buffer_capacity, seed=seed)

    config = {
        "cell": cell, "seed": seed, "outer_iters": outer_iters,
        "updates_per_iter": updates_per_iter, "n_envs": n_envs,
        "steps_per_env": steps_per_env, "configured_env_steps": configured,
        "y1_budget": y1_budget, "budget_frac": budget_frac,
        "fair_compute_ok": (configured == y1_budget) if budget_frac == 1.0 else "smoke",
        "gate": gate, "channel": overlay.params.channel,
        "deadline_head": bool(deadline_head),
        "label_source": label_source, "il_pure": bool(il_pure),
        "override_weight": override_weight,
        "confirm_weight": confirm_weight, "il_coef": il_coef,
        "buffer_capacity": buffer_capacity, "head_epochs": head_epochs,
        "ppo": {"gamma": gamma, "lam": lam, "clip": clip, "epochs": epochs,
                "ent_coef": ent_coef, "val_coef": val_coef, "lr": lr,
                "minibatch": minibatch},
        "y1_config_ref": {k: y1_cfg[k] for k in ("updates", "n_envs", "steps_per_env")},
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print("[y3_p2] Y1 budget=%d  configured=%d  frac=%.3f  updates/iter=%d  iters=%d"
          % (y1_budget, configured, budget_frac, updates_per_iter, outer_iters))
    print("[y3_p2] FAIR-COMPUTE %s"
          % ("OK (==Y1)" if configured == y1_budget else "SMOKE (%.1f%%)" % (100 * budget_frac)))
    if label_source != "preferred" or il_pure:
        print("[y3_p2] ABLATION MODE: label_source=%s il_pure=%s "
              "(eps=%.2f; at eps=0 executed==preferred on overrides => label_source is a no-op)"
              % (label_source, il_pure, cell.get("eps", 0.0)))

    csv_path = os.path.join(out_dir, "metrics.csv")
    cols = ["iter", "updates", "env_steps_cum", "override_rate", "confirmation_rate",
            "review_fraction", "n_reviews", "n_reviewable", "mean_return",
            "hat_s_sign_acc", "hat_s_exact_acc", "hat_s_zero_baseline",
            "hat_s_pearson_r", "obs_twt", "true_twt", "est_loss", "seconds"]
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerow(cols)

    slots = [make_slot(sampler, overlay, cell, seed) for _ in range(n_envs)]
    env_steps_cum = 0
    L = steps_per_env

    for it in range(outer_iters):
        t_it = time.perf_counter()
        it_dec = it_reviewable = it_reviews = it_over = it_conf = 0
        completed_returns = []
        ep_return = [0.0] * n_envs

        for _u in range(updates_per_iter):
            b_cand = np.zeros((L, n_envs, K_CAND, policy.f_job), np.float32)
            b_mask = np.zeros((L, n_envs, K_CAND), bool)
            b_ctx = np.zeros((L, n_envs, policy.f_ctx), np.float32)
            b_lat = np.zeros((L, n_envs, K_CAND, policy.lat_dim), np.float32)
            b_act = np.zeros((L, n_envs), np.int64)
            b_logp = np.zeros((L, n_envs), np.float32)
            b_val = np.zeros((L, n_envs), np.float32)
            b_rew = np.zeros((L, n_envs), np.float32)
            b_done = np.zeros((L, n_envs), np.float32)

            policy.eval()
            for t in range(L):
                # build the batched observation (+ latent features)
                cand, mask, ctx = _stack_obs([s.obs for s in slots])
                lat = np.stack([latfeat_for_candidates(s.env._candidates, s.feat_cache)
                                for s in slots])
                ct = torch.as_tensor(cand, dtype=torch.float32, device=device)
                mt = torch.as_tensor(mask, dtype=torch.bool, device=device)
                xt = torch.as_tensor(ctx, dtype=torch.float32, device=device)
                lt = torch.as_tensor(lat, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits, value = policy(ct, mt, xt, lt)
                    logp_all = torch.log_softmax(logits, dim=-1)
                    probs = logp_all.exp() * mt.to(logits.dtype)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)

                b_cand[t] = cand; b_mask[t] = mask; b_ctx[t] = ctx; b_lat[t] = lat
                b_val[t] = value.cpu().numpy()

                for i, s in enumerate(slots):
                    cands = s.env._candidates
                    now = s.env._cur_now
                    a_pi = int(sampled[i].item())
                    decider_pick = cands[a_pi]
                    # margin = top1-top2 masked softmax prob
                    p_i = probs[i]
                    kk = min(2, int(mt[i].sum().item()) or 1)
                    top = torch.topk(p_i, k=kk).values
                    margin = float(top[0] - top[1]) if top.numel() >= 2 else 1e9

                    it_dec += 1
                    if len(cands) >= 2:
                        it_reviewable += 1
                    executed_pick, entry = s.sup.review(decider_pick, cands, now, margin)
                    exec_action = cands.index(executed_pick)

                    if entry["reviewed"]:
                        it_reviews += 1
                        if entry["override"]:
                            it_over += 1
                        elif entry["confirmation"]:
                            it_conf += 1
                        # preferred index within the candidate set
                        pref_id = entry["preferred_pick"]
                        pref_idx = None
                        if pref_id is not None:
                            for ci, c in enumerate(cands):
                                if c["id"] == pref_id:
                                    pref_idx = ci
                                    break
                        D_int.add({
                            "cand": cand[i].copy(), "mask": mask[i].copy(),
                            "ctx": ctx[i].copy(), "latfeat": lat[i].copy(),
                            "decider_idx": a_pi, "preferred_idx": pref_idx,
                            "executed_idx": exec_action,
                            "override": bool(entry["override"]),
                            "confirmation": bool(entry["confirmation"]),
                            "margin": float(margin),
                        })

                    # store the EXECUTED transition for PPO (old_logp of executed)
                    b_act[t, i] = exec_action
                    b_logp[t, i] = float(logp_all[i, exec_action].item())

                    nobs, r, done, _info = s.env.step(exec_action)
                    b_rew[t, i] = r
                    b_done[t, i] = 1.0 if done else 0.0
                    ep_return[i] += r
                    if done:
                        completed_returns.append(ep_return[i])
                        ep_return[i] = 0.0
                        s.rebind(sampler.sample())
                    else:
                        s.obs = nobs
                env_steps_cum += n_envs

            # bootstrap value for the final obs
            cand, mask, ctx = _stack_obs([s.obs for s in slots])
            lat = np.stack([latfeat_for_candidates(s.env._candidates, s.feat_cache)
                            for s in slots])
            with torch.no_grad():
                _lg, last_val = policy(
                    torch.as_tensor(cand, dtype=torch.float32, device=device),
                    torch.as_tensor(mask, dtype=torch.bool, device=device),
                    torch.as_tensor(ctx, dtype=torch.float32, device=device),
                    torch.as_tensor(lat, dtype=torch.float32, device=device))
            last_val = last_val.cpu().numpy()

            # GAE
            adv = np.zeros((L, n_envs), np.float32)
            lastgae = np.zeros(n_envs, np.float32)
            for t in reversed(range(L)):
                nonterminal = 1.0 - b_done[t]
                nextval = last_val if t == L - 1 else b_val[t + 1]
                delta = b_rew[t] + gamma * nextval * nonterminal - b_val[t]
                lastgae = delta + gamma * lam * nonterminal * lastgae
                adv[t] = lastgae
            ret = adv + b_val

            N = L * n_envs
            f_cand = torch.as_tensor(b_cand.reshape(N, K_CAND, policy.f_job), device=device)
            f_mask = torch.as_tensor(b_mask.reshape(N, K_CAND), device=device)
            f_ctx = torch.as_tensor(b_ctx.reshape(N, policy.f_ctx), device=device)
            f_lat = torch.as_tensor(b_lat.reshape(N, K_CAND, policy.lat_dim), device=device)
            f_act = torch.as_tensor(b_act.reshape(N), device=device)
            f_logp = torch.as_tensor(b_logp.reshape(N), device=device)
            f_adv = torch.as_tensor(adv.reshape(N), device=device)
            f_ret = torch.as_tensor(ret.reshape(N), device=device)
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

            policy.train()
            mb = min(minibatch, N)
            idx = np.arange(N)
            for _ep in range(epochs):
                np.random.shuffle(idx)
                for start in range(0, N, mb):
                    sl = torch.as_tensor(idx[start:start + mb], device=device)
                    logp, entropy, value = policy.evaluate(
                        f_cand[sl], f_mask[sl], f_ctx[sl], f_act[sl], f_lat[sl])
                    ratio = torch.exp(logp - f_logp[sl])
                    a = f_adv[sl]
                    pg = torch.max(-a * ratio,
                                   -a * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                    v_loss = ((value - f_ret[sl]) ** 2).mean()
                    ent = entropy.mean()
                    # intervention-weighted imitation term on D_int
                    ib = D_int.sample_batch(mb, device=device, label_source=label_source)
                    il = imitation_loss(policy, ib, override_weight=override_weight,
                                        confirm_weight=confirm_weight, use_latent=True)
                    if il_pure:
                        # IL-PURE ablation: zero the PPO contribution (no pg / value /
                        # entropy gradient); learn from the imitation term ONLY. The
                        # rollout above is untouched, so the fair-compute env-step
                        # budget and the supervisor data stream are unchanged.
                        if il is None:
                            continue
                        loss = il_coef * il
                    else:
                        loss = pg + val_coef * v_loss - ent_coef * ent
                        if il is not None:
                            loss = loss + il_coef * il
                    optim.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(backbone_params, max_grad)
                    optim.step()

        # ---- train the latent-shift head on the aggregated override stream ---- #
        Xw, yw, ww = weak_labels_from_entries(D_int.entries, override_weight,
                                              confirm_weight, label_source=label_source)
        est_loss = train_estimator(policy.shift_head, Xw, yw, ww, epochs=head_epochs,
                                   device=device, seed=seed + it)

        # ---- metrics ---- #
        acc = augmented_rule.probe_shift_accuracy(policy.shift_head, probe, overlay,
                                                  device=device)
        obs_twt, true_twt = eval_policy_alone(policy, probe, overlay, device=device)
        orr = (it_over / it_reviews) if it_reviews else 0.0
        confr = (it_conf / it_reviews) if it_reviews else 0.0
        revf = (it_reviews / it_reviewable) if it_reviewable else 0.0
        mret = float(np.mean(completed_returns)) if completed_returns else 0.0
        secs = time.perf_counter() - t_it
        row = [it, (it + 1) * updates_per_iter, env_steps_cum, orr, confr, revf,
               it_reviews, it_reviewable, mret, acc["sign_acc_nonzero"],
               acc["exact_class_acc"], acc["zero_baseline_acc"], acc["pearson_r"],
               obs_twt, true_twt, est_loss, secs]
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(["%.6f" % v if isinstance(v, float) else v for v in row])
        policy.save(os.path.join(out_dir, "iter%d.pt" % it))
        print("[it%d] over_rate=%.3f conf_rate=%.3f rev_frac=%.3f reviews=%d "
              "ret=%.3f | hat_s sign=%.3f exact=%.3f (zero=%.3f) r=%.3f | "
              "obsTWT=%.3f trueTWT=%.3f | Dint=%d | %.1fs"
              % (it, orr, confr, revf, it_reviews, mret, acc["sign_acc_nonzero"],
                 acc["exact_class_acc"], acc["zero_baseline_acc"], acc["pearson_r"],
                 obs_twt, true_twt, len(D_int), secs))

    policy.save(os.path.join(out_dir, "final.pt"))
    print("[y3_p2] done -> %s  (D_int stats: %s)" % (out_dir, D_int.stats()))
    return {"out_dir": out_dir, "D_int": D_int.stats()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--mechanism", type=str, default="targeted",
                    choices=["targeted", "random"])
    ap.add_argument("--family", type=str, default="F-NL", choices=["F-LIN", "F-NL"])
    ap.add_argument("--master-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--il-coef", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true",
                    help="2 outer iters, ~5%% budget (fair-compute equality relaxed)")
    ap.add_argument("--budget-frac", type=float, default=1.0)
    ap.add_argument("--deadline-head", action="store_true",
                    help="fair M1: add the in-network ATC-slack deadline head "
                         "(E0-preserving, zero-init); default OFF = old M1")
    ap.add_argument("--label-source", type=str, default="preferred",
                    choices=["preferred", "executed"],
                    help="imitation/weak-label target on OVERRIDES: 'preferred' "
                         "(committed) or 'executed' (honest under eps>0 noise; "
                         "bit-identical at eps=0)")
    ap.add_argument("--il-pure", action="store_true",
                    help="IL-PURE ablation: zero the PPO loss, learn from the "
                         "imitation term only (rollouts/env-steps unchanged)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    cell = {"beta": args.beta, "rho": args.rho, "eps": args.eps, "theta": args.theta,
            "mechanism": args.mechanism, "family": args.family,
            "master_seed": args.master_seed}
    outer_iters = 2 if args.smoke else args.outer_iters
    budget_frac = 0.05 if args.smoke else args.budget_frac
    train_m1(cell, args.seed, args.out, outer_iters=outer_iters,
             budget_frac=budget_frac, gate=args.gate, il_coef=args.il_coef,
             smoke=args.smoke, deadline_head=args.deadline_head,
             label_source=args.label_source, il_pure=args.il_pure)


if __name__ == "__main__":
    main()
