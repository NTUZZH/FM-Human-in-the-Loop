"""True-objective scoring for the supervisor overlay (Paper Y3, Phase P1/P1.5).

The RECORDED-objective path in ``fmwos.validator`` is left untouched: every
released number and the E0 anchor depend on its byte-for-byte behaviour. This
module adds a sibling scorer that reuses the independent validator's feasibility
verdict and recomputes ONLY the weighted-tardiness metric block against the
realized latent quantities:

    TWT* = sum_j w*(c*_j) * max(0, C_j - due_j)   (+ access penalties when on)

The DEADLINE used in ``due_j`` depends on the private-information channel
(``deadline_mode``, P1.5 re-scope):

  * ``"true"``     (full_class_shift HEADLINE): due_j = d*_j = r_j + SLA(c*_j).
    The latent moves BOTH the cost of lateness (w -> w*) AND the clock
    (d -> d*). This is the P1.5 headline objective.
  * ``"recorded"`` (weight_only E6 BOUNDARY): due_j = recorded d_j. The latent
    moves only the cost of lateness; the deadline is frozen. This reproduces the
    pre-P1.5 ``score_true`` byte-for-byte.

The default is resolved FROM THE OVERLAY'S CHANNEL (full_class_shift -> "true",
weight_only -> "recorded") so a caller who passes nothing gets scoring
consistent with the active channel; pass ``deadline_mode`` explicitly to force a
mode. The independent-validator feasibility path is unchanged in both modes.

Like the validator, this module shares no code with the scheduler or the
environment; it depends only on the validator (feasibility) and the overlay
(true weights / true deadline).
"""

from __future__ import annotations

from .. import validator as _validator

_BREACH_TOL = 1e-9   # matches fmwos.validator

# channel -> default deadline_mode when the caller passes deadline_mode=None.
_CHANNEL_DEADLINE_MODE = {"full_class_shift": "true", "weight_only": "recorded"}


def resolve_deadline_mode(overlay, deadline_mode=None) -> str:
    """The deadline mode to score with: explicit override, else the overlay's
    channel default (full_class_shift -> "true", weight_only -> "recorded")."""
    if deadline_mode is not None:
        if deadline_mode not in ("true", "recorded"):
            raise ValueError("deadline_mode must be 'true' or 'recorded'")
        return deadline_mode
    channel = getattr(getattr(overlay, "params", None), "channel", "full_class_shift")
    return _CHANNEL_DEADLINE_MODE.get(channel, "true")


def score_true(instance: dict, schedule: dict, overlay, applied: dict | None = None,
               deadline_mode: str | None = None) -> dict:
    """Score ``schedule`` on the TRUE objective under ``overlay``.

    Parameters
    ----------
    instance : dict          the (unmodified) benchmark instance
    schedule : dict          an executed schedule (assignments)
    overlay  : Overlay       the bound supervisor overlay
    applied  : dict|None     optional precomputed ``overlay.apply(instance)``
                             (pass it to avoid re-drawing the latent)
    deadline_mode : str|None "true" => due = d* (r+SLA(c*)); "recorded" => due =
                             recorded due_bh. ``None`` (default) resolves from the
                             overlay channel (see ``resolve_deadline_mode``).

    Returns a dict with feasibility (from the independent validator), the true
    objective ``TWT_true``, the recorded objective ``TWT_recorded`` (for
    reference), the resolved ``deadline_mode``, the access penalty, and a
    per-class true-tardiness breakdown.
    """
    base = _validator.validate(instance, schedule)
    if applied is None:
        applied = overlay.apply(instance)
    wstar = applied["w_star"]
    cstar = applied["c_star"]
    dstar = applied.get("d_star", {})
    mode = resolve_deadline_mode(overlay, deadline_mode)

    wo_by_id = {wo["id"]: wo for wo in instance.get("work_orders", []) or []}

    twt_true = 0.0
    twt_recorded = 0.0
    n = 0
    breaches = 0
    per_class_true = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for a in schedule.get("assignments", []) or []:
        wo = wo_by_id.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        end = float(end)
        d_recorded = float(wo["due_bh"])
        if mode == "true":
            due = float(dstar.get(wo["id"], d_recorded))
        else:
            due = d_recorded
        tard = max(0.0, end - due)
        w_true = wstar.get(wo["id"], float(wo["weight"]))
        twt_true += w_true * tard
        # recorded objective: recorded weight against the recorded deadline
        # (unchanged reference, independent of ``mode``).
        twt_recorded += float(wo["weight"]) * max(0.0, end - d_recorded)
        n += 1
        if end > due + _BREACH_TOL:
            breaches += 1
        c = cstar.get(wo["id"], int(wo["priority"]))
        if c in per_class_true:
            per_class_true[c] += w_true * tard

    access = overlay.access_penalty(instance, schedule)
    twt_true_total = twt_true + access

    return {
        "feasible": base["feasible"],
        "violations": base["violations"],
        "TWT_true": twt_true_total,
        "TWT_true_tardiness": twt_true,   # without access penalty
        "TWT_recorded": twt_recorded,
        "deadline_mode": mode,
        "access_penalty": access,
        "n_scored": n,
        "breaches": breaches,
        "per_class_true": per_class_true,
    }
