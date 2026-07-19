"""P1 deciders wired around the supervisor (cheap, no training).

* RULE          -- a Y1 dispatching rule, alone (reference; identical to
                   ``env.run_policy``).
* RULE+SUP      -- a Y1 rule deployed with the supervisor in the loop.
* ORACLE-GREEDY -- execute the supervisor's myopic-greedy preferred pick at
                   every event (the imitation skyline, Sec.8.1).

All three drive ``env.run_supervised`` so their event semantics are identical.

Channel awareness (P1.5): ORACLE-GREEDY and RULE+SUP read the TRUE deadline d*
for the oracle / preferred pick THROUGH the supervisor, whose ``self.due`` is set
to d* (r+SLA(c*)) under the full_class_shift channel and to the recorded due
under the weight_only E6 boundary (see ``Supervisor.__init__``). RULE alone is a
recorded-field rule and is unchanged. The deciders stay decider-agnostic: they
never read the latent themselves; d* enters only via the supervisor's preferred
pick and override improvement.
"""

from __future__ import annotations

from .. import pdrs


# --------------------------------------------------------------------------- #
# Deciders: callable(queue, t, rng) -> (job, margin)                          #
# --------------------------------------------------------------------------- #
def rule_decider(rule_name):
    """A Y1 rule as a (job, margin) decider (pick unchanged from the rule)."""
    def _decider(queue, t, rng):
        return pdrs.pick_with_margin(rule_name, queue, t, rng)
    return _decider


def oracle_decider(supervisor):
    """A decider that always returns the supervisor's preferred pick.

    Margin is irrelevant (no review runs for ORACLE-GREEDY); reported as +inf.
    """
    def _decider(queue, t, rng):
        return supervisor.preferred_pick(list(queue), t), pdrs._BIG_MARGIN
    return _decider


# --------------------------------------------------------------------------- #
# Runners                                                                     #
# --------------------------------------------------------------------------- #
def run_rule(env, rule_name, seed=0):
    """RULE alone. Returns a schedule dict (byte-identical to run_policy)."""
    sched, _log = env.run_supervised(rule_decider(rule_name), supervisor=None,
                                     method=rule_name, seed=seed)
    return sched


def run_rule_sup(env, rule_name, supervisor, seed=0):
    """RULE+SUP. Returns (schedule, override_log)."""
    return env.run_supervised(rule_decider(rule_name), supervisor=supervisor,
                              method=rule_name + "+sup", seed=seed)


def run_oracle_greedy(env, supervisor, seed=0):
    """ORACLE-GREEDY. The supervisor object is used only for its preferred-pick
    logic (no review / override protocol); returns a schedule dict."""
    sched, _log = env.run_supervised(oracle_decider(supervisor), supervisor=None,
                                     method="oracle_greedy", seed=seed)
    return sched
