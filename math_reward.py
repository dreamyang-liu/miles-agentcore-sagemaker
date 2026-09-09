"""RLVR reward for the AgentCore math recipe, plus a rollout class that logs agent metrics.

Grading lives here rather than in the agent so the ground truth never leaves the training
cluster: the AgentCore agent only ever sees the question, submits an answer, and Miles decides
whether it was right.

The reward is binary. Comparison is numeric when both sides parse as numbers, so ``72``,
``72.0``, ``$72`` and ``1,000`` behave the way a human would expect, with a string fallback
for non-numeric answers.
"""

from __future__ import annotations

import logging
import math
import re

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Absolute tolerance, deliberately not relative. A tolerance scaled by magnitude makes
# 1000000 and 1000001 compare equal (1e-6 * 1e6 == 1), which rewards a wrong answer and
# quietly poisons training.
#
# A pure absolute tolerance is wrong too, once answers get large. GSM-Hard's median answer is
# ~2.9e6 and 22.9% are non-integer, including ground truths like 3244047.0999999996 -- double
# representation error alone exceeds 1e-9 at that magnitude, so a correct answer would grade
# as wrong.
#
# math.isclose with both bounds is the shape that satisfies both: rel_tol scales with the
# value (1e-9 * 1e6 = 1e-3, still far below the 0.5 that would let adjacent integers pass),
# and abs_tol keeps tiny values from matching zero.
_REL_TOL = 1e-9
_ABS_TOL = 1e-9

_NUMERIC_JUNK = re.compile(r"[,$%\s]")
# Last number in the string, so "the answer is 72" still grades. Handles 1e3 and -4.5.
_LAST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _to_float(text: str) -> float | None:
    cleaned = _NUMERIC_JUNK.sub("", text)
    try:
        return float(cleaned)
    except ValueError:
        pass
    matches = _LAST_NUMBER.findall(cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def answers_match(submitted: str | None, truth: str | None) -> bool:
    """True when the submitted answer equals the ground truth, numerically if possible."""
    if submitted is None or truth is None:
        return False
    submitted, truth = str(submitted).strip(), str(truth).strip()
    if not submitted or not truth:
        return False
    got, want = _to_float(submitted), _to_float(truth)
    if got is not None and want is not None:
        return math.isclose(got, want, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    return submitted.casefold() == truth.casefold()


# Cost of trying to answer without reading a tool result. The agent's guard already makes such
# a submit_answer fail, but a guard alone changes only what is reachable, not what is
# rewarded -- and a rejected attempt costs the policy nothing, so GRPO converges on "fire
# everything in parallel, and resubmit if it bounces". Across two 50-rollout runs
# rejected_submits climbed 0.17 -> 1.39 while reward kept rising.
#
# Deliberately small. Being correct is worth 1.0, so at 0.05 a bounce is a 5% nudge, not a
# gate: the policy still prefers a correct answer that took three bounces (0.85) over a wrong
# one that took none (0.0). The cap stops a pathological episode from dominating its group's
# advantage.
_REJECT_PENALTY = 0.05
_MAX_REJECT_PENALTY = 0.15


def is_correct(sample: Sample) -> bool:
    metadata = sample.metadata or {}
    return answers_match(metadata.get("submitted_answer"), metadata.get("answer"))


def reject_penalty(sample: Sample) -> float:
    rejected = ((sample.metadata or {}).get("agent_metrics") or {}).get("rejected_submits") or 0
    return min(_MAX_REJECT_PENALTY, _REJECT_PENALTY * float(rejected))


def score_sample(sample: Sample) -> float:
    """Correctness, minus a small toll on submits that never read a tool result.

    Can go slightly negative (a wrong answer that also bounced). That is fine for GRPO, which
    only ever uses advantages within a group.
    """
    return (1.0 if is_correct(sample) else 0.0) - reject_penalty(sample)


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Binary correctness reward.

    Handles both the single-sample call from ``async_rm`` and the batched call from
    ``batched_async_rm`` when ``--custom-rm-path`` is set.
    """
    if isinstance(samples, list):
        return [score_sample(sample) for sample in samples]
    return score_sample(samples)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Roll up agent-side counters, plus how episodes ended, for the training log."""
    metrics: dict[str, float] = {}
    all_metrics = [
        s.metadata["agent_metrics"]
        for s in samples
        if getattr(s, "metadata", None) and s.metadata.get("agent_metrics")
    ]
    if all_metrics:
        # rejected_submits is the one to watch when the parallel-submit guard is on: it counts
        # attempts to answer without reading a tool result, so it should fall as the policy
        # learns the turn discipline. Omitting a key here silently drops it from wandb.
        for key in (
            "turns",
            "tool_calls",
            "calculator_calls",
            "rejected_submits",
            "total_tool_time",
            "total_time",
        ):
            values = [value for m in all_metrics if (value := m.get(key)) is not None]
            if values:
                metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    scored = [s for s in samples if getattr(s, "metadata", None)]
    if scored:
        # Kept separate from rollout/raw_reward, which now carries the rejection toll: this is
        # the number to read as "how good is the policy", the reward is the training signal.
        metrics["agent/correct_rate"] = sum(1 for s in scored if is_correct(s)) / len(scored)
        metrics["agent/reject_penalty_mean"] = sum(reject_penalty(s) for s in scored) / len(scored)

    statuses = [s.metadata.get("exit_status", "") for s in samples if getattr(s, "metadata", None)]
    if statuses:
        # submit_rate is the one to watch early: a model that never calls submit_answer
        # scores zero everywhere and the reward curve looks broken rather than untrained.
        metrics["agent/submit_rate"] = sum(1 for s in statuses if s == "submitted") / len(statuses)
        for status in {s for s in statuses if s}:
            metrics[f"agent/exit_{status}"] = statuses.count(status) / len(statuses)
    return metrics


class RolloutFn(InferenceRolloutFn):
    """Rollout function that adds agent-metric aggregation to the training log."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        all_samples: list[Sample] = []
        for group in output.samples:
            all_samples.extend(group) if isinstance(group, list) else all_samples.append(group)

        agent_metrics = aggregate_agent_metrics(all_samples)
        if agent_metrics:
            metrics = output.metrics or {}
            metrics.update(agent_metrics)
            output.metrics = metrics
            logger.info("agent metrics for rollout %s: %s", input.rollout_id, agent_metrics)
        return output
