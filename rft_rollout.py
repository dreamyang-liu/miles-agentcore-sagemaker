"""Bounded whole-trajectory retries for synchronous RFT training.

Each attempt calls Miles' generator again, which creates a fresh traced session.
An exhausted trajectory fails the batch in the filter: the upstream sampler catches
generate-task exceptions and would otherwise keep replacing failed prompt groups.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import replace

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.types import Sample

from math_reward import RolloutFn as MathRolloutFn

logger = logging.getLogger(__name__)


def _flatten(samples):
    if isinstance(samples, list):
        for sample in samples:
            yield from _flatten(sample)
    else:
        yield samples


def _failure_reason(output):
    samples = list(_flatten(output.samples))
    if not samples:
        return "empty trajectory"
    for sample in samples:
        if sample.status == Sample.Status.ABORTED:
            return "aborted trajectory"
        status = (sample.metadata or {}).get("exit_status")
        if not status or status == "error":
            return f"missing or failed RFT result: {status!r}"
    return None


async def generate(input):
    tasks = getattr(input.state, "_rft_generation_tasks", None)
    if tasks is None:
        tasks = input.state._rft_generation_tasks = set()
    task = asyncio.current_task()
    tasks.add(task)
    attempts = input.args.rft_rollout_max_retries + 1
    reason = "no attempt"
    try:
        for attempt in range(1, attempts + 1):
            if input.state.aborted or task.cancelling():
                raise asyncio.CancelledError
            try:
                output = await agentic_tool_call.generate(
                    replace(input, sample=deepcopy(input.sample))
                )
                if task.cancelling():
                    raise asyncio.CancelledError
                reason = _failure_reason(output)
                if reason is None:
                    for sample in _flatten(output.samples):
                        sample.metadata["rollout_attempts"] = attempt
                    return output
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "RFT trajectory index=%s attempt=%s/%s failed: %s",
                input.sample.index, attempt, attempts, reason,
            )
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        sample.metadata = {
            **(sample.metadata or {}),
            "rft_retry_exhausted": True,
            "rollout_attempts": attempts,
            "rollout_failure": reason,
        }
        return GenerateFnOutput(samples=sample)
    finally:
        tasks.discard(task)


def _add_arguments(parser):
    agentic_tool_call.generate.add_arguments(parser)
    parser.add_argument("--rft-rollout-max-retries", type=int, default=3)
    parser.add_argument("--rft-max-epochs", type=int, default=1)


generate.add_arguments = _add_arguments


def require_complete_group(args, samples, **kwargs):
    for sample in _flatten(samples):
        if sample.status == Sample.Status.ABORTED:
            metadata = sample.metadata or {}
            raise RuntimeError(
                f"RFT batch failed at trajectory {sample.index}: "
                f"{metadata.get('rollout_failure', 'aborted trajectory')}; "
                f"attempts={metadata.get('rollout_attempts')}. No prompt replacement."
            )
    return DynamicFilterOutput(keep=True)


class _EpochBoundedSource:
    def __init__(self, source, max_epochs):
        self.source = source
        self.max_epochs = max_epochs

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_samples(self, count):
        size = len(self.source.dataset)
        consumed = self.source.epoch_id * size + self.source.sample_offset
        if consumed + count > self.max_epochs * size:
            raise RuntimeError("RFT epoch limit reached; refusing to cycle or replace prompt rows")
        return self.source.get_samples(count)


class RolloutFn(MathRolloutFn):
    def __init__(self, input):
        args = input.args
        if args.rft_rollout_max_retries < 0 or args.rft_max_epochs < 1:
            raise ValueError("RFT retries must be nonnegative and max epochs must be positive")
        source = _EpochBoundedSource(input.data_source, args.rft_max_epochs)
        if args.num_rollout * args.rollout_batch_size > len(source.dataset) * args.rft_max_epochs:
            raise ValueError("Requested RFT steps exceed the prompt-row epoch limit")
        super().__init__(replace(input, data_source=source))

    async def _call_train(self, input):
        try:
            return await super()._call_train(input)
        except BaseException:
            self.state.aborted = True
            tasks = list(getattr(self.state, "_rft_generation_tasks", ()))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            from rft_agent_function import abort

            await abort(self.state.args)
            raise
