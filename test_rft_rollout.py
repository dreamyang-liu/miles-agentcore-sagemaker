"""Run inside the pinned Miles image: python -m unittest -v test_rft_rollout."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

import rft_rollout as rollout


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def make_input(self):
        sample = Sample(index=7, prompt=[{"role": "user", "content": "2+2?"}], metadata={"answer": "4"})
        state = SimpleNamespace(args=SimpleNamespace(rft_rollout_max_retries=3), aborted=False)
        return GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)

    async def test_fresh_sample_on_retry_and_fourth_attempt_succeeds(self):
        input = self.make_input()
        seen = []

        async def fake_generate(attempt_input):
            seen.append(attempt_input.sample)
            self.assertNotIn("dirty", attempt_input.sample.metadata)
            attempt_input.sample.metadata["dirty"] = True
            attempt_input.sample.status = Sample.Status.ABORTED if len(seen) < 4 else Sample.Status.COMPLETED
            attempt_input.sample.metadata["exit_status"] = "submitted"
            return GenerateFnOutput(samples=attempt_input.sample)

        with patch.object(rollout.agentic_tool_call, "generate", fake_generate):
            output = await rollout.generate(input)
        self.assertEqual(output.samples.metadata["rollout_attempts"], 4)
        self.assertEqual(len({id(x) for x in seen}), 4)
        self.assertNotIn("dirty", input.sample.metadata)

    async def test_exhaustion_fails_group_instead_of_replacing_prompt(self):
        input = self.make_input()

        async def failure(attempt_input):
            raise ConnectionError("callback unavailable")

        with patch.object(rollout.agentic_tool_call, "generate", AsyncMock(side_effect=failure)) as generate:
            output = await rollout.generate(input)
        self.assertEqual(generate.await_count, 4)
        with self.assertRaisesRegex(RuntimeError, "No prompt replacement"):
            rollout.require_complete_group(input.args, [output.samples])

    async def test_no_answer_is_valid_zero_reward_trajectory(self):
        input = self.make_input()
        sample = Sample(status=Sample.Status.COMPLETED, metadata={"exit_status": "stopped_without_submitting"})
        with patch.object(
            rollout.agentic_tool_call, "generate", AsyncMock(return_value=GenerateFnOutput(samples=sample))
        ) as generate:
            output = await rollout.generate(input)
        self.assertEqual(generate.await_count, 1)
        self.assertIsNone(rollout._failure_reason(output))

    async def test_cancellation_does_not_start_another_attempt(self):
        input = self.make_input()
        entered = asyncio.Event()

        async def wait(_):
            entered.set()
            await asyncio.Future()

        with patch.object(rollout.agentic_tool_call, "generate", AsyncMock(side_effect=wait)) as generate:
            task = asyncio.create_task(rollout.generate(input))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(generate.await_count, 1)
        self.assertFalse(input.state._rft_generation_tasks)

    async def test_actual_miles_sampler_propagates_exhausted_group(self):
        from miles.rollout.inference_rollout import inference_rollout_train as driver

        args = SimpleNamespace(
            rollout_global_dataset=True,
            dynamic_sampling_filter_path="rft_rollout.require_complete_group",
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            rollout_submission_granularity=None,
            over_sampling_batch_size=1,
        )
        failed = Sample(index=3, status=Sample.Status.ABORTED, metadata={
            "rollout_failure": "network failed", "rollout_attempts": 4,
        })
        source = unittest.mock.Mock(return_value=[[failed]])

        async def complete_failed_group(*_args, **_kwargs):
            return [failed]

        with patch.object(driver.dumper_utils, "configure_sglang", AsyncMock()), \
             patch.object(driver, "generate_and_rm_group", complete_failed_group), \
             patch.object(driver, "sample_text_preview", return_value="failed"), \
             patch.object(driver, "reward_log_summary", return_value="none"):
            with self.assertRaisesRegex(RuntimeError, "No prompt replacement"):
                await asyncio.wait_for(
                    driver.generate_rollout_async(SimpleNamespace(args=args, sampling_params={}), 0, source),
                    timeout=2,
                )
        source.assert_called_once_with(1)


class EpochTests(unittest.TestCase):
    def test_refuses_crossing_dataset_boundary(self):
        source = SimpleNamespace(dataset=list(range(2560)), epoch_id=0, sample_offset=2540)
        source.get_samples = unittest.mock.Mock()
        bounded = rollout._EpochBoundedSource(source, 1)
        with self.assertRaisesRegex(RuntimeError, "epoch limit"):
            bounded.get_samples(32)
        source.get_samples.assert_not_called()


if __name__ == "__main__":
    unittest.main()
