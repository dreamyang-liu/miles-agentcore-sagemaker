"""CPU scheduler checks; runtime tensor/loss checks also run in the pinned image."""

import random
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from bshd_token_batching import batch_widths, make_batches, split_to_count


class BudgetTests(unittest.TestCase):
    def check_schedule(self, lengths, budget=20000, multiple=512, count=None):
        batches = make_batches(lengths, budget, multiple)
        if count is not None:
            batches = split_to_count(batches, count)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(lengths))))
        widths = batch_widths(lengths, batches, multiple)
        for batch in batches:
            self.assertLessEqual(len(batch) * widths[batch[0]], budget)
            self.assertTrue(all(widths[i] >= lengths[i] for i in batch))
            self.assertEqual(len({widths[i] for i in batch}), 1)
        return batches, widths

    def test_long_outlier_does_not_pad_short_microbatches(self):
        lengths = [1400] * 127 + [11529]
        batches, widths = self.check_schedule(lengths)
        self.assertEqual(next(batch for batch in batches if 127 in batch), [127])
        self.assertEqual(widths[127], 11776)
        self.assertTrue(all(width == 1536 for width in widths[:127]))

    def test_5000_token_rows_account_for_padding(self):
        batches, _ = self.check_schedule([5000] * 128)
        self.assertEqual(max(map(len, batches)), 3)

    def test_dp_alignment_preserves_budget_and_all_rows(self):
        shards = [[1400] * 128, [5000] * 127 + [11529]]
        count = max(len(make_batches(shard, 20000, 512)) for shard in shards)
        for shard in shards:
            batches, _ = self.check_schedule(shard, count=count)
            self.assertEqual(len(batches), count)

    def test_single_oversized_sequence_is_not_silently_admitted(self):
        with self.assertRaisesRegex(ValueError, "No sample was truncated or dropped"):
            make_batches([1400, 20000], 20000, 512)
        self.check_schedule([19968])

    def test_many_length_distributions_and_split_counts(self):
        rng = random.Random(123)
        for _ in range(50):
            lengths = [rng.randint(1, 19968) for _ in range(128)]
            minimum = len(make_batches(lengths, 20000, 512))
            self.check_schedule(lengths, count=rng.randint(minimum, len(lengths)))

    def test_missing_or_duplicate_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "omitted"):
            batch_widths([100, 200], [[0]], 512)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            batch_widths([100, 200], [[0], [0, 1]], 512)


@unittest.skipUnless(find_spec("torch"), "Runtime checks require the pinned Miles image")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from miles.backends.training_utils import cp_utils, data, log_utils, loss
        cls.torch, cls.cp_utils, cls.data, cls.log_utils, cls.loss = torch, cp_utils, data, log_utils, loss

    def setUp(self):
        group = SimpleNamespace(size=2, rank=0, group="test-dp")
        self.parallel = SimpleNamespace(
            tp=SimpleNamespace(size=1), pp=SimpleNamespace(size=1), vpp_size=1,
            cp=SimpleNamespace(size=1, rank=0), effective_dp=group,
            intra_dp=group, intra_dp_cp=group, is_ulysses_cp=False,
        )
        self.args = SimpleNamespace(
            qkv_format="bshd", train_backend="megatron", use_dynamic_batch_size=True,
            max_tokens_per_gpu=48, global_batch_size=8, data_pad_size_multiplier=4,
            use_dynamic_global_batch_size=False, compress_ratios=None,
            calculate_per_token_loss=False, recompute_loss_function=False,
            true_on_policy_mode=False, allgather_cp=False,
        )
        for module in (self.data, self.cp_utils, self.loss):
            patcher = patch.object(module, "get_parallel_state", return_value=self.parallel)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(self.torch.cuda, "current_device", return_value="cpu")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_real_batch_builder_preserves_tokens_masks_and_logprob_order(self):
        from bshd_token_batching import get_data_iterator
        from miles.utils.ft_utils.process_group_utils import GeneralPGUtil

        lengths = [11, 29, 45, 20]
        tokens = [self.torch.arange(n) + 1000 * (i + 1) for i, n in enumerate(lengths)]
        masks = [self.torch.tensor([1, 0, 1, 1]) for _ in lengths]
        rollout = {
            "tokens": tokens, "total_lengths": lengths,
            "response_lengths": [4] * 4, "loss_masks": masks, "max_seq_lens": [48] * 4,
            # Real rollout-manager output includes a precomputed fixed-size
            # schedule; the extension must replace it, not trust or reject it.
            "micro_batch_indices": [[0, 1, 2, 3]],
            "num_microbatches": [1], "num_rollouts": [8],
        }
        with patch.object(GeneralPGUtil, "create", return_value=SimpleNamespace(all_reduce=lambda *a, **k: None)):
            iterators, counts = get_data_iterator(self.args, None, rollout)
        iterator = iterators[0]
        results = []
        for indices in iterator.micro_batch_indices:
            batch = self.data.get_batch(
                iterator, ["tokens", "total_lengths", "response_lengths", "loss_masks", "max_seq_lens"],
                pad_multiplier=4, qkv_format="bshd",
            )
            self.assertLessEqual(batch["tokens"].numel(), 48)
            for row, index in enumerate(indices):
                length = lengths[index]
                self.torch.testing.assert_close(batch["tokens"][row, :length], tokens[index])
                self.assertEqual(batch["tokens"][row, length:].count_nonzero().item(), 0)
                expected_mask = self.torch.zeros(batch["tokens"].shape[1], dtype=self.torch.int64)
                expected_mask[length - 5:length - 1] = masks[index]
                self.torch.testing.assert_close(batch["full_loss_masks"][row], expected_mask)
            results.append({"log_probs": [f"sample-{i}" for i in indices]})
        restored = self.log_utils.aggregate_forward_results(results, iterator, self.args)
        self.assertEqual(restored["log_probs"], [f"sample-{i}" for i in range(4)])
        self.assertEqual(sum(counts), len(iterator.micro_batch_indices))
        self.assertEqual(rollout["micro_batch_indices"], iterator.micro_batch_indices)
        self.assertEqual(rollout["num_microbatches"], counts)
        self.assertGreater(counts[0], 1)
        iterator.reset()
        self.assertEqual(iterator.offset, 0)

    def test_remote_oversized_sample_fails_before_scheduling(self):
        from bshd_token_batching import get_data_iterator
        from miles.utils.ft_utils.process_group_utils import GeneralPGUtil

        rollout = {"tokens": [self.torch.ones(8)] * 4, "total_lengths": [8] * 4}
        reducer = SimpleNamespace(all_reduce=lambda tensor, *a, **k: tensor.fill_(1))
        with patch.object(GeneralPGUtil, "create", return_value=reducer):
            with self.assertRaisesRegex(ValueError, "No sample was truncated or dropped"):
                get_data_iterator(self.args, None, rollout)

    def test_real_miles_loss_scaling_preserves_gradients_across_batch_shapes(self):
        def gradient(schedule):
            theta = self.torch.tensor(0.3, dtype=self.torch.float64, requires_grad=True)
            for indices in schedule:
                response_lengths = [i % 3 + 2 for i in indices]
                batch = {
                    "total_lengths": [n + 3 for n in response_lengths],
                    "response_lengths": response_lengths,
                    "loss_masks": [self.torch.ones(n, dtype=self.torch.float64) for n in response_lengths],
                    "max_seq_lens": [8] * len(indices),
                }
                logits = self.torch.cat([
                    theta * (i + 1) * self.torch.ones(n, dtype=self.torch.float64)
                    for i, n in zip(indices, response_lengths, strict=True)
                ])
                loss_fn = lambda args, batch, logits, reduce: (reduce(logits.square()), {})
                with patch.object(self.loss, "get_loss_function", return_value=loss_fn):
                    value, _, _ = self.loss.loss_function(
                        self.args, batch, len(schedule), logits, apply_megatron_loss_scaling=True,
                    )
                # Megatron accumulation divides by microbatch count; DP averages
                # across ranks. Both must cancel the wrapper's corresponding factors.
                (value / len(schedule) / self.parallel.intra_dp.size).backward()
            return theta.grad

        fixed = [[0, 1], [2, 3], [4, 5], [6, 7]]
        variable = [[7], [4, 0, 1], [2, 3, 5, 6]]
        self.torch.testing.assert_close(gradient(fixed), gradient(variable), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
