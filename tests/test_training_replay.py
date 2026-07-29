from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs
from humanoidverse.agents.fb_cpr.agent import transition_discount


class TrainingReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = TrajectoryDictBufferMultiDim(
            capacity=8,
            device="cpu",
            n_dim=2,
            end_key="done",
            output_key_t=[
                "observation",
                "action",
                "reward",
                "terminated",
                "truncated",
                "done",
            ],
            output_key_tp1=["observation"],
        )
        # Avoid compiling this small deterministic unit-test workload.
        self.buffer._get_idxs = get_idxs

    def _extend(self, observations: tuple[float, float], done: bool) -> None:
        flags = np.full((1, 2, 1), done, dtype=bool)
        values = np.asarray(observations, dtype=np.float32).reshape(1, 2, 1)
        self.buffer.extend(
            {
                "observation": {"state": values},
                "action": np.zeros((1, 2, 1), dtype=np.float32),
                "reward": np.ones((1, 2, 1), dtype=np.float32),
                "terminated": flags,
                "truncated": np.zeros_like(flags),
                "done": flags,
            }
        )

    def test_autoreset_boundary_is_not_sampled_as_a_transition(self) -> None:
        self._extend((0.0, 10.0), done=False)
        self.assertFalse(self.buffer.can_sample(batch_size=8))

        # These observations are the last valid states. Their actions lead to an
        # autoreset whose real final observations are unavailable, so they are endpoints.
        self._extend((1.0, 11.0), done=True)
        self.assertTrue(self.buffer.can_sample(batch_size=8))

        self._extend((100.0, 110.0), done=False)
        self._extend((101.0, 111.0), done=False)
        batch = self.buffer.sample(batch_size=256)

        current = batch["observation"]["state"].squeeze(-1)
        following = batch["next"]["observation"]["state"].squeeze(-1)
        sampled_pairs = set(zip(current.tolist(), following.tolist()))
        allowed_pairs = {(0.0, 1.0), (10.0, 11.0), (100.0, 101.0), (110.0, 111.0)}

        self.assertTrue(sampled_pairs)
        self.assertTrue(sampled_pairs.issubset(allowed_pairs))
        self.assertFalse(batch["done"].any())
        self.assertFalse(batch["terminated"].any())

    def test_transition_discount_uses_current_boundary_flags(self) -> None:
        batch = {
            "terminated": torch.tensor([[False], [True], [False]]),
            "truncated": torch.tensor([[False], [False], [True]]),
            "next": {
                # These deliberately disagree to ensure current flags take precedence.
                "terminated": torch.zeros(3, 1, dtype=torch.bool),
            },
        }

        result = transition_discount(batch, discount=0.99, device="cpu")

        torch.testing.assert_close(result, torch.tensor([[0.99], [0.0], [0.0]]))

    def test_sampler_can_select_last_legal_transition(self) -> None:
        with patch("humanoidverse.agents.buffers.trajectory.torch.rand", return_value=torch.tensor([0.75])):
            indices = get_idxs(
                seq_length=1,
                num_slices=1,
                lengths=torch.tensor([3]),
                start_idx=torch.tensor([[0, 0]]),
                storage_length=8,
            )

        self.assertEqual(indices[0, 0].item(), 1)


if __name__ == "__main__":
    unittest.main()
