#!/usr/bin/env python3
"""Unit contracts for checkpointed cardiac RPR+."""

from __future__ import annotations

import unittest

import torch

from apsa.cardiac import apsa_mms as runner


def state(value: float):
    return {"model": {"weight": torch.tensor([value])}, "optimizer": {"step": value}}


class RPRCheckpointControllerTests(unittest.TestCase):
    def test_reject_discards_uncommitted_safe_updates(self):
        controller = runner.RPRCheckpointController(interval=3)
        initial = state(0.0)
        controller.ensure_initialized(initial)
        self.assertFalse(controller.accept(state(1.0)))
        self.assertFalse(controller.accept(state(2.0)))
        restored, span = controller.reject(state(2.0))
        self.assertIs(restored, initial)
        self.assertEqual(span, 3)
        self.assertEqual(controller.safe_streak, 0)
        self.assertEqual(controller.rollback_span_total, 3)

    def test_completed_safe_interval_advances_trusted_state(self):
        controller = runner.RPRCheckpointController(interval=2)
        initial = state(0.0)
        committed = state(2.0)
        controller.ensure_initialized(initial)
        self.assertFalse(controller.accept(state(1.0)))
        self.assertTrue(controller.accept(committed))
        restored, span = controller.reject(committed)
        self.assertIs(restored, committed)
        self.assertEqual(span, 1)
        self.assertEqual(controller.commit_count, 1)
        self.assertEqual(controller.rollback_count, 1)

    def test_interval_one_is_reserved_for_ordinary_preupdate_rpr(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            runner.RPRCheckpointController(interval=1)


if __name__ == "__main__":
    unittest.main()
