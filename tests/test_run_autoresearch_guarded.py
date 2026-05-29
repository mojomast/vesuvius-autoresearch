from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_autoresearch_guarded as guard


class RunAutoresearchGuardedTest(unittest.TestCase):
    def test_choose_proposals_honors_explicit_autoresearch_proposals(self) -> None:
        with patch.dict("os.environ", {"AUTORESEARCH_PROPOSALS": "1"}, clear=False):
            self.assertEqual(guard.choose_proposals(load1=0.0, mem_gib=128.0), 1)

    def test_choose_proposals_rejects_invalid_explicit_value(self) -> None:
        with patch.dict("os.environ", {"AUTORESEARCH_PROPOSALS": "nope"}, clear=False):
            self.assertEqual(guard.choose_proposals(load1=0.0, mem_gib=128.0), 0)

    def test_main_passes_explicit_proposals_and_safe_promotion_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "guard.log"
            with patch.object(guard, "GUARD_LOG", log_path), \
                patch.object(guard, "LOGS", Path(tmpdir)), \
                patch.object(guard, "loadavg", return_value=(0.0, 0.0, 0.0)), \
                patch.object(guard, "mem_available_gib", return_value=128.0), \
                patch.object(guard, "root_free_gib", return_value=500.0), \
                patch.object(guard, "active_autoresearch_processes", return_value=[]), \
                patch("scripts.run_autoresearch_guarded.subprocess.run") as run_mock, \
                patch.dict("os.environ", {"AUTORESEARCH_PROPOSALS": "1", "AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION": "1", "AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY": "0"}, clear=False):
                run_mock.return_value.returncode = 0

                self.assertEqual(guard.main(), 0)

        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["AUTORESEARCH_PROPOSALS"], "1")
        self.assertEqual(env["AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION"], "0")
        self.assertEqual(env["AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY"], "1")
        self.assertEqual(env["AUTORESEARCH_AUTO_PROMOTE"], "1")
        self.assertEqual(env["AUTORESEARCH_PROMOTION_TIMEOUT_SECONDS"], "450")

    def test_main_allows_guard_specific_promotion_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "guard.log"
            with patch.object(guard, "GUARD_LOG", log_path), \
                patch.object(guard, "LOGS", Path(tmpdir)), \
                patch.object(guard, "loadavg", return_value=(0.0, 0.0, 0.0)), \
                patch.object(guard, "mem_available_gib", return_value=128.0), \
                patch.object(guard, "root_free_gib", return_value=500.0), \
                patch.object(guard, "active_autoresearch_processes", return_value=[]), \
                patch("scripts.run_autoresearch_guarded.subprocess.run") as run_mock, \
                patch.dict("os.environ", {"AUTORESEARCH_PROPOSALS": "1", "AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION": "1", "SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE": "1"}, clear=False):
                run_mock.return_value.returncode = 0

                self.assertEqual(guard.main(), 0)

        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["AUTORESEARCH_PROPOSALS"], "1")
        self.assertEqual(env["AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION"], "1")
        self.assertEqual(env["AUTORESEARCH_AUTO_PROMOTE"], "1")

    def test_main_uses_adaptive_proposals_when_no_explicit_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "guard.log"
            with patch.object(guard, "GUARD_LOG", log_path), \
                patch.object(guard, "LOGS", Path(tmpdir)), \
                patch.object(guard, "loadavg", return_value=(0.0, 0.0, 0.0)), \
                patch.object(guard, "mem_available_gib", return_value=128.0), \
                patch.object(guard, "root_free_gib", return_value=500.0), \
                patch.object(guard, "active_autoresearch_processes", return_value=[]), \
                patch("scripts.run_autoresearch_guarded.subprocess.run") as run_mock, \
                patch.dict("os.environ", {}, clear=True):
                run_mock.return_value.returncode = 0

                self.assertEqual(guard.main(), 0)

        self.assertEqual(run_mock.call_args.kwargs["env"]["AUTORESEARCH_PROPOSALS"], "4")


if __name__ == "__main__":
    unittest.main()
