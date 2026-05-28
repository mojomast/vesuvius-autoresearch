from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.jobs import enqueue_experiment_config, get_job, init_db, lease_one_job, run_leased_job, run_worker


class JobsTest(unittest.TestCase):
    def test_init_is_idempotent_and_enqueue_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            init_db(db_path)
            init_db(db_path)

            first = enqueue_experiment_config("configs/a.yaml", db_path, dedupe_key="same")
            second = enqueue_experiment_config("configs/a.yaml", db_path, dedupe_key="same")
            third = enqueue_experiment_config("configs/a.yaml", db_path)

            self.assertEqual(first["id"], second["id"])
            self.assertNotEqual(first["id"], third["id"])
            self.assertEqual(first["payload"]["experiment_config"], "configs/a.yaml")
            self.assertEqual(first["status"], "queued")

    def test_leases_one_job_and_recovers_stale_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            stale = enqueue_experiment_config("configs/stale.yaml", db_path)
            fresh = enqueue_experiment_config("configs/fresh.yaml", db_path)

            leased = lease_one_job("worker-a", -1, db_path)
            self.assertEqual(leased["id"], stale["id"])
            self.assertEqual(leased["worker_id"], "worker-a")

            recovered = lease_one_job("worker-b", 60, db_path)
            self.assertEqual(recovered["id"], stale["id"])
            self.assertEqual(recovered["worker_id"], "worker-b")

            next_job = lease_one_job("worker-c", 60, db_path)
            self.assertEqual(next_job["id"], fresh["id"])

    def test_run_leased_job_calls_runner_and_records_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            experiments_db = Path(tmp) / "experiments.db"
            enqueued = enqueue_experiment_config("configs/success.yaml", db_path)
            leased = lease_one_job("worker-a", 60, db_path)

            with patch("experiments.jobs.run_experiment", return_value={"run_id": "run-1", "main_metric": 0.25}) as mock_run:
                updated = run_leased_job(leased, db_path, experiments_db)

            mock_run.assert_called_once_with("configs/success.yaml", db_path=experiments_db)
            self.assertEqual(updated["id"], enqueued["id"])
            self.assertEqual(updated["status"], "succeeded")
            self.assertEqual(json.loads(updated["result_json"])["run_id"], "run-1")
            self.assertIsNone(updated["error"])

    def test_run_leased_job_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            enqueue_experiment_config("configs/fail.yaml", db_path)
            leased = lease_one_job("worker-a", 60, db_path)

            with patch("experiments.jobs.run_experiment", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    run_leased_job(leased, db_path)

            failed = get_job(leased["id"], db_path)
            self.assertEqual(failed["status"], "failed")
            self.assertIn("boom", failed["error"])
            self.assertIsNone(failed["result_json"])

    def test_worker_respects_once_and_max_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            first = enqueue_experiment_config("configs/one.yaml", db_path)
            second = enqueue_experiment_config("configs/two.yaml", db_path)

            with patch("experiments.jobs.run_experiment", return_value={"ok": True}) as mock_run:
                ran = run_worker(db_path=db_path, worker_id="worker-a", lease_seconds=60, once=True, max_jobs=2)

            self.assertEqual(ran, 1)
            self.assertEqual(mock_run.call_count, 1)
            self.assertEqual(get_job(first["id"], db_path)["status"], "succeeded")
            self.assertEqual(get_job(second["id"], db_path)["status"], "queued")

    def test_worker_records_failure_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            failed = enqueue_experiment_config("configs/fail.yaml", db_path)
            succeeded = enqueue_experiment_config("configs/succeed.yaml", db_path)

            def fake_run(config_path: str, **_kwargs: object) -> dict[str, bool]:
                if "fail" in config_path:
                    raise RuntimeError("boom")
                return {"ok": True}

            with patch("experiments.jobs.run_experiment", side_effect=fake_run):
                ran = run_worker(db_path=db_path, worker_id="worker-a", lease_seconds=60, max_jobs=2)

            self.assertEqual(ran, 2)
            self.assertEqual(get_job(failed["id"], db_path)["status"], "failed")
            self.assertIn("boom", get_job(failed["id"], db_path)["error"])
            self.assertEqual(get_job(succeeded["id"], db_path)["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
