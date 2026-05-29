# Research Harness

`ResearchHarness` defines the minimal lifecycle for an autoresearch loop: propose an experiment, evaluate it, decide whether to promote it, and handle promotion side effects.

`VesuviusHarness` is a thin adapter over the existing `autoresearch.py` functions. New ScrollPrize harnesses should subclass `ResearchHarness`, define typed history/result conversion, and keep promotion gates explicit so dashboard and cron automation can inspect decisions.

Warning: AutoResearch initializes the harness before pruning configs or planning work. A broken harness import prevents AutoResearch from running at all; test harness imports before deploying changes.
