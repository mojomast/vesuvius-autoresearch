# Weights

Model weights are not stored in git.

If a release or prize submission requires weights, publish them as external assets and document:

- download URL
- file name and checksum
- source commit hash
- training config path
- training data provenance and held-out segments
- intended inference command
- license or usage restriction for the weights

Weights trained on Vesuvius Challenge data may inherit data-use constraints. Check the Scroll Prize data terms before redistribution.

## Manifest Template

```yaml
weight_name: <model-name>
file_name: <weights-file>
sha256: <sha256>
source_commit: <git-sha>
training_config: configs/<config>.yaml
training_data:
  train_segments: ["<segment_id>"]
  heldout_segments: ["<segment_id>"]
  no_overlap_statement: "Held-out prediction/evaluation segments were excluded from training labels, crops, mined negatives, and manual review."
intended_inference_command: >-
  .venv/bin/python scripts/infer_full_tile.py --artifact experiments/runs/<run_id>
license_or_terms: "Code is MIT; weights may be constrained by Vesuvius/Scroll Prize data terms."
download_url: <release-or-archive-url>
```
