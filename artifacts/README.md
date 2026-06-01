# Artifacts

Generated artifacts are not stored in git.

Use this directory only for small manifests that describe externally stored outputs, such as GitHub Release assets, Zenodo archives, S3 objects, or Hugging Face files.

Each artifact manifest should include:

- source commit hash
- config path and config hash
- command line
- input data source and segment IDs
- validation or prediction region
- output file names and checksums
- model weight checksum, if applicable
- statement that no training data overlaps the submitted prediction region

Do not place `*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, full-tile outputs, or decoded images in git.

## Manifest Template

```yaml
artifact_name: example-full-tile-metrics
source_commit: <git-sha>
config_path: configs/<config>.yaml
config_sha256: <sha256>
command: >-
  .venv/bin/python scripts/infer_full_tile.py --artifact experiments/runs/<run_id>
  --segment-id <segment_id> --output-dir experiments/runs/<run_id>/full_tile_<segment_id>
input_data:
  source: scrollprize_public_segment_zarr
  segment_ids: ["<segment_id>"]
  prepared_npz_metadata: data/<path>.metadata.json
validation_region: whole_segment
outputs:
  - file: metrics.json
    sha256: <sha256>
  - file: metrics_by_threshold.csv
    sha256: <sha256>
no_overlap_statement: "No training labels, crops, mined negatives, or manual review regions overlap the held-out prediction/evaluation segment."
external_url: <release-or-archive-url>
```
