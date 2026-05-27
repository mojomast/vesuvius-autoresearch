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
