# Full-Tile Comparison: 20260529T005819Z_28af43a2

No previous full-tile metrics file was found for `20260528T205650Z_4b8dff1c`. Manual comparison uses the previous best contract-compliant DB row and the candidate full-tile validation-segment metrics.

| Metric | Previous `20260528T205650Z_4b8dff1c` | Candidate `20260529T005819Z_28af43a2` | Delta |
| --- | ---: | ---: | ---: |
| val_f1 | 0.1462 | 0.2187 | +0.0725 |
| AP | 0.1151 | 0.1463 | +0.0311 |
| ap_prevalence_lift | 4.0787 | 1.8166 | -2.2621 |
| pred/val ratio | 1.1056 | 2.9745 | +1.8689 |

The candidate full-tile validation segment improves val_f1 and AP but fails promotion eligibility because its artifact provenance is `spatial-same-segment`.
