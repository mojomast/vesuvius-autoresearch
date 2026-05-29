# Residual 2.5D Search Plan

Base config: configs/residual_25d_cpu_safe.yaml

Base val_f1 (historical): 0.4286 from run 20260528T155452Z_8846f887, AP 0.3122, pred/val 2.993. Historical config used max_train_samples=4096; CPU-safe base reduces this to 1024.

Hardware: CPU only; system Python has no torch, venv torch reports cuda=false. Feasible bounds: max_train_samples <= 1024, base_channels <= 8, no 3D conv.

| Config slug | Changed axis | Value | Rationale |
|-------------|-------------|-------|-----------|
| prw010 | positive_rate_loss_weight | 0.10 | Test stronger positive-rate control near the high historical AP run 20260528T195931Z_444e4b86. |
| prw006 | positive_rate_loss_weight | 0.06 | Test whether the CPU-reduced sample budget needs less positive-rate penalty to preserve recall. |
| cap2p5 | max_pred_positive_rate_ratio | 2.5 | Test the lower end of the current 2.5-3.0 cap band while staying above the strict cap-2.0 historical variant. |
| cap2p75 | max_pred_positive_rate_ratio | 2.75 | Test an intermediate positive-rate cap that may reduce inflation without suppressing F1 as much as 2.5. |
| cap3p5 | max_pred_positive_rate_ratio | 3.5 | Test the upper PARAM_BOUNDS cap to see whether sampled F1 is recall-limited on CPU-safe training. |
| tol005 | positive_rate_loss_tolerance | 0.005 | Compare the stricter tolerance from the high-AP cap-2.0 historical run. |
| hn085 | hard_negative_fraction | 0.85 | With patch_sampling=hard_mining, test stronger hard-negative pressure for precision. |

Stopping criteria:
- Stop generating new configs if any run achieves val_f1 >= 0.25 AND pred/val ratio <= 3.0 (proceed to LOO).
- Stop if 3 consecutive runs show val_f1 < 0.10 (architecture may not work in this setup).
- Max configs to run before reassessing: 7
