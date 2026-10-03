# Research reports

- [Current model assessment](current_model_assessment.md): the primary eight-seed
  assessment of LightGBM, learned fusion, selective recovery and anomaly baselines.
- [Validated binary checkpoint](binary_validated_result.md): the earlier frozen
  binary selection result, with its own evaluation protocol and historical comparisons.

Canonical raw evidence stays local and Git-ignored:

| Directory under `artifacts/` | Purpose |
| --- | --- |
| `anomaly_comparison_v1/` | Original three-seed fitted models and split provenance |
| `anomaly_comparison_v2/` | Same three seeds plus selective recovery |
| `fusion_extra_seeds_v2/` | Five additional seeds |
| `fusion_eight_seed_statistics_v2/` | Combined 960-row comparison and statistics |
| `fusion_report_v1/`, `fusion_report_v2/` | Historical three-seed visualizations; not eight-seed plots |
| `archive/cleanup_20261003/` | Older generated reports retained locally for reference |

Duplicate single-run/subset outputs and smoke artifacts were removed after
checking their reported metrics against the canonical suite. Do not remove the
remaining model bundles, split indices, manifests or calibration files: they
are needed to reproduce and audit the current findings.

To regenerate figures for all eight seeds, use the reporting script with
`--input-dirs artifacts/anomaly_comparison_v2 artifacts/fusion_extra_seeds_v2`
and a new `--output-dir`, plus `--plots --reconnaissance`. See the
[command guide](../training/FUSION_EXPERIMENTS.md).
