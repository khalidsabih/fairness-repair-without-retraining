# Results

Execution logs and audit artifacts for the seven experiments reported in the
thesis: five synthetic runs, OULAD, and xAPI-Edu-Data. Every value quoted in
Chapter 4 and in Appendix A can be traced to a file in this directory.

The two record types differ. The synthetic runs are preserved as complete
console logs, one per run, which contain the printed metric tables for every
phase. The two tabular pipelines wrote per-phase artifact directories, and that
structure is preserved as it stood.

---

## Layout

```
results/
├── synthetic/
│   ├── seed42_p050.txt       # Run 1
│   ├── seed42_p070.txt       # Run 2
│   ├── seed42_p085.txt       # Run 3
│   ├── seed123_p085.txt      # Run 4
│   └── seed456_p085.txt      # Run 5
├── oulad/
│   ├── run.txt               # console log
│   └── oulad_phase1/ ... oulad_phase5/
└── xapi/
    ├── run.txt               # console log
    └── xapi_phase1/  ... xapi_phase5/
```

A few Phase-4 and Phase-5 files are also duplicated at the top of `oulad/`;
the copies inside the phase directories are the ones referenced below.

Run numbering for the synthetic experiments follows Table 4.1 of the thesis,
which orders the runs by corruption severity rather than by execution order.

| Thesis run | Seed | Bias probability | Directory |
|---|---|---|---|
| Run 1 | 42 | 0.50 | `synthetic/seed42_p050.txt` |
| Run 2 | 42 | 0.70 | `synthetic/seed42_p070.txt` |
| Run 3 | 42 | 0.85 | `synthetic/seed42_p085.txt` |
| Run 4 | 123 | 0.85 | `synthetic/seed123_p085.txt` |
| Run 5 | 456 | 0.85 | `synthetic/seed456_p085.txt` |

---

## What each phase writes

File names are those of the two tabular pipelines. The synthetic pipeline
writes an equivalent set under `data/` with its own names --- for instance
`influence_detection_metrics.csv` for the detection quality, and
`influence_fairness_direction.pt` for the repair direction. Those files are not
committed here, so for the synthetic runs the console log is the artifact.

| Phase | Principal artifacts |
|---|---|
| 1 | Baseline checkpoint, data partitions, preprocessing objects; for the synthetic runs, the corruption summary and the clean oracle reference |
| 2 | `influence_scores.csv`, `influence_summary.csv`, `influence_summary_by_group.csv`, `topk_influential_samples.csv`, `fairness_probe_gradient.pt`, `fairness_unlearning_directions.pt`, `influence_run_metadata.json` |
| 3 | `direction_summary.csv`, `direction_path_metrics.csv`, `direction_functional_diagnostics.csv`, `native_path_best_probe_points.csv`, `direction_comparison_metadata.json` |
| 4 | `repair_candidates.csv`, `repair_selection.csv`, `repair_selection_metadata.json`, `selected_repair_updates.pt` |
| 5 | `final_test_metrics.csv`, `final_test_bootstrap_intervals.csv`, `final_test_pairwise_changes.csv`, `final_test_group_confusions.csv`, `final_test_predictions.csv`, `final_test_audit_metadata.json` |

The untouched test partition is read only in Phase 5.

---

## Where the thesis values come from

### Synthetic benchmark

Every value below is in the per-run console log, which prints each phase's
metric table in full.

| Thesis element | Source |
|---|---|
| Table 4.1 — configurations and biased baselines | Corruption summary printed in Phase 1; `biased_baseline` row of the Phase-5 final metrics table |
| Table 4.2 — detection quality | Phase-2 detection table (`average_precision`, `roc_auc`, `precision_at_50`) |
| Table 4.3 — selected repairs | Phase-4 selection table |
| Table 4.4 — final untouched-test audit | Phase-5 final metrics table and pairwise-changes table |
| Table 4.5 — direction geometry | Phase-3 direction summary (`direction_norm`, `dot_with_fairness_gradient`); the cosine similarity between the two candidate repair directions is not printed and is computed from those two columns |
| §4.3.3 — deletion baseline changes no predictions | `deletion_baseline` row of the Phase-5 pairwise-changes table, all five runs |

### Real-data experiments

| Thesis element | Source |
|---|---|
| §4.4.3 and §4.4.4 — point estimates | `{oulad,xapi}_phase5/final_test_metrics.csv` |
| §4.4.3 and §4.4.4 — selected fraction and constraint slack | `{oulad,xapi}_phase4/repair_selection_metadata.json` |
| §4.4.3 and §4.4.4 — prediction changes | `{oulad,xapi}_phase5/final_test_pairwise_changes.csv` |
| §4.4.4 — fairness-probe cell counts | `xapi_phase1/fairness_probe.csv` |
| Appendix A, Tables A.1 and A.2 | `{oulad,xapi}_phase5/final_test_bootstrap_intervals.csv` |
| §4.5.1 — direction norms and gradient inner products | `{oulad,xapi}_phase2/influence_run_metadata.json` |

The finding that the utility constraint was not binding at the selected
fraction — reported for both real datasets — is visible in the Phase-4
metadata, which records the baseline validation metrics, the permitted floor,
and the metrics attained at the selected fraction.

---

## Reproducibility

These runs were executed on an Apple M1 with 16 GB unified memory using the
PyTorch Metal Performance Shaders backend, which is not bitwise deterministic.
Re-executing an identical configuration, whether on this hardware or on
different hardware, may produce different numerical results. Section 4.8 of the
thesis discusses the consequences; the synthetic experiments at high corruption
severity are particularly sensitive to this, with threshold-level outcomes
varying substantially across random seeds.

The artifacts in this directory are therefore the record of the specific
executions reported in the thesis, not a prediction of what a repetition will
produce.

---

## Model checkpoints

Checkpoints for the tabular experiments are small and are included in the
Phase-1 and Phase-4 directories: `m0_oulad.pt` and `m0_xapi.pt` for the
baselines, `selected_repair_updates.pt` for the selected repairs, and
`preprocessor.joblib` for the fitted preprocessing.
