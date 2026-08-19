# Results

Execution logs and audit artifacts for the seven experiments reported in the
thesis. Every figure quoted in Chapters 6 and 8 and in Appendix A can be traced
to a file in this directory.

Each phase of the pipeline writes its own output directory, and that structure
is preserved here.

---

## Layout

```
results/
├── synthetic/
│   ├── seed42_p050/          # Run 1
│   ├── seed42_p070/          # Run 2
│   ├── seed42_p085/          # Run 3
│   ├── seed123_p085/         # Run 4
│   └── seed456_p085/         # Run 5
├── oulad/
│   ├── oulad_phase1/ ... oulad_phase5/
└── xapi/
    └── xapi_phase1/  ... xapi_phase5/
```

Run numbering for the synthetic experiments follows Table 6.1 of the thesis,
which orders the runs by corruption severity rather than by execution order.

| Thesis run | Seed | Bias probability | Directory |
|---|---|---|---|
| Run 1 | 42 | 0.50 | `synthetic/seed42_p050/` |
| Run 2 | 42 | 0.70 | `synthetic/seed42_p070/` |
| Run 3 | 42 | 0.85 | `synthetic/seed42_p085/` |
| Run 4 | 123 | 0.85 | `synthetic/seed123_p085/` |
| Run 5 | 456 | 0.85 | `synthetic/seed456_p085/` |

---

## What each phase writes

| Phase | Principal artifacts |
|---|---|
| 1 | Baseline checkpoint, data partitions, preprocessing objects; for the synthetic runs, the corruption summary and the clean oracle reference |
| 2 | `influence_scores.csv`, `influence_detection_metrics.csv`, `fairness_probe_gradient.pt`, `influence_fairness_direction.pt`, `influence_run_metadata.json` |
| 3 | Parameter-space direction summary, candidate repair paths, phase-consistency check |
| 4 | Frozen repaired checkpoints, `repair_selection_metadata.json` |
| 5 | `final_test_metrics.csv`, `final_test_bootstrap_intervals.csv`, `final_test_pairwise_changes.csv`, `final_test_group_confusions.csv`, `final_test_predictions.csv`, `final_test_audit_metadata.json` |

The untouched test partition is read only in Phase 5.

---

## Where the thesis values come from

### Synthetic benchmark

| Thesis element | Source |
|---|---|
| Table 6.1 — configurations and biased baselines | Corruption summary printed in Phase 1; `biased_baseline` row of the Phase-5 `final_test_metrics.csv` |
| Table 6.2 — detection quality | `influence_detection_metrics.csv` (Phase 2) |
| Table 6.3 — selected repairs | Phase-4 selection table |
| Table 6.4 — final untouched-test audit | Phase-5 `final_test_metrics.csv` and `final_test_pairwise_changes.csv` |
| Table 6.5 — direction geometry | Parameter-space direction summary (Phase 3); the cosine similarity between the two oracle-free directions is computed from the reported direction norms together with the inner product of each direction with the fairness gradient |
| §6.3.4 — deletion baseline changes no predictions | `deletion_baseline` row of `final_test_pairwise_changes.csv`, all five runs |

### Real-data experiments

| Thesis element | Source |
|---|---|
| §6.4, §6.5 — point estimates | `{oulad,xapi}_phase5/final_test_metrics.csv` |
| §6.4, §6.5 — selected fraction and constraint slack | `{oulad,xapi}_phase4/repair_selection_metadata.json` |
| §6.4, §6.5 — prediction changes | `{oulad,xapi}_phase5/final_test_pairwise_changes.csv` |
| Appendix A, Tables A.1 and A.2 | `{oulad,xapi}_phase5/final_test_bootstrap_intervals.csv` |
| §6.4, §6.5 — cosine similarity and gradient norms | `{oulad,xapi}_phase2/influence_run_metadata.json` |

The finding that the utility constraint was not binding at the selected
fraction — reported for both real datasets — is visible in the Phase-4
metadata, which records the baseline validation metrics, the permitted floor,
and the metrics attained at the selected fraction.

---

## Reproducibility

These runs were executed on an Apple M1 with 16 GB unified memory using the
PyTorch Metal Performance Shaders backend, which is not bitwise deterministic.
Re-executing an identical configuration, whether on this hardware or on
different hardware, may produce different numerical results. Section 6.7 of the
thesis discusses the consequences; the synthetic experiments at high corruption
severity are particularly sensitive to this, with threshold-level outcomes
varying substantially across random seeds.

The artifacts in this directory are therefore the record of the specific
executions reported in the thesis, not a prediction of what a repetition will
produce.

---

## Model checkpoints

Checkpoints for the tabular experiments are small and are included in the
Phase-1 and Phase-4 directories. The synthetic experiments fine-tune
`bert-base-uncased`, so each of those runs produces several hundred megabytes
of weights.

<!--
Choose one and delete the rest:

  * The synthetic checkpoints are archived at <DOI or URL>.
  * The synthetic checkpoints are available on request.
  * The synthetic checkpoints are retained locally and not published.
-->