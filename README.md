# Repairing Fairness Without Retraining

Implementation accompanying the master's thesis *Repairing Fairness Without
Retraining: An Influence-Guided Approach for Educational Machine Learning*
(Khalid Sabih, Universität Leipzig, 2026).

The code implements a five-phase pipeline that repairs group-fairness
behaviour in a trained classifier through a localized parameter update. The
update is derived from a differentiable Equalized Odds surrogate evaluated on
a held-out fairness probe and preconditioned by an inverse Hessian–vector
product estimated with LiSSA. No retraining, reference model, or counterfactual
dataset is required.

---

## Pipeline

Each phase reads the artifacts written by the previous one and writes its own.
Phases can be re-executed independently.

| Phase | Purpose | Principal outputs |
|-------|---------|-------------------|
| 1 | Data preparation and baseline training | model checkpoint, partitions, preprocessing objects |
| 2 | Fairness gradient, LiSSA inverse-HVP, per-sample influence scores | `influence_scores.csv`, `influence_fairness_direction.pt` |
| 3 | Comparison of candidate repair directions | direction summary, candidate paths |
| 4 | Utility-constrained selection of a repair fraction | frozen repaired checkpoints, `repair_selection_metadata.json` |
| 5 | Single audit on the untouched test partition | `final_test_metrics.csv`, bootstrap intervals, prediction changes |

The untouched test partition is read only in Phase 5.

---

## Repository layout

```
.
├── synthetic/                 # controlled benchmark (BERT, generated essays)
│   ├── 01_bias_injection.py
│   ├── 02_influence_diagnostics_direction.py
│   ├── 03_direction_comparison.py
│   ├── 04_geometric_intervention.py
│   └── 05_model_audit.py
├── oulad/                     # Open University Learning Analytics Dataset
│   ├── 01_oulad_data_and_training.py
│   ├── 02_oulad_influence_diagnostics.py
│   ├── 03_oulad_direction_comparison.py
│   ├── 04_oulad_repair_selection.py
│   └── 05_oulad_final_audit.py
├── xapi/                      # xAPI-Edu-Data
│   ├── 01_xapi_data_and_training.py
│   ├── 02_xapi_influence_diagnostics.py
│   ├── 03_xapi_direction_comparison.py
│   ├── 04_xapi_repair_selection.py
│   └── 05_xapi_final_audit.py
├── results/                   # logs and audit artifacts for the reported runs
│   ├── README.md
│   ├── synthetic/
│   ├── oulad/
│   └── xapi/
├── data/                      # raw datasets (not tracked; see below)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Model checkpoints are written to a directory supplied on the command line and
are not tracked in version control. The execution logs and audit CSVs for the
runs reported in the thesis are in `results/`; see `results/README.md` for the
mapping from files to thesis tables.

---

## Installation

Python 3.10 or later. The reported results were produced on an Apple M1 with
16 GB unified memory using the PyTorch Metal Performance Shaders backend.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The synthetic benchmark downloads `bert-base-uncased` from the Hugging Face
Hub on first run.

Approximate runtimes on the hardware above: a full synthetic run takes about
fifteen minutes, of which Phase 1 (fine-tuning) accounts for roughly five and
Phase 5 (audit with bootstrap resampling) for two. The tabular pipelines
complete in seconds.

---

## Data

Neither dataset is redistributed here.

**OULAD** — download from <https://analyse.kmi.open.ac.uk/open-dataset> and
place `studentInfo.csv` under `data/oulad/`. The experiments use module BBB.

**xAPI-Edu-Data** — download from
<https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data> and place
`xAPI-Edu-Data.csv` under `data/xapi/`.

The synthetic benchmark generates its own data in Phase 1 and requires no
download.

---

## Running the pipeline

### Synthetic benchmark

```bash
cd synthetic
python 01_bias_injection.py --seed 42 --bias-probability 0.85
python 02_influence_diagnostics_direction.py
python 03_direction_comparison.py
python 04_geometric_intervention.py
python 05_model_audit.py
```

The five runs reported in the thesis use seed 42 at bias probabilities 0.50,
0.70 and 0.85, and probability 0.85 at seeds 123 and 456.

### OULAD

```bash
cd oulad
python 01_oulad_data_and_training.py --oulad-dir ../data/oulad --output-dir runs/oulad_phase1
python 02_oulad_influence_diagnostics.py --phase1-dir runs/oulad_phase1 --output-dir runs/oulad_phase2
python 03_oulad_direction_comparison.py --phase1-dir runs/oulad_phase1 --phase2-dir runs/oulad_phase2 --output-dir runs/oulad_phase3
python 04_oulad_repair_selection.py  --phase1-dir runs/oulad_phase1 --phase2-dir runs/oulad_phase2 --phase3-dir runs/oulad_phase3 --output-dir runs/oulad_phase4
python 05_oulad_final_audit.py       --phase1-dir runs/oulad_phase1 --phase4-dir runs/oulad_phase4 --output-dir runs/oulad_phase5
```

The xAPI pipeline follows the same pattern with `--xapi-dir` in place of
`--oulad-dir`.

Run `python <script> --help` for the full option list of any phase.

---

## Configuration

Values used for the results reported in the thesis:

| Setting | Value |
|---|---|
| LiSSA recursion depth | 100 |
| LiSSA damping | 0.01 |
| LiSSA scale | 1000 |
| LiSSA repetitions (averaged) | 3 |
| Decision threshold | 0.50 |
| Utility tolerance | 0.02 |
| Bootstrap resamples | 1000, stratified |
| Synthetic: learning rate / epochs / batch | 2e-5 / 4 / 16 |
| Synthetic: max sequence length | 96 |
| Tabular: hidden widths | 64, 32 |
| Tabular: learning rate | 1e-3 |
| Repair grid (influence–Hessian, synthetic) | 0, 0.10, 0.25, 0.50, 0.75, 1.00 |
| Repair grid (raw gradient, synthetic) | 0, 0.001, 0.005, 0.010, 0.025, 0.050 |
| Repair grid (both, real data) | 0, 0.25, 0.50, 0.75, 1, 2, 5, 10 |

The two synthetic grids differ because the raw gradient has a norm roughly an
order of magnitude larger than the preconditioned direction; the grids were
chosen to span comparable ranges of parameter displacement.

---

## Reproducibility

Random seeds are set for data generation, partitioning, model initialization
and stochastic estimation, and every phase writes a metadata file recording its
configuration.

The results reported in the thesis were produced on Apple Silicon using the
PyTorch Metal Performance Shaders backend, which is not bitwise deterministic.
Repeating a run on different hardware may therefore produce small numerical
differences, and the synthetic experiments in particular are sensitive to these:
the thesis documents substantial variation in threshold-level outcomes across
random seeds at high corruption severity. Passing `--device cpu` where the
option is available gives more reproducible behaviour at the cost of speed.

Phase-level consistency checks verify that each phase operates on the same
model and fairness state as the one before it; the reproduced fairness values
are recorded in the phase metadata.

---

## Citation

```bibtex
@mastersthesis{sabih2026fairnessrepair,
  author = {Khalid Sabih},
  title  = {Repairing Fairness Without Retraining: An Influence-Guided
            Approach for Educational Machine Learning},
  school = {Universit\"at Leipzig},
  year   = {2026}
}
```

## License

See `LICENSE`.