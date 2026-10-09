# An Influence-Guided, Machine-Unlearning-Inspired Framework for Post-Training Fairness Repair

Implementation accompanying the master's thesis *An Influence-Guided,
Machine-Unlearning-Inspired Framework for Post-Training Fairness Repair in
Educational Machine Learning* (Khalid Sabih, Universität Leipzig, 2026).

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
| 2 | Fairness gradient, LiSSA inverse-HVP, per-sample influence scores | `influence_scores.csv`; the repair direction, as `influence_fairness_direction.pt` (synthetic) or `fairness_unlearning_directions.pt` (tabular) |
| 3 | Comparison of candidate repair directions | direction summary, candidate paths |
| 4 | Utility-constrained selection of a repair fraction | frozen repaired checkpoints, `repair_selection_metadata.json` |
| 5 | Single audit on the untouched test partition | `final_test_metrics.csv`, bootstrap intervals, prediction changes |

The untouched test partition is read only in Phase 5. The synthetic pipeline
writes its artifacts to `data/` and `models/` beside the scripts under its own
file names; the tabular pipelines write to the `--output-dir` of each phase.

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

The code uses no syntax newer than Python 3.9. The reported results were
produced on an Apple M1 with 16 GB unified memory using the PyTorch Metal
Performance Shaders backend.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The synthetic benchmark downloads `bert-base-uncased` from the Hugging Face
Hub on first run.

The versions in `requirements.txt` are those of the environment that produced
the reported results. The eight packages it lists are the complete set of
third-party imports across all fifteen phase scripts.

Approximate runtimes on the hardware above, taken from the logs in `results/`.
A full synthetic run takes fifteen to seventeen minutes, of which Phase 1
accounts for about ten: it fine-tunes two models, the biased baseline and the
clean-label oracle reference, at roughly five minutes each. Phase 5, the audit
with bootstrap resampling, takes just under two minutes. Phases 2 to 5 of the
tabular pipelines together take about ninety seconds on OULAD and sixty on
xAPI-Edu-Data, with Phase 1 training faster still.

---

## Data

Neither dataset is redistributed here.

**OULAD** — download `anonymisedData.zip` from
<https://research.stem.open.ac.uk/ouanalyse/dataset/> and unpack it under
`data/oulad/`; Phase 1 finds `studentInfo.csv` there or in any subfolder. The
experiments use module BBB. The dataset is described in Kuzilek, Hlosta and
Zdrahal, "Open University Learning Analytics Dataset", *Scientific Data* 4
(2017), 170171, <https://doi.org/10.1038/sdata.2017.171>.

**xAPI-Edu-Data** — download from
<https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data> and place
`xAPI-Edu-Data.csv` under `data/xapi/`.

The synthetic benchmark generates its own data in Phase 1 and requires no
download.

---

## Running the pipeline

### Synthetic benchmark

The synthetic scripts take no command-line arguments. Each one carries a
`Config` class near the top of the file, and a run is configured by editing it;
paths are relative, so the scripts must be run from inside `synthetic/`.

```bash
cd synthetic
# edit the Config class in 01_bias_injection.py: seed, bias_probability
python 01_bias_injection.py
python 02_influence_diagnostics_direction.py
python 03_direction_comparison.py
python 04_geometric_intervention.py
python 05_model_audit.py
```

The five runs reported in the thesis use `seed = 42` at `bias_probability`
0.50, 0.70 and 0.85, and `bias_probability = 0.85` at seeds 123 and 456.
Phase 1 writes the baseline checkpoint to `models/m0_biased` and the partitions
under `data/`, where the later phases expect them.

### OULAD

The tabular pipelines are argument-driven, and every phase takes the output
directories of the phases before it.

```bash
cd oulad
python 01_oulad_data_and_training.py --oulad-dir ../data/oulad --output-dir runs/oulad_phase1
python 02_oulad_influence_diagnostics.py --phase1-dir runs/oulad_phase1 --output-dir runs/oulad_phase2
python 03_oulad_direction_comparison.py --phase1-dir runs/oulad_phase1 --phase2-dir runs/oulad_phase2 --output-dir runs/oulad_phase3
python 04_oulad_repair_selection.py  --phase1-dir runs/oulad_phase1 --phase2-dir runs/oulad_phase2 --phase3-dir runs/oulad_phase3 --output-dir runs/oulad_phase4
python 05_oulad_final_audit.py       --phase1-dir runs/oulad_phase1 --phase4-dir runs/oulad_phase4 --output-dir runs/oulad_phase5
```

Phase 1 selects module BBB by default; pass `--module ALL` for the full
dataset. The protected attribute defaults to `gender` with groups `M` and `F`.

### xAPI-Edu-Data

Identical to OULAD except that Phase 1 takes `--dataset-dir` rather than
`--oulad-dir`:

```bash
cd xapi
python 01_xapi_data_and_training.py --dataset-dir ../data/xapi --output-dir runs/xapi_phase1
```

Phases 2 to 5 take the same flags as their OULAD counterparts.

If no `--*-dir` is given, each tabular script falls back to a directory beside
itself: `archive/` for the raw data and `<pipeline>_phase<N>/` for its own
output.

Run `python <script> --help` for the full option list of any phase.

---

## Configuration

Values used for the results reported in the thesis:

| Setting | Value |
|---|---|
| LiSSA recursion depth (synthetic / real) | 100 / 200 |
| LiSSA damping | 0.01 |
| LiSSA scale (synthetic / real) | 1000 / 100 |
| LiSSA repetitions, averaged (synthetic / real) | 3 / 1 |
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

The two synthetic grids differ because the raw fairness gradient has a norm
between roughly thirty and seventy-five times larger than the preconditioned
direction; the grids were chosen to span comparable ranges of parameter
displacement. Equal fractions on a shared grid would therefore not be
comparable, which is why the thesis also reports common-distance results that
normalise both directions to the same parameter-space radius.

The LiSSA settings above are the defaults in the synthetic `Config` classes and
the defaults of `--recursion-depth`, `--damping`, `--scale` and
`--lissa-repetitions` in the tabular Phase 2 scripts.

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
  title  = {An Influence-Guided, Machine-Unlearning-Inspired Framework for
            Post-Training Fairness Repair in Educational Machine Learning},
  school = {Universit\"at Leipzig},
  year   = {2026}
}
```

## License

See `LICENSE`.