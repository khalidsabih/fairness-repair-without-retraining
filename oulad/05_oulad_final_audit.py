"""
Phase 5 - Final untouched-test audit for the natural OULAD experiment.

This script performs no training, tuning, repair selection, or test-dependent
model choice. It loads the frozen checkpoints created by Phase 4 and evaluates
all models exactly once on the untouched OULAD test split created in Phase 1.

Corresponds to Section 5.6 of the thesis.

Compared models
---------------
- baseline
- influence_hessian_repair
- raw_fairness_gradient_repair

Reported outputs
----------------
- utility metrics: accuracy, balanced accuracy, macro-F1, positive-class
  precision/recall, average precision, and AUROC;
- fairness metrics: smooth EO surrogate, hard equalized-odds difference,
  group-specific TPR/FPR/FNR/selection rates, and their gaps;
- group confusion matrices;
- prediction and probability changes relative to the baseline;
- stratified bootstrap confidence intervals;
- per-registration predictions and audit metadata.

Expected layout
---------------
project/
├── 01_oulad_data_and_training.py
├── 02_oulad_influence_diagnostics.py
├── 03_oulad_direction_comparison.py
├── 04_oulad_repair_selection.py
├── 05_oulad_final_audit.py
├── oulad_phase1/
└── oulad_phase4/

Run
---
python 05_oulad_final_audit.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fairlearn.metrics import equalized_odds_difference
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class Config:
    phase1_dir: str = ""
    phase4_dir: str = ""
    output_dir: str = ""
    seed: int = 42
    device: str = "auto"
    batch_size: int = 512
    threshold: float = 0.50
    bootstrap_repetitions: int = 1000
    confidence_level: float = 0.95


class OULADMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim_2, 2)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.encoder(features))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return (
            np.asarray(data["features"], dtype=np.float32),
            np.asarray(data["labels"], dtype=np.int64),
            np.asarray(data["demo"]).astype(str),
            np.asarray(data["sample_id"]).astype(str),
        )


def load_model(path: Path, device: torch.device) -> Tuple[OULADMLP, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    required = {
        "model_state_dict",
        "input_dim",
        "hidden_dim_1",
        "hidden_dim_2",
        "dropout",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing fields: {sorted(missing)}")

    model = OULADMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim_1=int(checkpoint["hidden_dim_1"]),
        hidden_dim_2=int(checkpoint["hidden_dim_2"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        torch.as_tensor(features, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            parts.append(probabilities.detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


def safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def confusion_counts(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, int]:
    return {
        "TP": int(np.sum((labels == 1) & (predictions == 1))),
        "FN": int(np.sum((labels == 1) & (predictions == 0))),
        "FP": int(np.sum((labels == 0) & (predictions == 1))),
        "TN": int(np.sum((labels == 0) & (predictions == 0))),
    }


def group_rates(
    labels: np.ndarray,
    predictions: np.ndarray,
    demo: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for group in ("A", "B"):
        mask = demo == group
        counts = confusion_counts(labels[mask], predictions[mask])
        tp, fn, fp, tn = (
            counts["TP"],
            counts["FN"],
            counts["FP"],
            counts["TN"],
        )
        result[f"n_{group}"] = int(mask.sum())
        result[f"TPR_{group}"] = tp / (tp + fn) if tp + fn else float("nan")
        result[f"FPR_{group}"] = fp / (fp + tn) if fp + tn else float("nan")
        result[f"FNR_{group}"] = fn / (tp + fn) if tp + fn else float("nan")
        result[f"selection_rate_{group}"] = (
            float(np.mean(predictions[mask])) if np.any(mask) else float("nan")
        )
    result["TPR_gap"] = abs(result["TPR_A"] - result["TPR_B"])
    result["FPR_gap"] = abs(result["FPR_A"] - result["FPR_B"])
    result["FNR_gap"] = abs(result["FNR_A"] - result["FNR_B"])
    result["selection_rate_gap"] = abs(
        result["selection_rate_A"] - result["selection_rate_B"]
    )
    return result


def smooth_eo_surrogate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    terms: List[float] = []
    means_out: Dict[str, float] = {}
    for label in (0, 1):
        means: Dict[str, float] = {}
        for group in ("A", "B"):
            mask = (labels == label) & (demo == group)
            if not np.any(mask):
                raise ValueError(f"Empty fairness cell {group}_{label}.")
            value = float(np.mean(probabilities[mask]))
            means[group] = value
            means_out[f"mean_probability_{group}_y{label}"] = value
        terms.append((means["A"] - means["B"]) ** 2)
    return float(sum(terms)), means_out


def evaluate_split(
    probabilities: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_positive": float(precision_score(labels, predictions, zero_division=0)),
        "recall_positive": float(recall_score(labels, predictions, zero_division=0)),
        "average_precision": safe_average_precision(labels, probabilities),
        "roc_auc": safe_roc_auc(labels, probabilities),
        "equalized_odds_difference": float(
            equalized_odds_difference(labels, predictions, sensitive_features=demo)
        ),
        "threshold": float(threshold),
    }
    smooth, cell_means = smooth_eo_surrogate(probabilities, labels, demo)
    metrics["smooth_eo_surrogate"] = smooth
    metrics.update(group_rates(labels, predictions, demo))
    metrics.update(cell_means)
    return metrics


def prediction_change_metrics(
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
    threshold: float,
) -> Dict[str, int | float]:
    baseline_predictions = (baseline_probabilities >= threshold).astype(np.int64)
    candidate_predictions = (candidate_probabilities >= threshold).astype(np.int64)
    changed = candidate_predictions != baseline_predictions
    beneficial = changed & (candidate_predictions == labels) & (baseline_predictions != labels)
    harmful = changed & (candidate_predictions != labels) & (baseline_predictions == labels)

    result: Dict[str, int | float] = {
        "changed_predictions": int(changed.sum()),
        "fraction_changed": float(changed.mean()),
        "flips_0_to_1": int(np.sum((baseline_predictions == 0) & (candidate_predictions == 1))),
        "flips_1_to_0": int(np.sum((baseline_predictions == 1) & (candidate_predictions == 0))),
        "beneficial_flips": int(beneficial.sum()),
        "harmful_flips": int(harmful.sum()),
        "mean_probability_change": float(
            np.mean(candidate_probabilities - baseline_probabilities)
        ),
        "mean_absolute_probability_change": float(
            np.mean(np.abs(candidate_probabilities - baseline_probabilities))
        ),
    }
    for group in ("A", "B"):
        mask = demo == group
        result[f"changed_predictions_{group}"] = int(np.sum(changed & mask))
        result[f"beneficial_flips_{group}"] = int(np.sum(beneficial & mask))
        result[f"harmful_flips_{group}"] = int(np.sum(harmful & mask))
        result[f"{group}_0_to_1"] = int(
            np.sum(mask & (baseline_predictions == 0) & (candidate_predictions == 1))
        )
        result[f"{group}_1_to_0"] = int(
            np.sum(mask & (baseline_predictions == 1) & (candidate_predictions == 0))
        )
        result[f"mean_probability_change_{group}"] = float(
            np.mean(candidate_probabilities[mask] - baseline_probabilities[mask])
        )
        result[f"mean_absolute_probability_change_{group}"] = float(
            np.mean(np.abs(candidate_probabilities[mask] - baseline_probabilities[mask]))
        )
    return result


def build_group_confusion_rows(
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    demo: np.ndarray,
    threshold: float,
) -> List[Dict[str, object]]:
    predictions = (probabilities >= threshold).astype(np.int64)
    rows: List[Dict[str, object]] = []
    for group in ("A", "B"):
        mask = demo == group
        counts = confusion_counts(labels[mask], predictions[mask])
        rows.append(
            {
                "model": model_name,
                "group": group,
                "n": int(mask.sum()),
                **counts,
                "TPR": counts["TP"] / (counts["TP"] + counts["FN"])
                if counts["TP"] + counts["FN"]
                else float("nan"),
                "FPR": counts["FP"] / (counts["FP"] + counts["TN"])
                if counts["FP"] + counts["TN"]
                else float("nan"),
                "FNR": counts["FN"] / (counts["TP"] + counts["FN"])
                if counts["TP"] + counts["FN"]
                else float("nan"),
                "selection_rate": float(np.mean(predictions[mask])),
            }
        )
    return rows


def stratified_bootstrap_indices(
    labels: np.ndarray,
    demo: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_parts: List[np.ndarray] = []
    for group in ("A", "B"):
        for label in (0, 1):
            indices = np.flatnonzero((demo == group) & (labels == label))
            if len(indices) == 0:
                raise ValueError(f"Cannot bootstrap empty stratum {group}_{label}.")
            sampled_parts.append(rng.choice(indices, size=len(indices), replace=True))
    combined = np.concatenate(sampled_parts)
    rng.shuffle(combined)
    return combined


def bootstrap_intervals(
    model_probabilities: Mapping[str, np.ndarray],
    labels: np.ndarray,
    demo: np.ndarray,
    threshold: float,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "average_precision",
        "roc_auc",
        "smooth_eo_surrogate",
        "equalized_odds_difference",
        "TPR_gap",
        "FPR_gap",
        "FNR_gap",
        "selection_rate_gap",
    ]
    values: Dict[Tuple[str, str], List[float]] = {
        (model_name, metric): []
        for model_name in model_probabilities
        for metric in metric_names
    }
    rng = np.random.default_rng(seed)

    for repetition in range(repetitions):
        indices = stratified_bootstrap_indices(labels, demo, rng)
        bootstrap_labels = labels[indices]
        bootstrap_demo = demo[indices]
        for model_name, probabilities in model_probabilities.items():
            metrics = evaluate_split(
                probabilities[indices], bootstrap_labels, bootstrap_demo, threshold
            )
            for metric in metric_names:
                values[(model_name, metric)].append(float(metrics[metric]))
        if (repetition + 1) % 100 == 0 or repetition + 1 == repetitions:
            print(f"Bootstrap: {repetition + 1}/{repetitions}")

    alpha = 1.0 - confidence_level
    lower_quantile = 100.0 * alpha / 2.0
    upper_quantile = 100.0 * (1.0 - alpha / 2.0)
    rows: List[Dict[str, object]] = []
    for (model_name, metric), samples in values.items():
        array = np.asarray(samples, dtype=float)
        finite = array[np.isfinite(array)]
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "bootstrap_repetitions": repetitions,
                "confidence_level": confidence_level,
                "bootstrap_mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "bootstrap_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "ci_lower": float(np.percentile(finite, lower_quantile))
                if len(finite)
                else float("nan"),
                "ci_upper": float(np.percentile(finite, upper_quantile))
                if len(finite)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OULAD Phase 5: final untouched-test audit")
    parser.add_argument("--phase1-dir", default=None)
    parser.add_argument("--phase4-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    phase1_dir = (
        Path(args.phase1_dir).expanduser().resolve()
        if args.phase1_dir
        else script_dir / "oulad_phase1"
    )
    phase4_dir = (
        Path(args.phase4_dir).expanduser().resolve()
        if args.phase4_dir
        else script_dir / "oulad_phase4"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "oulad_phase5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        phase1_dir=str(phase1_dir),
        phase4_dir=str(phase4_dir),
        output_dir=str(output_dir),
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        bootstrap_repetitions=args.bootstrap_repetitions,
        confidence_level=args.confidence_level,
    )
    set_seed(config.seed)
    device = choose_device(config.device)
    print(f"Device: {device}")

    paths = {
        "test_npz": phase1_dir / "test_features.npz",
        "test_csv": phase1_dir / "test.csv",
        "baseline": phase4_dir / "models" / "baseline" / "model.pt",
        "influence": phase4_dir / "models" / "m1_influence_repaired" / "model.pt",
        "raw": phase4_dir / "models" / "m1_raw_gradient_repaired" / "model.pt",
        "selection": phase4_dir / "repair_selection.csv",
        "selection_metadata": phase4_dir / "repair_selection_metadata.json",
    }
    require_files(list(paths.values()))

    started = time.time()
    features, labels, demo, sample_ids = load_npz(paths["test_npz"])
    test_frame = pd.read_csv(paths["test_csv"])
    if len(test_frame) != len(labels):
        raise ValueError("test.csv and test_features.npz have different row counts.")
    if "sample_id" in test_frame.columns:
        csv_ids = test_frame["sample_id"].astype(str).to_numpy()
        if not np.array_equal(csv_ids, sample_ids):
            raise ValueError("test.csv and test_features.npz sample IDs are not aligned.")

    model_specs = {
        "baseline": paths["baseline"],
        "influence_hessian_repair": paths["influence"],
        "raw_fairness_gradient_repair": paths["raw"],
    }

    probabilities_by_model: Dict[str, np.ndarray] = {}
    checkpoint_metadata: Dict[str, object] = {}
    metric_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []

    print("\nEvaluating frozen models on untouched OULAD test split...")
    for model_name, model_path in model_specs.items():
        print(f"- {model_name}")
        model, checkpoint = load_model(model_path, device)
        probabilities = predict_probabilities(model, features, config.batch_size, device)
        probabilities_by_model[model_name] = probabilities
        metrics = evaluate_split(probabilities, labels, demo, config.threshold)
        metric_rows.append({"model": model_name, **metrics})
        confusion_rows.extend(
            build_group_confusion_rows(
                model_name, labels, probabilities, demo, config.threshold
            )
        )
        checkpoint_metadata[model_name] = {
            "path": str(model_path),
            "model_role": checkpoint.get("model_role"),
            "repair_method": checkpoint.get("repair_method"),
            "selected_fraction": checkpoint.get("selected_fraction"),
            "applied_scale": checkpoint.get("applied_scale"),
            "parameter_distance": checkpoint.get("parameter_distance"),
            "test_set_used_for_selection": checkpoint.get("test_set_used_for_selection"),
        }

    metrics_frame = pd.DataFrame(metric_rows)
    confusion_frame = pd.DataFrame(confusion_rows)

    baseline_probabilities = probabilities_by_model["baseline"]
    pairwise_rows: List[Dict[str, object]] = []
    for model_name in (
        "influence_hessian_repair",
        "raw_fairness_gradient_repair",
    ):
        changes = prediction_change_metrics(
            baseline_probabilities,
            probabilities_by_model[model_name],
            labels,
            demo,
            config.threshold,
        )
        pairwise_rows.append(
            {
                "reference_model": "baseline",
                "candidate_model": model_name,
                **changes,
            }
        )
    pairwise_frame = pd.DataFrame(pairwise_rows)

    prediction_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "demo": demo,
        }
    )
    for model_name, probabilities in probabilities_by_model.items():
        prediction_frame[f"probability_{model_name}"] = probabilities
        prediction_frame[f"prediction_{model_name}"] = (
            probabilities >= config.threshold
        ).astype(np.int64)

    print("\nComputing stratified bootstrap confidence intervals...")
    bootstrap_frame = bootstrap_intervals(
        probabilities_by_model,
        labels,
        demo,
        config.threshold,
        config.bootstrap_repetitions,
        config.confidence_level,
        config.seed + 1000,
    )

    metrics_path = output_dir / "final_test_metrics.csv"
    confusion_path = output_dir / "final_test_group_confusions.csv"
    pairwise_path = output_dir / "final_test_pairwise_changes.csv"
    bootstrap_path = output_dir / "final_test_bootstrap_intervals.csv"
    predictions_path = output_dir / "final_test_predictions.csv"
    metadata_path = output_dir / "final_test_audit_metadata.json"

    metrics_frame.to_csv(metrics_path, index=False)
    confusion_frame.to_csv(confusion_path, index=False)
    pairwise_frame.to_csv(pairwise_path, index=False)
    bootstrap_frame.to_csv(bootstrap_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False)

    selection_frame = pd.read_csv(paths["selection"])
    with paths["selection_metadata"].open("r", encoding="utf-8") as handle:
        selection_metadata = json.load(handle)

    runtime = time.time() - started
    metadata = {
        "phase": 5,
        "experiment": "OULAD natural-disparity final audit",
        "dataset": "OULAD",
        "artificial_bias_injection": False,
        "oracle_model_available": False,
        "test_split_role": "untouched final evaluation only",
        "test_set_used_for_training": False,
        "test_set_used_for_repair_estimation": False,
        "test_set_used_for_selection": False,
        "n_test": int(len(labels)),
        "test_group_counts": {
            group: int(np.sum(demo == group)) for group in ("A", "B")
        },
        "test_group_label_counts": {
            f"{group}_{label}": int(np.sum((demo == group) & (labels == label)))
            for group in ("A", "B")
            for label in (0, 1)
        },
        "models": checkpoint_metadata,
        "phase4_selection": selection_frame.to_dict(orient="records"),
        "phase4_selection_metadata": selection_metadata,
        "config": asdict(config),
        "runtime_seconds": runtime,
        "outputs": {
            "metrics": str(metrics_path),
            "group_confusions": str(confusion_path),
            "pairwise_changes": str(pairwise_path),
            "bootstrap_intervals": str(bootstrap_path),
            "predictions": str(predictions_path),
        },
    }
    write_json(metadata_path, metadata)

    summary_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "average_precision",
        "roc_auc",
        "smooth_eo_surrogate",
        "equalized_odds_difference",
        "TPR_gap",
        "FPR_gap",
        "selection_rate_gap",
    ]
    print("\nFinal untouched-test summary:")
    print(metrics_frame[summary_columns].to_string(index=False))

    print("\nPrediction changes relative to baseline:")
    print(pairwise_frame.to_string(index=False))

    print("\nSaved outputs:")
    for path in (
        metrics_path,
        confusion_path,
        pairwise_path,
        bootstrap_path,
        predictions_path,
        metadata_path,
    ):
        print(f"- {path}")
    print(f"\nPhase 5 completed in {runtime:.2f} seconds.")
    print("No training, tuning, or repair selection was performed in this phase.")


if __name__ == "__main__":
    main()