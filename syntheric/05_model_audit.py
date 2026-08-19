"""
Phase 5: Frozen final audit on the untouched test split.

This script evaluates only models selected and saved in Phase 4. It never
searches over fractions, thresholds, or hyperparameters, so the test split
remains a genuine final evaluation set.

Corresponds to Section 5.6 of the thesis.

Evaluated models
----------------
- biased_baseline
- influence_hessian_repair (primary method)
- raw_gradient_baseline
- deletion_baseline
- oracle_direction_reference (diagnostic, selected-subspace oracle direction)
- full_oracle_reference (upper-bound reference)

Outputs
-------
- data/final_test_metrics.csv
- data/final_test_group_confusions.csv
- data/final_test_pairwise_changes.csv
- data/final_test_bootstrap_intervals.csv
- data/final_test_predictions.csv
- data/final_test_audit_metadata.json
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import equalized_odds_difference
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer


@dataclass(frozen=True)
class Config:
    test_path: str = "data/test.csv"
    selection_metadata_path: str = "data/repair_selection_metadata.json"

    biased_model_path: str = "models/m0_biased"
    influence_model_path: str = "models/m1_influence_repaired"
    raw_gradient_model_path: str = "models/m1_raw_gradient_baseline"
    deletion_model_path: str = "models/m1_deletion_baseline"
    oracle_direction_model_path: str = "models/m1_oracle_direction_reference"
    full_oracle_model_path: str = "models/mr_oracle_reference"

    metrics_output: str = "./data/final_test_metrics.csv"
    confusions_output: str = "./data/final_test_group_confusions.csv"
    changes_output: str = "./data/final_test_pairwise_changes.csv"
    bootstrap_output: str = "./data/final_test_bootstrap_intervals.csv"
    predictions_output: str = "./data/final_test_predictions.csv"
    metadata_output: str = "./data/final_test_audit_metadata.json"

    seed: int = 42
    max_length: int = 96
    batch_size: int = 32
    decision_threshold: float = 0.5

    # Stratified bootstrap by demographic group x true label.
    bootstrap_repetitions: int = 1000
    confidence_level: float = 0.95

    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


config = Config()
DEVICE = torch.device(config.device)
EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(config.seed)


def require_paths(paths: Sequence[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required Phase 5 inputs:\n- " + "\n- ".join(missing)
        )


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Test data is missing required columns: {missing}")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer) -> None:
        self.encodings = tokenizer(
            df["essay_text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
        }


@torch.no_grad()
def predict_probabilities(
    model: BertForSequenceClassification,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: List[np.ndarray] = []
    probability_parts: List[np.ndarray] = []

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        )
        logits = outputs.logits.detach().float().cpu().numpy()
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
        logits_parts.append(logits)
        probability_parts.append(probabilities.detach().float().cpu().numpy())

    return np.concatenate(logits_parts), np.concatenate(probability_parts)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if bool(np.any(mask)) else float("nan")


def smooth_eo_surrogate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    terms: List[float] = []
    for label in (0, 1):
        mask_a = (groups == "A") & (labels == label)
        mask_b = (groups == "B") & (labels == label)
        if not np.any(mask_a) or not np.any(mask_b):
            raise ValueError(
                "Smooth EO requires both demographic groups for each label."
            )
        gap = float(np.mean(probabilities[mask_a]) - np.mean(probabilities[mask_b]))
        terms.append(gap * gap)
    return float(sum(terms))


def group_confusion_rows(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        matrix = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
        rows.append(
            {
                "model": model_name,
                "group": group,
                "n": int(np.sum(mask)),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tpr": safe_divide(tp, tp + fn),
                "fpr": safe_divide(fp, fp + tn),
                "tnr": safe_divide(tn, tn + fp),
                "fnr": safe_divide(fn, fn + tp),
                "positive_prediction_rate": float(np.mean(y_pred[mask])),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }
        )
    return rows


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    predictions = (probabilities >= config.decision_threshold).astype(int)
    rows = group_confusion_rows("temporary", y_true, predictions, groups)
    rates = {str(row["group"]): row for row in rows}

    tpr_a = float(rates["A"]["tpr"])
    tpr_b = float(rates["B"]["tpr"])
    fpr_a = float(rates["A"]["fpr"])
    fpr_b = float(rates["B"]["fpr"])
    pass_a = float(rates["A"]["positive_prediction_rate"])
    pass_b = float(rates["B"]["positive_prediction_rate"])

    return {
        "n_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "positive_class_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "smooth_eo_surrogate": smooth_eo_surrogate(probabilities, y_true, groups),
        "equalized_odds_difference": float(
            equalized_odds_difference(
                y_true,
                predictions,
                sensitive_features=groups,
            )
        ),
        "tpr_A": tpr_a,
        "tpr_B": tpr_b,
        "tpr_gap": abs(tpr_a - tpr_b),
        "fpr_A": fpr_a,
        "fpr_B": fpr_b,
        "fpr_gap": abs(fpr_a - fpr_b),
        "pass_rate_A": pass_a,
        "pass_rate_B": pass_b,
        "pass_rate_gap": abs(pass_a - pass_b),
        "mean_probability_A": safe_mean(probabilities, groups == "A"),
        "mean_probability_B": safe_mean(probabilities, groups == "B"),
        "mean_probability_gap": abs(
            safe_mean(probabilities, groups == "A")
            - safe_mean(probabilities, groups == "B")
        ),
        "mean_probability": float(np.mean(probabilities)),
        "prediction_positive_rate": float(np.mean(predictions)),
    }


def pairwise_change_metrics(
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, object]:
    changed = candidate_predictions != reference_predictions
    row: Dict[str, object] = {
        "changed_predictions": int(np.sum(changed)),
        "fraction_changed": float(np.mean(changed)),
        "flips_0_to_1": int(np.sum((reference_predictions == 0) & (candidate_predictions == 1))),
        "flips_1_to_0": int(np.sum((reference_predictions == 1) & (candidate_predictions == 0))),
        "beneficial_flips": int(
            np.sum(changed & (candidate_predictions == y_true) & (reference_predictions != y_true))
        ),
        "harmful_flips": int(
            np.sum(changed & (candidate_predictions != y_true) & (reference_predictions == y_true))
        ),
        "mean_probability_change": float(
            np.mean(candidate_probabilities - reference_probabilities)
        ),
        "mean_absolute_probability_change": float(
            np.mean(np.abs(candidate_probabilities - reference_probabilities))
        ),
    }
    for group in ("A", "B"):
        mask = groups == group
        row[f"changed_predictions_{group}"] = int(np.sum(changed & mask))
        row[f"{group}_0_to_1"] = int(
            np.sum(mask & (reference_predictions == 0) & (candidate_predictions == 1))
        )
        row[f"{group}_1_to_0"] = int(
            np.sum(mask & (reference_predictions == 1) & (candidate_predictions == 0))
        )
        row[f"mean_probability_change_{group}"] = float(
            np.mean(candidate_probabilities[mask] - reference_probabilities[mask])
        )
    return row


def stratified_bootstrap_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_parts: List[np.ndarray] = []
    for group in sorted(np.unique(groups).tolist()):
        for label in sorted(np.unique(labels).tolist()):
            stratum = np.flatnonzero((groups == group) & (labels == label))
            if len(stratum) == 0:
                continue
            sampled_parts.append(rng.choice(stratum, size=len(stratum), replace=True))
    combined = np.concatenate(sampled_parts)
    rng.shuffle(combined)
    return combined


def bootstrap_intervals(
    model_probabilities: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "smooth_eo_surrogate",
        "equalized_odds_difference",
        "tpr_gap",
        "fpr_gap",
        "pass_rate_gap",
    ]
    storage: Dict[Tuple[str, str], List[float]] = {
        (model_name, metric): []
        for model_name in model_probabilities
        for metric in metric_names
    }
    rng = np.random.default_rng(config.seed)

    for repetition in range(config.bootstrap_repetitions):
        indices = stratified_bootstrap_indices(labels, groups, rng)
        for model_name, probabilities in model_probabilities.items():
            metrics = calculate_metrics(
                labels[indices], probabilities[indices], groups[indices]
            )
            for metric in metric_names:
                storage[(model_name, metric)].append(float(metrics[metric]))
        if (repetition + 1) % 100 == 0:
            print(
                f"Bootstrap: {repetition + 1}/{config.bootstrap_repetitions}",
                flush=True,
            )

    alpha = 1.0 - config.confidence_level
    rows: List[Dict[str, object]] = []
    for (model_name, metric), values in storage.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "estimate": float(
                    calculate_metrics(
                        labels,
                        model_probabilities[model_name],
                        groups,
                    )[metric]
                ),
                "ci_lower": float(np.quantile(array, alpha / 2.0)),
                "ci_upper": float(np.quantile(array, 1.0 - alpha / 2.0)),
                "confidence_level": config.confidence_level,
                "bootstrap_repetitions": config.bootstrap_repetitions,
            }
        )
    return pd.DataFrame(rows)


def load_model(path: str) -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(path).to(DEVICE)


def main() -> None:
    start_time = time.time()
    print(f"Using device: {DEVICE}")

    required_paths = [
        config.test_path,
        config.selection_metadata_path,
        config.biased_model_path,
        config.influence_model_path,
        config.raw_gradient_model_path,
        config.deletion_model_path,
        config.oracle_direction_model_path,
        config.full_oracle_model_path,
    ]
    require_paths(required_paths)

    with open(config.selection_metadata_path, "r", encoding="utf-8") as handle:
        selection_metadata = json.load(handle)

    test_df = pd.read_csv(config.test_path)
    validate_columns(
        test_df,
        ["sample_id", "essay_text", "true_quality", "demo", "split"],
    )
    if not bool((test_df["split"].astype(str) == "test").all()):
        raise ValueError("Phase 5 input contains rows outside the test split.")
    if test_df["sample_id"].duplicated().any():
        raise ValueError("Test sample_id values must be unique.")

    labels = test_df["true_quality"].astype(int).to_numpy()
    groups = test_df["demo"].astype(str).to_numpy()
    if set(np.unique(groups).tolist()) != {"A", "B"}:
        raise ValueError("Expected exactly demographic groups A and B.")

    tokenizer = BertTokenizer.from_pretrained(config.biased_model_path)
    dataset = TestDataset(test_df, tokenizer)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model_paths: Dict[str, str] = {
        "biased_baseline": config.biased_model_path,
        "influence_hessian_repair": config.influence_model_path,
        "raw_gradient_baseline": config.raw_gradient_model_path,
        "deletion_baseline": config.deletion_model_path,
        "oracle_direction_reference": config.oracle_direction_model_path,
        "full_oracle_reference": config.full_oracle_model_path,
    }

    probabilities_by_model: Dict[str, np.ndarray] = {}
    logits_by_model: Dict[str, np.ndarray] = {}
    metrics_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []

    print("\nEvaluating frozen models on untouched test.csv...")
    for model_name, model_path in model_paths.items():
        print(f"- {model_name}")
        model = load_model(model_path)
        logits, probabilities = predict_probabilities(model, loader)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        logits_by_model[model_name] = logits
        probabilities_by_model[model_name] = probabilities
        predictions = (probabilities >= config.decision_threshold).astype(int)

        metrics = calculate_metrics(labels, probabilities, groups)
        metrics_rows.append(
            {
                "model": model_name,
                "role": (
                    "primary"
                    if model_name == "influence_hessian_repair"
                    else "baseline"
                    if model_name in {
                        "biased_baseline",
                        "raw_gradient_baseline",
                        "deletion_baseline",
                    }
                    else "reference"
                ),
                **metrics,
            }
        )
        confusion_rows.extend(
            group_confusion_rows(model_name, labels, predictions, groups)
        )

    metrics_df = pd.DataFrame(metrics_rows)
    confusions_df = pd.DataFrame(confusion_rows)

    baseline_probabilities = probabilities_by_model["biased_baseline"]
    baseline_predictions = (
        baseline_probabilities >= config.decision_threshold
    ).astype(int)

    changes_rows: List[Dict[str, object]] = []
    for model_name, probabilities in probabilities_by_model.items():
        if model_name == "biased_baseline":
            continue
        predictions = (probabilities >= config.decision_threshold).astype(int)
        changes_rows.append(
            {
                "reference_model": "biased_baseline",
                "candidate_model": model_name,
                **pairwise_change_metrics(
                    baseline_predictions,
                    predictions,
                    baseline_probabilities,
                    probabilities,
                    labels,
                    groups,
                ),
            }
        )
    changes_df = pd.DataFrame(changes_rows)

    prediction_output = test_df[
        ["sample_id", "true_quality", "demo", "split"]
    ].copy()
    for model_name, probabilities in probabilities_by_model.items():
        prediction_output[f"probability_{model_name}"] = probabilities
        prediction_output[f"prediction_{model_name}"] = (
            probabilities >= config.decision_threshold
        ).astype(int)
        prediction_output[f"margin_{model_name}"] = probabilities - config.decision_threshold

    print("\nComputing stratified bootstrap confidence intervals...")
    bootstrap_df = bootstrap_intervals(probabilities_by_model, labels, groups)

    os.makedirs(os.path.dirname(config.metrics_output), exist_ok=True)
    metrics_df.to_csv(config.metrics_output, index=False)
    confusions_df.to_csv(config.confusions_output, index=False)
    changes_df.to_csv(config.changes_output, index=False)
    bootstrap_df.to_csv(config.bootstrap_output, index=False)
    prediction_output.to_csv(config.predictions_output, index=False)

    selected_fractions: Dict[str, object] = {}
    if isinstance(selection_metadata, Mapping):
        for key in ("selected_fractions", "selections", "selected_repairs"):
            if key in selection_metadata:
                selected_fractions[key] = selection_metadata[key]

    metadata = {
        "phase": 5,
        "description": "Frozen final audit on untouched test split",
        "config": asdict(config),
        "test_file_sha256": file_sha256(config.test_path),
        "test_sample_ids_sha256": hashlib.sha256(
            "\n".join(test_df["sample_id"].astype(str).tolist()).encode("utf-8")
        ).hexdigest(),
        "n_test_examples": int(len(test_df)),
        "group_counts": {
            str(key): int(value)
            for key, value in test_df["demo"].value_counts().to_dict().items()
        },
        "label_counts": {
            str(key): int(value)
            for key, value in test_df["true_quality"].value_counts().to_dict().items()
        },
        "model_paths": model_paths,
        "selection_metadata_path": config.selection_metadata_path,
        "selection_metadata_snapshot": selection_metadata,
        "selected_fraction_summary": selected_fractions,
        "test_set_used_for_selection": False,
        "threshold_tuned_on_test": False,
        "decision_threshold": config.decision_threshold,
        "runtime_seconds": float(time.time() - start_time),
    }
    with open(config.metadata_output, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    display_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "smooth_eo_surrogate",
        "equalized_odds_difference",
        "tpr_gap",
        "fpr_gap",
        "pass_rate_gap",
    ]
    print("\nFinal untouched-test summary:")
    print(metrics_df[display_columns].to_string(index=False))

    print("\nPrediction changes relative to biased baseline:")
    print(changes_df.to_string(index=False))

    print("\nSaved outputs:")
    for path in (
        config.metrics_output,
        config.confusions_output,
        config.changes_output,
        config.bootstrap_output,
        config.predictions_output,
        config.metadata_output,
    ):
        print(f"- {path}")

    print(f"\nPhase 5 completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()