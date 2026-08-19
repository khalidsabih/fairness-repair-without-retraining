"""
Phase 3 - xAPI-Edu-Data direction comparison

Corresponds to Section 4.5 of the thesis.

Natural-disparity xAPI-Edu-Data experiment. This phase does not use artificial
corruption labels, an oracle model, or the untouched test split.

It compares two oracle-free parameter-space repair directions produced by
Phase 2:

    raw_fairness_gradient = -g_fair
    influence_hessian     = -H^{-1} g_fair

The script evaluates two path families:

1. native_update
   theta(alpha) = theta_0 + alpha * d

2. common_distance
   theta(alpha) = theta_0 + alpha * r * d / ||d||
   where r is the native norm of the influence-Hessian direction.

Fairness is evaluated on the exact balanced fairness-probe subset used in
Phase 2. Predictive utility is evaluated on the separate validation split.
The untouched test split is never loaded.

Expected layout
---------------
project/
├── 01_xapi_data_and_training.py
├── 02_xapi_influence_diagnostics_fixed.py
├── 03_xapi_direction_comparison.py
├── xapi_phase1/
└── xapi_phase2/

Run
---
python 03_xapi_direction_comparison.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
    phase2_dir: str = ""
    output_dir: str = ""
    seed: int = 42
    device: str = "auto"
    batch_size: int = 512
    threshold: float = 0.50

    common_distance_fractions: Tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
        2.0,
        5.0,
        10.0,
    )
    native_update_fractions: Tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
        2.0,
        5.0,
        10.0,
    )


class XAPIMLP(nn.Module):
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


def load_model(path: Path, device: torch.device) -> Tuple[XAPIMLP, dict]:
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
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")

    model = XAPIMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim_1=int(checkpoint["hidden_dim_1"]),
        hidden_dim_2=int(checkpoint["hidden_dim_2"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def vector_norm(vector: Sequence[Tensor]) -> float:
    if not vector:
        return 0.0
    total = sum(torch.sum(x.detach().float() ** 2) for x in vector)
    return float(torch.sqrt(total).item())


def vector_dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector lengths do not match.")
    return float(sum(torch.sum(x.float() * y.float()) for x, y in zip(a, b)).item())


def cosine_similarity(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    denominator = vector_norm(a) * vector_norm(b)
    return vector_dot(a, b) / denominator if denominator > 0 else float("nan")


def validate_direction_payload(
    payload: Mapping[str, object],
    model: nn.Module,
) -> Tuple[List[str], Dict[str, List[Tensor]]]:
    required = {
        "parameter_names",
        "influence_repair_direction",
        "raw_fairness_repair_direction",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Phase 2 direction payload is missing: {sorted(missing)}")

    names = [str(name) for name in payload["parameter_names"]]
    parameter_map = dict(model.named_parameters())
    unknown = [name for name in names if name not in parameter_map]
    if unknown:
        raise ValueError(f"Direction parameter names not found in model: {unknown}")

    directions: Dict[str, List[Tensor]] = {}
    for method, key in (
        ("influence_hessian", "influence_repair_direction"),
        ("raw_fairness_gradient", "raw_fairness_repair_direction"),
    ):
        tensors = [tensor.detach().cpu().float() for tensor in payload[key]]
        if len(tensors) != len(names):
            raise ValueError(f"{method} tensor count does not match parameter names.")
        for name, tensor in zip(names, tensors):
            expected = parameter_map[name].shape
            if tensor.shape != expected:
                raise ValueError(
                    f"Shape mismatch for {method}:{name}: {tensor.shape} != {expected}"
                )
        directions[method] = tensors

    return names, directions


def apply_direction(
    base_model: nn.Module,
    parameter_names: Sequence[str],
    direction: Sequence[Tensor],
    scale: float,
    device: torch.device,
) -> nn.Module:
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    parameter_map = dict(model.named_parameters())
    with torch.no_grad():
        for name, update in zip(parameter_names, direction):
            parameter_map[name].add_(update.to(device=device, dtype=parameter_map[name].dtype), alpha=float(scale))
    return model


def predict(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        torch.as_tensor(features, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits_parts.append(model(batch.to(device)).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 1]
    margins = logits[:, 1] - logits[:, 0]
    return logits, probabilities, margins


def safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def group_rates(
    labels: np.ndarray,
    predictions: np.ndarray,
    demo: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for group in ("A", "B"):
        mask = demo == group
        y = labels[mask]
        p = predictions[mask]
        tp = int(np.sum((y == 1) & (p == 1)))
        fn = int(np.sum((y == 1) & (p == 0)))
        fp = int(np.sum((y == 0) & (p == 1)))
        tn = int(np.sum((y == 0) & (p == 0)))
        result[f"n_{group}"] = int(mask.sum())
        result[f"TPR_{group}"] = tp / (tp + fn) if tp + fn else float("nan")
        result[f"FPR_{group}"] = fp / (fp + tn) if fp + tn else float("nan")
        result[f"FNR_{group}"] = fn / (tp + fn) if tp + fn else float("nan")
        result[f"selection_rate_{group}"] = float(np.mean(p)) if len(p) else float("nan")

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
    cell_means: Dict[str, float] = {}
    for label in (0, 1):
        means: Dict[str, float] = {}
        for group in ("A", "B"):
            mask = (labels == label) & (demo == group)
            if not np.any(mask):
                raise ValueError(f"Empty fairness cell {group}_{label}.")
            value = float(np.mean(probabilities[mask]))
            means[group] = value
            cell_means[f"mean_probability_{group}_y{label}"] = value
        terms.append((means["A"] - means["B"]) ** 2)
    return float(sum(terms)), cell_means


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
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": safe_roc_auc(labels, probabilities),
        "equalized_odds_difference": float(
            equalized_odds_difference(labels, predictions, sensitive_features=demo)
        ),
    }
    smooth, cell_means = smooth_eo_surrogate(probabilities, labels, demo)
    metrics["smooth_eo_surrogate"] = smooth
    metrics.update(group_rates(labels, predictions, demo))
    metrics.update(cell_means)
    return metrics


def changed_prediction_metrics(
    base_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
) -> Dict[str, int | float]:
    changed = candidate_predictions != base_predictions
    beneficial = changed & (candidate_predictions == labels) & (base_predictions != labels)
    harmful = changed & (candidate_predictions != labels) & (base_predictions == labels)
    result: Dict[str, int | float] = {
        "changed_predictions": int(changed.sum()),
        "fraction_changed": float(changed.mean()),
        "flips_0_to_1": int(np.sum((base_predictions == 0) & (candidate_predictions == 1))),
        "flips_1_to_0": int(np.sum((base_predictions == 1) & (candidate_predictions == 0))),
        "beneficial_flips": int(beneficial.sum()),
        "harmful_flips": int(harmful.sum()),
    }
    for group in ("A", "B"):
        mask = demo == group
        result[f"changed_predictions_{group}"] = int(np.sum(changed & mask))
        result[f"{group}_0_to_1"] = int(
            np.sum(mask & (base_predictions == 0) & (candidate_predictions == 1))
        )
        result[f"{group}_1_to_0"] = int(
            np.sum(mask & (base_predictions == 1) & (candidate_predictions == 0))
        )
    return result


def parse_float_grid(value: str) -> Tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid fraction grid: {value}") from error
    if not result:
        raise argparse.ArgumentTypeError("Fraction grid cannot be empty.")
    if any(x < 0 or not math.isfinite(x) for x in result):
        raise argparse.ArgumentTypeError("Fractions must be finite and non-negative.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xAPI Phase 3: direction comparison")
    parser.add_argument("--phase1-dir", default=None)
    parser.add_argument("--phase2-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--common-fractions",
        type=parse_float_grid,
        default=parse_float_grid("0,0.25,0.5,0.75,1,2,5,10"),
    )
    parser.add_argument(
        "--native-fractions",
        type=parse_float_grid,
        default=parse_float_grid("0,0.25,0.5,0.75,1,2,5,10"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    phase1_dir = (
        Path(args.phase1_dir).expanduser().resolve()
        if args.phase1_dir
        else script_dir / "xapi_phase1"
    )
    phase2_dir = (
        Path(args.phase2_dir).expanduser().resolve()
        if args.phase2_dir
        else script_dir / "xapi_phase2"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "xapi_phase3"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        phase1_dir=str(phase1_dir),
        phase2_dir=str(phase2_dir),
        output_dir=str(output_dir),
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        common_distance_fractions=args.common_fractions,
        native_update_fractions=args.native_fractions,
    )

    set_seed(config.seed)
    device = choose_device(config.device)
    print(f"Device: {device}")

    paths = {
        "checkpoint": phase1_dir / "m0_xapi.pt",
        "validation_npz": phase1_dir / "validation_features.npz",
        "probe_npz": phase1_dir / "fairness_probe_features.npz",
        "directions": phase2_dir / "fairness_unlearning_directions.pt",
        "probe_gradient": phase2_dir / "fairness_probe_gradient.pt",
    }
    require_files(list(paths.values()))

    start_time = time.time()
    base_model, checkpoint = load_model(paths["checkpoint"], device)
    direction_payload = torch.load(paths["directions"], map_location="cpu")
    gradient_payload = torch.load(paths["probe_gradient"], map_location="cpu")
    parameter_names, directions = validate_direction_payload(direction_payload, base_model)

    validation_features, validation_labels, validation_demo, validation_ids = load_npz(
        paths["validation_npz"]
    )
    probe_features, probe_labels, probe_demo, probe_ids = load_npz(paths["probe_npz"])

    saved_probe_ids = np.asarray(gradient_payload.get("probe_sample_ids", []), dtype=str)
    if len(saved_probe_ids) == 0:
        raise ValueError("Phase 2 gradient payload contains no probe_sample_ids.")
    probe_index_map = {sample_id: index for index, sample_id in enumerate(probe_ids)}
    missing_probe_ids = [sample_id for sample_id in saved_probe_ids if sample_id not in probe_index_map]
    if missing_probe_ids:
        raise ValueError(
            "Some Phase 2 probe sample IDs are absent from Phase 1 probe data: "
            f"{missing_probe_ids[:5]}"
        )
    selected_indices = np.asarray([probe_index_map[x] for x in saved_probe_ids], dtype=np.int64)
    selected_probe_features = probe_features[selected_indices]
    selected_probe_labels = probe_labels[selected_indices]
    selected_probe_demo = probe_demo[selected_indices]
    selected_probe_ids = probe_ids[selected_indices]

    base_probe_logits, base_probe_probabilities, base_probe_margins = predict(
        base_model,
        selected_probe_features,
        config.batch_size,
        device,
    )
    base_validation_logits, base_validation_probabilities, base_validation_margins = predict(
        base_model,
        validation_features,
        config.batch_size,
        device,
    )
    base_probe_predictions = (base_probe_probabilities >= config.threshold).astype(np.int64)
    base_validation_predictions = (
        base_validation_probabilities >= config.threshold
    ).astype(np.int64)
    base_probe_metrics = evaluate_split(
        base_probe_probabilities,
        selected_probe_labels,
        selected_probe_demo,
        config.threshold,
    )
    base_validation_metrics = evaluate_split(
        base_validation_probabilities,
        validation_labels,
        validation_demo,
        config.threshold,
    )

    saved_fairness_loss = float(gradient_payload["fairness_loss"])
    consistency_difference = abs(
        base_probe_metrics["smooth_eo_surrogate"] - saved_fairness_loss
    )
    print("\nPhase-2/Phase-3 fairness consistency check:")
    print(f"Phase 2 saved fairness loss : {saved_fairness_loss:.10f}")
    print(
        "Phase 3 reproduced loss     : "
        f"{base_probe_metrics['smooth_eo_surrogate']:.10f}"
    )
    print(f"Absolute difference         : {consistency_difference:.10f}")
    if consistency_difference > 1e-6:
        raise ValueError(
            "Phase 2 and Phase 3 fairness losses do not match. "
            "Check probe IDs, preprocessing, and the fairness objective."
        )

    direction_rows: List[Dict[str, object]] = []
    for method, direction in directions.items():
        direction_rows.append(
            {
                "method": method,
                "n_parameter_tensors": len(direction),
                "n_parameter_elements": int(sum(x.numel() for x in direction)),
                "direction_norm": vector_norm(direction),
            }
        )
    direction_rows.append(
        {
            "method": "raw_vs_influence",
            "n_parameter_tensors": len(parameter_names),
            "n_parameter_elements": int(
                sum(dict(base_model.named_parameters())[name].numel() for name in parameter_names)
            ),
            "direction_norm": float("nan"),
            "cosine_similarity": cosine_similarity(
                directions["raw_fairness_gradient"],
                directions["influence_hessian"],
            ),
        }
    )
    direction_summary = pd.DataFrame(direction_rows)
    direction_summary.to_csv(output_dir / "direction_summary.csv", index=False)
    print("\nDirection summary:")
    print(direction_summary.to_string(index=False))

    reference_radius = vector_norm(directions["influence_hessian"])
    if reference_radius <= 0:
        raise ValueError("Influence-Hessian direction has zero norm.")
    print(f"\nCommon-distance reference radius: {reference_radius:.10f}")

    rows: List[Dict[str, object]] = []
    functional_rows: List[Dict[str, object]] = []

    path_definitions = [
        ("common_distance", config.common_distance_fractions),
        ("native_update", config.native_update_fractions),
    ]

    for comparison_mode, fractions in path_definitions:
        print(f"\nRunning {comparison_mode} comparison...")
        for method, direction in directions.items():
            norm = vector_norm(direction)
            if norm <= 0:
                raise ValueError(f"{method} has zero direction norm.")
            for fraction in fractions:
                if comparison_mode == "common_distance":
                    applied_scale = float(fraction * reference_radius / norm)
                    parameter_distance = float(fraction * reference_radius)
                else:
                    applied_scale = float(fraction)
                    parameter_distance = float(fraction * norm)

                candidate = apply_direction(
                    base_model,
                    parameter_names,
                    direction,
                    applied_scale,
                    device,
                )
                _, probe_probabilities, probe_margins = predict(
                    candidate,
                    selected_probe_features,
                    config.batch_size,
                    device,
                )
                _, validation_probabilities, validation_margins = predict(
                    candidate,
                    validation_features,
                    config.batch_size,
                    device,
                )

                probe_metrics = evaluate_split(
                    probe_probabilities,
                    selected_probe_labels,
                    selected_probe_demo,
                    config.threshold,
                )
                validation_metrics = evaluate_split(
                    validation_probabilities,
                    validation_labels,
                    validation_demo,
                    config.threshold,
                )
                probe_predictions = (
                    probe_probabilities >= config.threshold
                ).astype(np.int64)
                validation_predictions = (
                    validation_probabilities >= config.threshold
                ).astype(np.int64)

                probe_changes = changed_prediction_metrics(
                    base_probe_predictions,
                    probe_predictions,
                    selected_probe_labels,
                    selected_probe_demo,
                )
                validation_changes = changed_prediction_metrics(
                    base_validation_predictions,
                    validation_predictions,
                    validation_labels,
                    validation_demo,
                )

                row: Dict[str, object] = {
                    "comparison_mode": comparison_mode,
                    "method": method,
                    "path_fraction": float(fraction),
                    "applied_scale": applied_scale,
                    "parameter_distance": parameter_distance,
                    "direction_norm": norm,
                    "probe_smooth_eo_before": base_probe_metrics["smooth_eo_surrogate"],
                    "probe_smooth_eo_after": probe_metrics["smooth_eo_surrogate"],
                    "probe_smooth_eo_change": (
                        probe_metrics["smooth_eo_surrogate"]
                        - base_probe_metrics["smooth_eo_surrogate"]
                    ),
                    "probe_equalized_odds_before": base_probe_metrics[
                        "equalized_odds_difference"
                    ],
                    "probe_equalized_odds_after": probe_metrics[
                        "equalized_odds_difference"
                    ],
                    "probe_TPR_gap_after": probe_metrics["TPR_gap"],
                    "probe_FPR_gap_after": probe_metrics["FPR_gap"],
                    "probe_selection_rate_gap_after": probe_metrics[
                        "selection_rate_gap"
                    ],
                    "validation_accuracy_before": base_validation_metrics["accuracy"],
                    "validation_accuracy_after": validation_metrics["accuracy"],
                    "validation_balanced_accuracy_before": base_validation_metrics[
                        "balanced_accuracy"
                    ],
                    "validation_balanced_accuracy_after": validation_metrics[
                        "balanced_accuracy"
                    ],
                    "validation_macro_f1_before": base_validation_metrics["macro_f1"],
                    "validation_macro_f1_after": validation_metrics["macro_f1"],
                    "validation_roc_auc_before": base_validation_metrics["roc_auc"],
                    "validation_roc_auc_after": validation_metrics["roc_auc"],
                    "validation_average_precision_before": base_validation_metrics[
                        "average_precision"
                    ],
                    "validation_average_precision_after": validation_metrics[
                        "average_precision"
                    ],
                    "validation_equalized_odds_after": validation_metrics[
                        "equalized_odds_difference"
                    ],
                }
                row.update({f"probe_{k}": v for k, v in probe_changes.items()})
                row.update({f"validation_{k}": v for k, v in validation_changes.items()})
                rows.append(row)

                probe_probability_change = probe_probabilities - base_probe_probabilities
                probe_margin_change = probe_margins - base_probe_margins
                validation_probability_change = (
                    validation_probabilities - base_validation_probabilities
                )
                validation_margin_change = validation_margins - base_validation_margins
                functional_rows.append(
                    {
                        "comparison_mode": comparison_mode,
                        "method": method,
                        "path_fraction": float(fraction),
                        "parameter_distance": parameter_distance,
                        "mean_abs_probe_probability_change": float(
                            np.mean(np.abs(probe_probability_change))
                        ),
                        "max_abs_probe_probability_change": float(
                            np.max(np.abs(probe_probability_change))
                        ),
                        "mean_abs_probe_margin_change": float(
                            np.mean(np.abs(probe_margin_change))
                        ),
                        "mean_abs_validation_probability_change": float(
                            np.mean(np.abs(validation_probability_change))
                        ),
                        "max_abs_validation_probability_change": float(
                            np.max(np.abs(validation_probability_change))
                        ),
                        "mean_abs_validation_margin_change": float(
                            np.mean(np.abs(validation_margin_change))
                        ),
                    }
                )

                del candidate
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    path_metrics = pd.DataFrame(rows)
    functional_metrics = pd.DataFrame(functional_rows)
    path_metrics.to_csv(output_dir / "direction_path_metrics.csv", index=False)
    functional_metrics.to_csv(
        output_dir / "direction_functional_diagnostics.csv", index=False
    )

    native_summary = (
        path_metrics[path_metrics["comparison_mode"] == "native_update"]
        .sort_values(["method", "probe_smooth_eo_after", "path_fraction"])
        .groupby("method", as_index=False)
        .first()
    )
    native_summary.to_csv(output_dir / "native_path_best_probe_points.csv", index=False)

    print("\nBest native probe point per method (descriptive only; Phase 4 selects formally):")
    columns = [
        "method",
        "path_fraction",
        "parameter_distance",
        "probe_smooth_eo_after",
        "probe_equalized_odds_after",
        "validation_balanced_accuracy_after",
        "validation_macro_f1_after",
        "probe_changed_predictions",
    ]
    print(native_summary[columns].to_string(index=False))

    runtime = time.time() - start_time
    metadata = {
        **asdict(config),
        "resolved_device": str(device),
        "runtime_seconds": runtime,
        "selected_parameter_names": parameter_names,
        "phase2_saved_fairness_loss": saved_fairness_loss,
        "phase3_reproduced_fairness_loss": base_probe_metrics[
            "smooth_eo_surrogate"
        ],
        "fairness_consistency_absolute_difference": consistency_difference,
        "common_distance_reference_radius": reference_radius,
        "raw_vs_influence_cosine": cosine_similarity(
            directions["raw_fairness_gradient"],
            directions["influence_hessian"],
        ),
        "probe_sample_ids": selected_probe_ids.tolist(),
        "validation_size": int(len(validation_features)),
        "balanced_probe_size": int(len(selected_probe_features)),
        "test_split_loaded": False,
        "oracle_used": False,
        "corruption_metadata_used": False,
        "selection_performed": False,
        "interpretation": (
            "Phase 3 compares oracle-free repair paths. Phase 4 must perform "
            "formal model selection using probe fairness and validation utility."
        ),
    }
    with (output_dir / "direction_comparison_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)

    print("\nSaved outputs:")
    for name in (
        "direction_summary.csv",
        "direction_path_metrics.csv",
        "direction_functional_diagnostics.csv",
        "native_path_best_probe_points.csv",
        "direction_comparison_metadata.json",
    ):
        print(f"- {output_dir / name}")
    print(f"\nPhase 3 completed in {runtime:.2f} seconds.")
    print("The untouched test split was not loaded.")


if __name__ == "__main__":
    main()