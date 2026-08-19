"""
Phase 4 - OULAD repair selection and frozen model creation

Corresponds to Section 4.5 (selection) of the thesis.

Natural-disparity OULAD experiment. This phase preserves the same logic as
Phase 4 in the synthetic pipeline while respecting the real-data setting:

- no artificial corruption labels,
- no oracle model,
- no use of the untouched test split,
- fairness selection on the exact Phase-2 fairness probe,
- predictive-utility constraints on the separate validation split.

The script consumes Phase-3 native-update path results, formally selects one
repair fraction per oracle-free method, reconstructs the selected parameter
updates from Phase 2, verifies the selected metrics by fresh inference, and
saves frozen model checkpoints for the final audit.

Selection rule
--------------
For each method, select the candidate with the lowest probe smooth-EO
surrogate subject to both constraints:

    validation balanced accuracy >= baseline - tolerance
    validation macro-F1          >= baseline - tolerance

The zero-update candidate is always included as a safe fallback when present.
The untouched test split is never loaded.

Expected layout
---------------
project/
├── 01_oulad_data_and_training.py
├── 02_oulad_influence_diagnostics.py
├── 03_oulad_direction_comparison.py
├── 04_oulad_repair_selection.py
├── oulad_phase1/
├── oulad_phase2/
└── oulad_phase3/

Run
---
python 04_oulad_repair_selection.py
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
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
    phase2_dir: str = ""
    phase3_dir: str = ""
    output_dir: str = ""
    seed: int = 42
    device: str = "auto"
    batch_size: int = 512
    threshold: float = 0.50
    utility_tolerance: float = 0.02
    fairness_tie_tolerance: float = 1e-12
    comparison_mode: str = "native_update"


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
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")

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


def vector_norm(vector: Sequence[Tensor]) -> float:
    if not vector:
        return 0.0
    total = sum(torch.sum(item.detach().float() ** 2) for item in vector)
    return float(torch.sqrt(total).item())


def apply_direction_in_place(
    model: nn.Module,
    parameter_names: Sequence[str],
    direction: Sequence[Tensor],
    scale: float,
    device: torch.device,
) -> None:
    parameter_map = dict(model.named_parameters())
    with torch.no_grad():
        for name, update in zip(parameter_names, direction):
            parameter_map[name].add_(
                update.to(device=device, dtype=parameter_map[name].dtype),
                alpha=float(scale),
            )


def build_candidate(
    base_model: nn.Module,
    parameter_names: Sequence[str],
    direction: Sequence[Tensor],
    scale: float,
    device: torch.device,
) -> nn.Module:
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    apply_direction_in_place(model, parameter_names, direction, scale, device)
    return model


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
            parts.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(parts, axis=0)


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
    base_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
    threshold: float,
) -> Dict[str, int | float]:
    base_predictions = (base_probabilities >= threshold).astype(np.int64)
    candidate_predictions = (candidate_probabilities >= threshold).astype(np.int64)
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
        "mean_probability_change": float(
            np.mean(candidate_probabilities - base_probabilities)
        ),
        "mean_absolute_probability_change": float(
            np.mean(np.abs(candidate_probabilities - base_probabilities))
        ),
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


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


def save_model_checkpoint(
    output_path: Path,
    model: nn.Module,
    base_checkpoint: Mapping[str, object],
    method: str,
    selected_fraction: float,
    applied_scale: float,
    parameter_distance: float,
    parameter_names: Sequence[str],
    selection_metrics: Mapping[str, object],
    config: Config,
) -> None:
    checkpoint = dict(base_checkpoint)
    checkpoint["model_state_dict"] = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    checkpoint.update(
        {
            "model_role": "selected_oracle_free_fairness_repair",
            "repair_method": method,
            "selected_fraction": float(selected_fraction),
            "applied_scale": float(applied_scale),
            "parameter_distance": float(parameter_distance),
            "updated_parameter_names": list(parameter_names),
            "selection_split_fairness": "fairness_probe",
            "selection_split_utility": "validation",
            "test_set_used_for_selection": False,
            "selection_metrics": dict(selection_metrics),
            "phase4_config": asdict(config),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OULAD Phase 4: repair selection")
    parser.add_argument("--phase1-dir", default=None)
    parser.add_argument("--phase2-dir", default=None)
    parser.add_argument("--phase3-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--utility-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    phase1_dir = (
        Path(args.phase1_dir).expanduser().resolve()
        if args.phase1_dir
        else script_dir / "oulad_phase1"
    )
    phase2_dir = (
        Path(args.phase2_dir).expanduser().resolve()
        if args.phase2_dir
        else script_dir / "oulad_phase2"
    )
    phase3_dir = (
        Path(args.phase3_dir).expanduser().resolve()
        if args.phase3_dir
        else script_dir / "oulad_phase3"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "oulad_phase4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        phase1_dir=str(phase1_dir),
        phase2_dir=str(phase2_dir),
        phase3_dir=str(phase3_dir),
        output_dir=str(output_dir),
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        utility_tolerance=args.utility_tolerance,
    )
    set_seed(config.seed)
    device = choose_device(config.device)
    print(f"Device: {device}")

    paths = {
        "checkpoint": phase1_dir / "m0_oulad.pt",
        "validation_npz": phase1_dir / "validation_features.npz",
        "probe_npz": phase1_dir / "fairness_probe_features.npz",
        "directions": phase2_dir / "fairness_unlearning_directions.pt",
        "probe_gradient": phase2_dir / "fairness_probe_gradient.pt",
        "path_metrics": phase3_dir / "direction_path_metrics.csv",
    }
    require_files(list(paths.values()))

    started = time.time()
    base_model, base_checkpoint = load_model(paths["checkpoint"], device)
    direction_payload = torch.load(paths["directions"], map_location="cpu")
    gradient_payload = torch.load(paths["probe_gradient"], map_location="cpu")
    parameter_names, directions = validate_direction_payload(direction_payload, base_model)

    validation_features, validation_labels, validation_demo, _ = load_npz(
        paths["validation_npz"]
    )
    probe_features, probe_labels, probe_demo, probe_ids = load_npz(paths["probe_npz"])

    saved_probe_ids = np.asarray(gradient_payload.get("probe_sample_ids", []), dtype=str)
    if len(saved_probe_ids) == 0:
        raise ValueError("Phase 2 gradient payload contains no probe_sample_ids.")
    probe_index = {sample_id: index for index, sample_id in enumerate(probe_ids)}
    missing_ids = [sample_id for sample_id in saved_probe_ids if sample_id not in probe_index]
    if missing_ids:
        raise ValueError(
            "Some Phase 2 probe IDs are absent from Phase 1 probe data: "
            f"{missing_ids[:5]}"
        )
    selected_indices = np.asarray([probe_index[x] for x in saved_probe_ids], dtype=np.int64)
    selected_probe_features = probe_features[selected_indices]
    selected_probe_labels = probe_labels[selected_indices]
    selected_probe_demo = probe_demo[selected_indices]

    base_probe_probabilities = predict_probabilities(
        base_model, selected_probe_features, config.batch_size, device
    )
    base_validation_probabilities = predict_probabilities(
        base_model, validation_features, config.batch_size, device
    )
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
    print("\nPhase-2/Phase-4 fairness consistency check:")
    print(f"Phase 2 saved fairness loss : {saved_fairness_loss:.10f}")
    print(
        "Phase 4 reproduced loss     : "
        f"{base_probe_metrics['smooth_eo_surrogate']:.10f}"
    )
    print(f"Absolute difference         : {consistency_difference:.10f}")
    if consistency_difference > 1e-6:
        raise ValueError(
            "Phase 2 and Phase 4 fairness losses do not match. "
            "Check probe IDs and preprocessing."
        )

    path_metrics = pd.read_csv(paths["path_metrics"])
    required_columns = {
        "comparison_mode",
        "method",
        "path_fraction",
        "applied_scale",
        "parameter_distance",
        "probe_smooth_eo_after",
        "validation_balanced_accuracy_before",
        "validation_balanced_accuracy_after",
        "validation_macro_f1_before",
        "validation_macro_f1_after",
    }
    missing_columns = required_columns.difference(path_metrics.columns)
    if missing_columns:
        raise ValueError(
            f"Phase 3 path metrics are missing columns: {sorted(missing_columns)}"
        )

    candidates = path_metrics[
        path_metrics["comparison_mode"] == config.comparison_mode
    ].copy()
    candidates = candidates[candidates["method"].isin(directions)].copy()
    if candidates.empty:
        raise ValueError("No native-update candidates found for Phase 4 selection.")

    baseline_ba = float(base_validation_metrics["balanced_accuracy"])
    baseline_f1 = float(base_validation_metrics["macro_f1"])
    minimum_ba = baseline_ba - config.utility_tolerance
    minimum_f1 = baseline_f1 - config.utility_tolerance

    candidates["balanced_accuracy_constraint"] = (
        candidates["validation_balanced_accuracy_after"] >= minimum_ba
    )
    candidates["macro_f1_constraint"] = (
        candidates["validation_macro_f1_after"] >= minimum_f1
    )
    candidates["accepted"] = (
        candidates["balanced_accuracy_constraint"]
        & candidates["macro_f1_constraint"]
    )

    def rejection_reason(row: pd.Series) -> str:
        reasons: List[str] = []
        if not bool(row["balanced_accuracy_constraint"]):
            reasons.append("validation_balanced_accuracy_below_threshold")
        if not bool(row["macro_f1_constraint"]):
            reasons.append("validation_macro_f1_below_threshold")
        return "" if not reasons else ";".join(reasons)

    candidates["rejection_reason"] = candidates.apply(rejection_reason, axis=1)
    candidates["selected"] = False

    selected_rows: List[pd.Series] = []
    for method in directions:
        method_candidates = candidates[candidates["method"] == method].copy()
        feasible = method_candidates[method_candidates["accepted"]].copy()
        if feasible.empty:
            zero_candidates = method_candidates[
                np.isclose(method_candidates["path_fraction"].astype(float), 0.0)
            ]
            if zero_candidates.empty:
                raise ValueError(
                    f"No feasible candidate and no zero-update fallback for {method}."
                )
            feasible = zero_candidates.copy()

        minimum_fairness = float(feasible["probe_smooth_eo_after"].min())
        near_best = feasible[
            feasible["probe_smooth_eo_after"]
            <= minimum_fairness + config.fairness_tie_tolerance
        ].copy()
        # Deterministic tie-breaking: smallest parameter movement, then smallest fraction.
        chosen = near_best.sort_values(
            ["parameter_distance", "path_fraction"], ascending=[True, True]
        ).iloc[0]
        selected_rows.append(chosen)
        candidates.loc[chosen.name, "selected"] = True

    candidates = candidates.sort_values(["method", "path_fraction"]).reset_index(drop=True)
    candidates.to_csv(output_dir / "repair_candidates.csv", index=False)

    print("\nSelection constraints")
    print(f"Baseline validation balanced accuracy : {baseline_ba:.6f}")
    print(f"Minimum accepted balanced accuracy    : {minimum_ba:.6f}")
    print(f"Baseline validation macro-F1          : {baseline_f1:.6f}")
    print(f"Minimum accepted macro-F1             : {minimum_f1:.6f}")

    selection_records: List[Dict[str, object]] = []
    selected_updates: Dict[str, object] = {
        "parameter_names": list(parameter_names),
        "test_set_used_for_selection": False,
        "methods": {},
    }

    print("\nSelected repairs and fresh verification:")
    for chosen in selected_rows:
        method = str(chosen["method"])
        fraction = float(chosen["path_fraction"])
        applied_scale = float(chosen["applied_scale"])
        parameter_distance = float(chosen["parameter_distance"])
        direction = directions[method]

        repaired_model = build_candidate(
            base_model,
            parameter_names,
            direction,
            applied_scale,
            device,
        )
        probe_probabilities = predict_probabilities(
            repaired_model,
            selected_probe_features,
            config.batch_size,
            device,
        )
        validation_probabilities = predict_probabilities(
            repaired_model,
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
        probe_changes = changed_prediction_metrics(
            base_probe_probabilities,
            probe_probabilities,
            selected_probe_labels,
            selected_probe_demo,
            config.threshold,
        )
        validation_changes = changed_prediction_metrics(
            base_validation_probabilities,
            validation_probabilities,
            validation_labels,
            validation_demo,
            config.threshold,
        )

        tolerance = 1e-6
        checks = {
            "probe_smooth_eo": abs(
                probe_metrics["smooth_eo_surrogate"]
                - float(chosen["probe_smooth_eo_after"])
            ),
            "validation_balanced_accuracy": abs(
                validation_metrics["balanced_accuracy"]
                - float(chosen["validation_balanced_accuracy_after"])
            ),
            "validation_macro_f1": abs(
                validation_metrics["macro_f1"]
                - float(chosen["validation_macro_f1_after"])
            ),
        }
        if any(value > tolerance for value in checks.values()):
            raise ValueError(
                f"Fresh verification does not match Phase 3 for {method}: {checks}"
            )

        model_subdir = (
            "m1_influence_repaired"
            if method == "influence_hessian"
            else "m1_raw_gradient_repaired"
        )
        model_dir = models_dir / model_subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "model.pt"

        selection_metrics = {
            "probe": probe_metrics,
            "validation": validation_metrics,
            "probe_changes": probe_changes,
            "validation_changes": validation_changes,
        }
        save_model_checkpoint(
            checkpoint_path,
            repaired_model,
            base_checkpoint,
            method,
            fraction,
            applied_scale,
            parameter_distance,
            parameter_names,
            selection_metrics,
            config,
        )

        scaled_update = [
            (tensor * applied_scale).detach().cpu().float()
            for tensor in direction
        ]
        selected_updates["methods"][method] = {
            "selected_fraction": fraction,
            "applied_scale": applied_scale,
            "parameter_distance": parameter_distance,
            "direction_norm": vector_norm(direction),
            "scaled_update": scaled_update,
            "model_checkpoint": str(checkpoint_path),
        }

        record: Dict[str, object] = {
            "method": method,
            "selected_fraction": fraction,
            "applied_scale": applied_scale,
            "direction_norm": vector_norm(direction),
            "parameter_distance": parameter_distance,
            "probe_smooth_eo_surrogate": probe_metrics["smooth_eo_surrogate"],
            "probe_equalized_odds_difference": probe_metrics[
                "equalized_odds_difference"
            ],
            "probe_TPR_gap": probe_metrics["TPR_gap"],
            "probe_FPR_gap": probe_metrics["FPR_gap"],
            "probe_selection_rate_gap": probe_metrics["selection_rate_gap"],
            "probe_changed_predictions": probe_changes["changed_predictions"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_average_precision": validation_metrics["average_precision"],
            "validation_equalized_odds_difference": validation_metrics[
                "equalized_odds_difference"
            ],
            "validation_changed_predictions": validation_changes[
                "changed_predictions"
            ],
            "output_model_checkpoint": str(checkpoint_path),
        }
        selection_records.append(record)

        print(
            f"{method:25s} | fraction={fraction:7.4g} | "
            f"probe smooth EO={probe_metrics['smooth_eo_surrogate']:.10f} | "
            f"probe hard EO={probe_metrics['equalized_odds_difference']:.6f} | "
            f"validation BA={validation_metrics['balanced_accuracy']:.6f} | "
            f"validation F1={validation_metrics['macro_f1']:.6f}"
        )
        print(f"  saved: {checkpoint_path}")

        del repaired_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selection_df = pd.DataFrame(selection_records).sort_values("method")
    selection_df.to_csv(output_dir / "repair_selection.csv", index=False)
    torch.save(selected_updates, output_dir / "selected_repair_updates.pt")

    # Preserve an explicit frozen baseline checkpoint for Phase 5 convenience.
    baseline_dir = models_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["checkpoint"], baseline_dir / "model.pt")

    runtime = time.time() - started
    metadata = {
        **asdict(config),
        "resolved_device": str(device),
        "runtime_seconds": runtime,
        "phase2_saved_fairness_loss": saved_fairness_loss,
        "phase4_reproduced_fairness_loss": base_probe_metrics[
            "smooth_eo_surrogate"
        ],
        "fairness_consistency_absolute_difference": consistency_difference,
        "baseline_probe_metrics": base_probe_metrics,
        "baseline_validation_metrics": base_validation_metrics,
        "minimum_validation_balanced_accuracy": minimum_ba,
        "minimum_validation_macro_f1": minimum_f1,
        "selected_parameter_names": list(parameter_names),
        "selected_methods": selection_df.to_dict(orient="records"),
        "selection_principle": (
            "minimum probe smooth EO subject to validation balanced-accuracy "
            "and macro-F1 utility constraints"
        ),
        "test_set_used_for_selection": False,
        "untouched_test_loaded": False,
    }
    write_json(output_dir / "repair_selection_metadata.json", metadata)

    print("\nSaved outputs:")
    for name in (
        "repair_candidates.csv",
        "repair_selection.csv",
        "selected_repair_updates.pt",
        "repair_selection_metadata.json",
    ):
        print(f"- {output_dir / name}")
    print(f"- {baseline_dir / 'model.pt'}")
    for record in selection_records:
        print(f"- {record['output_model_checkpoint']}")

    print(f"\nPhase 4 completed in {runtime:.2f} seconds.")
    print("The untouched test split was not loaded or used for selection.")


if __name__ == "__main__":
    main()