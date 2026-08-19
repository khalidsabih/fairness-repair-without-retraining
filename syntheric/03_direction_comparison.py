"""
Phase 3 : Compare oracle-free fairness repair directions.

Corresponds to Section 4.5 of the thesis.

This script evaluates four directions in the exact parameter subspace saved by
Phase 2:

    raw_fairness_gradient = -g_fair
    influence_hessian     = -H^{-1} g_fair
    deletion_baseline     = existing influence-deletion update (optional)
    oracle_selected_subspace_direction = theta_oracle - theta_M0 in the Phase-2 subspace (evaluation only)

Two comparison modes are evaluated:

1. common_distance
   Every direction is normalized and applied at the same parameter distance.
   This isolates orientation from magnitude.

2. native_update
   Each direction is applied as a fraction of its native estimated vector.
   This tests the update that each method actually proposes.

The oracle is never used to construct or tune the proposed direction.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import equalized_odds_difference
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer


@dataclass(frozen=True)
class Config:
    # Paths from Phase 1.
    biased_model_path: str = "models/m0_biased"
    oracle_model_path: str = "models/mr_oracle_reference"

    # Phase 2 and deletion-baseline inputs.
    direction_path: str = "data/influence_fairness_direction.pt"
    deletion_update_path: str = "data/estimated_fairness_update.pt"
    fairness_probe_path: str = "data/fairness_probe.csv"

    # Outputs.
    parameter_output: str = "./data/direction_parameter_alignment.csv"
    path_output: str = "data/direction_path_metrics.csv"
    functional_output: str = "./data/direction_functional_alignment.csv"
    metadata_output: str = "./data/direction_comparison_metadata.json"

    seed: int = 42
    max_length: int = 96
    batch_size: int = 32
    probe_per_stratum: int = 24
    threshold: float = 0.5

    # Common-distance path, measured relative to the native IF norm.
    common_distance_fractions: Tuple[float, ...] = (
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
    )

    # Native paths for IF, deletion, and oracle directions.
    native_fractions: Tuple[float, ...] = (
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
    )

    # The raw gradient has a much larger native norm, so use smaller steps.
    raw_gradient_native_fractions: Tuple[float, ...] = (
        0.0,
        0.001,
        0.005,
        0.010,
        0.025,
        0.050,
    )

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
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def vector_dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector lengths differ.")
    total = sum(torch.sum(x.float() * y.float()) for x, y in zip(a, b))
    return float(total.item())


def vector_norm(v: Sequence[Tensor]) -> float:
    total = sum(torch.sum(x.float() ** 2) for x in v)
    return float(torch.sqrt(total).item())


def vector_cosine(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    denominator = vector_norm(a) * vector_norm(b)
    if denominator <= EPS:
        return float("nan")
    return vector_dot(a, b) / denominator


def scale_vector(v: Sequence[Tensor], scale: float) -> List[Tensor]:
    return [tensor.detach().cpu() * float(scale) for tensor in v]


def unit_vector(v: Sequence[Tensor]) -> List[Tensor]:
    norm = vector_norm(v)
    if not math.isfinite(norm) or norm <= EPS:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return scale_vector(v, 1.0 / norm)


def load_direction_payload(path: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 2 direction payload must be a mapping.")

    required = {"parameter_names", "direction", "fairness_gradient"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Phase 2 payload is missing keys: {missing}")
    return payload


def prepare_vector(
    model: BertForSequenceClassification,
    names: Sequence[str],
    tensors: Sequence[Tensor],
    label: str,
) -> List[Tensor]:
    if len(names) != len(tensors) or not names:
        raise ValueError(f"Invalid parameter names or tensors for {label}.")

    parameters = dict(model.named_parameters())
    prepared: List[Tensor] = []

    for name, tensor in zip(names, tensors):
        if name not in parameters:
            raise KeyError(f"{label}: parameter not found in model: {name}")
        if tuple(tensor.shape) != tuple(parameters[name].shape):
            raise ValueError(
                f"{label}: shape mismatch for {name}: "
                f"{tuple(tensor.shape)} vs {tuple(parameters[name].shape)}"
            )

        value = tensor.detach().cpu().to(dtype=parameters[name].dtype)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{label}: non-finite values for parameter {name}")
        prepared.append(value)

    return prepared


def build_oracle_direction(
    biased_model: BertForSequenceClassification,
    oracle_model: BertForSequenceClassification,
    names: Sequence[str],
) -> List[Tensor]:
    biased_parameters = dict(biased_model.named_parameters())
    oracle_parameters = dict(oracle_model.named_parameters())

    direction: List[Tensor] = []
    for name in names:
        if name not in biased_parameters or name not in oracle_parameters:
            raise KeyError(f"Oracle comparison parameter missing: {name}")
        direction.append(
            oracle_parameters[name].detach().cpu()
            - biased_parameters[name].detach().cpu()
        )
    return direction


def load_optional_deletion_direction(
    path: str,
    model: BertForSequenceClassification,
    target_names: Sequence[str],
) -> List[Tensor] | None:
    if not os.path.exists(path):
        print(f"Deletion baseline not found; skipping: {path}")
        return None

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Deletion update payload must be a mapping.")

    names = [str(name) for name in payload.get("parameter_names", [])]
    deltas = payload.get("delta_theta")
    if not isinstance(deltas, Sequence):
        raise ValueError("Deletion payload does not contain delta_theta.")

    by_name = {name: tensor for name, tensor in zip(names, deltas)}
    missing = [name for name in target_names if name not in by_name]
    if missing:
        print(
            "Deletion baseline uses a different parameter subspace; skipping. "
            f"Missing {len(missing)} target tensors."
        )
        return None

    ordered = [by_name[name] for name in target_names]
    return prepare_vector(model, target_names, ordered, "deletion_baseline")


class ProbeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.encodings = tokenizer(
            self.dataframe["essay_text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            self.dataframe["true_quality"].astype(int).to_numpy(),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
            "row_index": torch.tensor(index, dtype=torch.long),
        }


def balanced_probe_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    pieces: List[pd.DataFrame] = []

    for group in ("A", "B"):
        for label in (0, 1):
            cell = dataframe[
                (dataframe["demo"].astype(str) == group)
                & (dataframe["true_quality"].astype(int) == label)
            ]
            if cell.empty:
                raise ValueError(f"Fairness-probe cell is empty: {group}_{label}")

            n = min(config.probe_per_stratum, len(cell))
            positions = rng.choice(len(cell), size=n, replace=False)
            pieces.append(cell.iloc[positions])

    balanced = pd.concat(pieces, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)


def predict(
    model: BertForSequenceClassification,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            output = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            all_labels.append(batch["labels"].numpy())
            all_logits.append(output.logits.detach().cpu().numpy())
            all_indices.append(batch["row_index"].numpy())

    labels = np.concatenate(all_labels).astype(int)
    logits = np.concatenate(all_logits).astype(float)
    order = np.concatenate(all_indices).astype(int)
    inverse_order = np.argsort(order)
    return labels[inverse_order], logits[inverse_order]


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    tensor = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(tensor, dim=1).numpy()[:, 1].astype(float)


def margins_from_logits(logits: np.ndarray) -> np.ndarray:
    return (logits[:, 1] - logits[:, 0]).astype(float)


def safe_rate(mask_numerator: np.ndarray, mask_denominator: np.ndarray) -> float:
    denominator = int(np.sum(mask_denominator))
    if denominator == 0:
        return float("nan")
    return float(np.sum(mask_numerator & mask_denominator) / denominator)


def group_rates(
    labels: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    output: Dict[str, float] = {}

    for group in ("A", "B"):
        group_mask = groups == group
        positive_mask = labels == 1
        negative_mask = labels == 0

        output[f"TPR_{group}"] = safe_rate(
            predictions == 1,
            group_mask & positive_mask,
        )
        output[f"FPR_{group}"] = safe_rate(
            predictions == 1,
            group_mask & negative_mask,
        )
        output[f"pass_rate_{group}"] = (
            float(np.mean(predictions[group_mask]))
            if np.any(group_mask)
            else float("nan")
        )

    output["TPR_gap"] = abs(output["TPR_A"] - output["TPR_B"])
    output["FPR_gap"] = abs(output["FPR_A"] - output["FPR_B"])
    return output


def smooth_eo_surrogate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Exact Phase-2 smooth EO surrogate: sum of squared cell-mean gaps."""
    cell_means: Dict[Tuple[str, int], float] = {}
    for group in ("A", "B"):
        for label in (0, 1):
            mask = (groups == group) & (labels == label)
            if not np.any(mask):
                raise ValueError(f"Missing cell for smooth EO: {group}_{label}")
            cell_means[(group, label)] = float(np.mean(probabilities[mask]))

    y0_gap = cell_means[("A", 0)] - cell_means[("B", 0)]
    y1_gap = cell_means[("A", 1)] - cell_means[("B", 1)]
    return float(y0_gap ** 2 + y1_gap ** 2)


def compute_metrics(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    logits: np.ndarray,
) -> Dict[str, float]:
    probabilities = probabilities_from_logits(logits)
    predictions = (probabilities >= config.threshold).astype(int)
    groups = dataframe["demo"].astype(str).to_numpy()

    rates = group_rates(labels, predictions, groups)

    result: Dict[str, float] = {
        "smooth_eo_surrogate": smooth_eo_surrogate(
            probabilities,
            labels,
            groups,
        ),
        "equalized_odds": float(
            equalized_odds_difference(
                labels,
                predictions,
                sensitive_features=groups,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mean_positive_probability": float(np.mean(probabilities)),
    }
    result.update(rates)
    return result


def prediction_flip_metrics(
    dataframe: pd.DataFrame,
    base_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> Dict[str, int]:
    groups = dataframe["demo"].astype(str).to_numpy()
    changed = candidate_predictions != base_predictions

    output: Dict[str, int] = {
        "changed_predictions": int(np.sum(changed)),
    }

    for group in ("A", "B"):
        group_mask = groups == group
        output[f"changed_predictions_{group}"] = int(
            np.sum(changed & group_mask)
        )
        output[f"{group}_0_to_1"] = int(
            np.sum(
                group_mask
                & (base_predictions == 0)
                & (candidate_predictions == 1)
            )
        )
        output[f"{group}_1_to_0"] = int(
            np.sum(
                group_mask
                & (base_predictions == 1)
                & (candidate_predictions == 0)
            )
        )

    return output


def apply_scaled_native_direction(
    base_model: BertForSequenceClassification,
    names: Sequence[str],
    direction: Sequence[Tensor],
    fraction: float,
) -> BertForSequenceClassification:
    model = copy.deepcopy(base_model)
    parameters = dict(model.named_parameters())

    with torch.no_grad():
        for name, tensor in zip(names, direction):
            parameters[name].add_(
                tensor.to(
                    device=parameters[name].device,
                    dtype=parameters[name].dtype,
                )
                * float(fraction)
            )

    return model


def apply_common_distance_direction(
    base_model: BertForSequenceClassification,
    names: Sequence[str],
    direction: Sequence[Tensor],
    distance: float,
) -> BertForSequenceClassification:
    normalized = unit_vector(direction)
    model = copy.deepcopy(base_model)
    parameters = dict(model.named_parameters())

    with torch.no_grad():
        for name, tensor in zip(names, normalized):
            parameters[name].add_(
                tensor.to(
                    device=parameters[name].device,
                    dtype=parameters[name].dtype,
                )
                * float(distance)
            )

    return model


def array_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    mask = (np.abs(x) > EPS) | (np.abs(y) > EPS)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def append_evaluation_rows(
    *,
    comparison_mode: str,
    method: str,
    path_fraction: float,
    parameter_distance: float,
    candidate_logits: np.ndarray,
    oracle_path_logits: np.ndarray,
    full_oracle_logits: np.ndarray,
    base_logits: np.ndarray,
    base_metrics: Mapping[str, float],
    labels: np.ndarray,
    balanced_df: pd.DataFrame,
    path_rows: List[Dict[str, float | str | int]],
    functional_rows: List[Dict[str, float | str]],
) -> None:
    current_metrics = compute_metrics(balanced_df, labels, candidate_logits)

    base_probabilities = probabilities_from_logits(base_logits)
    candidate_probabilities = probabilities_from_logits(candidate_logits)
    oracle_path_probabilities = probabilities_from_logits(oracle_path_logits)

    base_predictions = (base_probabilities >= config.threshold).astype(int)
    candidate_predictions = (
        candidate_probabilities >= config.threshold
    ).astype(int)

    delta_logits = candidate_logits - base_logits
    oracle_delta_logits = oracle_path_logits - base_logits

    delta_probabilities = candidate_probabilities - base_probabilities
    oracle_delta_probabilities = (
        oracle_path_probabilities - base_probabilities
    )

    base_margins = margins_from_logits(base_logits)
    candidate_margins = margins_from_logits(candidate_logits)
    oracle_path_margins = margins_from_logits(oracle_path_logits)

    delta_margins = candidate_margins - base_margins
    oracle_delta_margins = oracle_path_margins - base_margins

    path_row: Dict[str, float | str | int] = {
        "comparison_mode": comparison_mode,
        "method": method,
        "path_fraction": float(path_fraction),
        "parameter_distance": float(parameter_distance),
        "mean_absolute_logit_change": float(np.mean(np.abs(delta_logits))),
        "max_absolute_logit_change": float(np.max(np.abs(delta_logits))),
        "mean_absolute_probability_change": float(
            np.mean(np.abs(delta_probabilities))
        ),
        "max_absolute_probability_change": float(
            np.max(np.abs(delta_probabilities))
        ),
        "mean_absolute_margin_change": float(np.mean(np.abs(delta_margins))),
        "max_absolute_margin_change": float(np.max(np.abs(delta_margins))),
    }
    path_row.update(
        prediction_flip_metrics(
            balanced_df,
            base_predictions,
            candidate_predictions,
        )
    )

    for key, value in base_metrics.items():
        path_row[f"{key}_before"] = float(value)
    for key, value in current_metrics.items():
        path_row[f"{key}_after"] = float(value)
        path_row[f"{key}_change"] = float(value - base_metrics[key])

    path_rows.append(path_row)

    base_to_full_oracle_distance = float(
        np.linalg.norm(base_logits - full_oracle_logits)
    )
    candidate_to_full_oracle_distance = float(
        np.linalg.norm(candidate_logits - full_oracle_logits)
    )

    functional_rows.append(
        {
            "comparison_mode": comparison_mode,
            "method": method,
            "path_fraction": float(path_fraction),
            "parameter_distance": float(parameter_distance),
            "logit_change_cosine_with_oracle_path": array_cosine(
                delta_logits,
                oracle_delta_logits,
            ),
            "probability_change_cosine_with_oracle_path": array_cosine(
                delta_probabilities,
                oracle_delta_probabilities,
            ),
            "margin_change_cosine_with_oracle": array_cosine(
                delta_margins,
                oracle_delta_margins,
            ),
            "margin_change_correlation_with_oracle": safe_correlation(
                delta_margins,
                oracle_delta_margins,
            ),
            "logit_change_sign_agreement": sign_agreement(
                delta_logits,
                oracle_delta_logits,
            ),
            "probability_change_sign_agreement": sign_agreement(
                delta_probabilities,
                oracle_delta_probabilities,
            ),
            "margin_change_sign_agreement": sign_agreement(
                delta_margins,
                oracle_delta_margins,
            ),
            "functional_distance_to_full_oracle": (
                candidate_to_full_oracle_distance
            ),
            "functional_distance_reduction": float(
                1.0
                - candidate_to_full_oracle_distance
                / max(base_to_full_oracle_distance, EPS)
            ),
        }
    )


def clear_device_cache() -> None:
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    start_time = time.time()

    require_paths(
        [
            config.biased_model_path,
            config.oracle_model_path,
            config.direction_path,
            config.fairness_probe_path,
        ]
    )
    os.makedirs("./data", exist_ok=True)

    print(f"Using device: {DEVICE}")
    print("Loading biased model, oracle reference, and Phase 2 direction...")

    tokenizer = BertTokenizer.from_pretrained(config.biased_model_path)
    biased_model = BertForSequenceClassification.from_pretrained(
        config.biased_model_path
    ).to(DEVICE)
    oracle_model = BertForSequenceClassification.from_pretrained(
        config.oracle_model_path
    ).to(DEVICE)
    biased_model.eval()
    oracle_model.eval()

    payload = load_direction_payload(config.direction_path)
    parameter_names = [str(name) for name in payload["parameter_names"]]
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("Duplicate parameter names in Phase 2 payload.")

    influence_direction = prepare_vector(
        biased_model,
        parameter_names,
        payload["direction"],
        "influence_hessian",
    )
    fairness_gradient = prepare_vector(
        biased_model,
        parameter_names,
        payload["fairness_gradient"],
        "fairness_gradient",
    )
    raw_gradient_direction = [-tensor for tensor in fairness_gradient]
    oracle_direction = build_oracle_direction(
        biased_model,
        oracle_model,
        parameter_names,
    )
    deletion_direction = load_optional_deletion_direction(
        config.deletion_update_path,
        biased_model,
        parameter_names,
    )

    directions: Dict[str, List[Tensor]] = {
        "raw_fairness_gradient": raw_gradient_direction,
        "influence_hessian": influence_direction,
        "oracle_selected_subspace_direction": oracle_direction,
    }
    if deletion_direction is not None:
        directions["deletion_baseline"] = deletion_direction

    parameter_rows: List[Dict[str, float | str | int]] = []
    for method, direction in directions.items():
        parameter_rows.append(
            {
                "method": method,
                "n_parameter_tensors": len(parameter_names),
                "n_parameter_elements": int(
                    sum(tensor.numel() for tensor in direction)
                ),
                "direction_norm": vector_norm(direction),
                "cosine_with_oracle": (
                    1.0
                    if method == "oracle_selected_subspace_direction"
                    else vector_cosine(direction, oracle_direction)
                ),
                "dot_with_fairness_gradient": vector_dot(
                    fairness_gradient,
                    direction,
                ),
            }
        )

    parameter_df = pd.DataFrame(parameter_rows)
    parameter_df.to_csv(config.parameter_output, index=False)

    print("\nParameter-space direction summary:")
    print(parameter_df.to_string(index=False))

    probe_df = pd.read_csv(config.fairness_probe_path)
    required_columns = {"essay_text", "true_quality", "demo", "sample_id"}
    missing_columns = sorted(required_columns - set(probe_df.columns))
    if missing_columns:
        raise ValueError(
            f"Fairness probe is missing columns: {missing_columns}"
        )

    saved_probe_ids = payload.get("probe_sample_ids")
    if not isinstance(saved_probe_ids, Sequence) or not saved_probe_ids:
        raise ValueError(
            "Phase 2 payload does not contain a non-empty probe_sample_ids list."
        )

    probe_by_id = probe_df.set_index("sample_id", drop=False)
    missing_probe_ids = [
        int(sample_id)
        for sample_id in saved_probe_ids
        if int(sample_id) not in probe_by_id.index
    ]
    if missing_probe_ids:
        raise ValueError(
            "Phase 3 cannot reproduce the Phase-2 probe because sample IDs are "
            f"missing from fairness_probe.csv: {missing_probe_ids[:10]}"
        )

    # Reconstruct the exact Phase-2 subset in the exact saved order.
    balanced_df = pd.DataFrame(
        [probe_by_id.loc[int(sample_id)] for sample_id in saved_probe_ids]
    ).reset_index(drop=True)

    if balanced_df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in the saved Phase-2 probe subset.")

    loader = DataLoader(
        ProbeDataset(balanced_df, tokenizer),
        batch_size=config.batch_size,
        shuffle=False,
    )

    labels, base_logits = predict(biased_model, loader)
    _, full_oracle_logits = predict(oracle_model, loader)
    base_metrics = compute_metrics(balanced_df, labels, base_logits)

    phase2_fairness_loss = payload.get("fairness_loss")
    if phase2_fairness_loss is None:
        raise ValueError("Phase 2 payload does not contain fairness_loss.")
    phase2_fairness_loss = float(phase2_fairness_loss)
    phase3_fairness_loss = float(base_metrics["smooth_eo_surrogate"])
    fairness_baseline_difference = abs(
        phase3_fairness_loss - phase2_fairness_loss
    )

    print("\nPhase-2/Phase-3 fairness consistency check:")
    print(f"Phase 2 saved fairness loss : {phase2_fairness_loss:.8f}")
    print(f"Phase 3 reproduced loss     : {phase3_fairness_loss:.8f}")
    print(f"Absolute difference         : {fairness_baseline_difference:.10f}")

    if fairness_baseline_difference > 1e-5:
        raise ValueError(
            "Phase 2 and Phase 3 fairness baselines do not match. "
            "Check probe IDs, tokenization, and surrogate definition."
        )

    full_oracle_metrics = compute_metrics(
        balanced_df, labels, full_oracle_logits
    )

    reference_radius = vector_norm(influence_direction)
    if reference_radius <= EPS:
        raise ValueError("Influence direction has zero norm.")

    print(
        "\nCommon-distance reference radius "
        f"(native IF norm): {reference_radius:.8f}"
    )

    path_rows: List[Dict[str, float | str | int]] = []
    functional_rows: List[Dict[str, float | str]] = []

    # -------------------------------------------------
    # A. Common-distance comparison
    # -------------------------------------------------
    print("\nRunning common-distance comparison...")

    for fraction in config.common_distance_fractions:
        distance = float(fraction * reference_radius)

        oracle_path_model = apply_common_distance_direction(
            biased_model,
            parameter_names,
            oracle_direction,
            distance,
        )
        _, oracle_path_logits = predict(oracle_path_model, loader)
        del oracle_path_model
        clear_device_cache()

        for method, direction in directions.items():
            model = apply_common_distance_direction(
                biased_model,
                parameter_names,
                direction,
                distance,
            )
            _, candidate_logits = predict(model, loader)

            append_evaluation_rows(
                comparison_mode="common_distance",
                method=method,
                path_fraction=float(fraction),
                parameter_distance=distance,
                candidate_logits=candidate_logits,
                oracle_path_logits=oracle_path_logits,
                full_oracle_logits=full_oracle_logits,
                base_logits=base_logits,
                base_metrics=base_metrics,
                labels=labels,
                balanced_df=balanced_df,
                path_rows=path_rows,
                functional_rows=functional_rows,
            )

            del model
            clear_device_cache()

    # -------------------------------------------------
    # B. Native-update comparison
    # -------------------------------------------------
    print("Running native-update comparison...")

    for method, direction in directions.items():
        fractions = (
            config.raw_gradient_native_fractions
            if method == "raw_fairness_gradient"
            else config.native_fractions
        )

        native_norm = vector_norm(direction)

        for fraction in fractions:
            candidate_model = apply_scaled_native_direction(
                biased_model,
                parameter_names,
                direction,
                float(fraction),
            )
            _, candidate_logits = predict(candidate_model, loader)

            # Native oracle reference at the same fraction. This ensures that
            # functional alignment is measured against the oracle's own native
            # path rather than against a common-distance surrogate.
            oracle_path_model = apply_scaled_native_direction(
                biased_model,
                parameter_names,
                oracle_direction,
                float(fraction),
            )
            _, oracle_path_logits = predict(oracle_path_model, loader)

            append_evaluation_rows(
                comparison_mode="native_update",
                method=method,
                path_fraction=float(fraction),
                parameter_distance=float(fraction * native_norm),
                candidate_logits=candidate_logits,
                oracle_path_logits=oracle_path_logits,
                full_oracle_logits=full_oracle_logits,
                base_logits=base_logits,
                base_metrics=base_metrics,
                labels=labels,
                balanced_df=balanced_df,
                path_rows=path_rows,
                functional_rows=functional_rows,
            )

            del candidate_model
            del oracle_path_model
            clear_device_cache()

    # Add the complete oracle model as a distinct reference. It is not a
    # direction restricted to the Phase-2 parameter subspace.
    full_oracle_row: Dict[str, float | str | int] = {
        "comparison_mode": "reference_model",
        "method": "full_oracle_model",
        "path_fraction": 1.0,
        "parameter_distance": float("nan"),
        "mean_absolute_logit_change": float(
            np.mean(np.abs(full_oracle_logits - base_logits))
        ),
        "max_absolute_logit_change": float(
            np.max(np.abs(full_oracle_logits - base_logits))
        ),
        "mean_absolute_probability_change": float(
            np.mean(
                np.abs(
                    probabilities_from_logits(full_oracle_logits)
                    - probabilities_from_logits(base_logits)
                )
            )
        ),
        "max_absolute_probability_change": float(
            np.max(
                np.abs(
                    probabilities_from_logits(full_oracle_logits)
                    - probabilities_from_logits(base_logits)
                )
            )
        ),
        "mean_absolute_margin_change": float(
            np.mean(
                np.abs(
                    margins_from_logits(full_oracle_logits)
                    - margins_from_logits(base_logits)
                )
            )
        ),
        "max_absolute_margin_change": float(
            np.max(
                np.abs(
                    margins_from_logits(full_oracle_logits)
                    - margins_from_logits(base_logits)
                )
            )
        ),
    }
    full_oracle_row.update(
        prediction_flip_metrics(
            balanced_df,
            (probabilities_from_logits(base_logits) >= config.threshold).astype(int),
            (probabilities_from_logits(full_oracle_logits) >= config.threshold).astype(int),
        )
    )
    for key, value in base_metrics.items():
        full_oracle_row[f"{key}_before"] = float(value)
    for key, value in full_oracle_metrics.items():
        full_oracle_row[f"{key}_after"] = float(value)
        full_oracle_row[f"{key}_change"] = float(value - base_metrics[key])
    path_rows.append(full_oracle_row)

    functional_rows.append(
        {
            "comparison_mode": "reference_model",
            "method": "full_oracle_model",
            "path_fraction": 1.0,
            "parameter_distance": float("nan"),
            "logit_change_cosine_with_oracle_path": 1.0,
            "probability_change_cosine_with_oracle_path": 1.0,
            "margin_change_cosine_with_oracle": 1.0,
            "margin_change_correlation_with_oracle": 1.0,
            "logit_change_sign_agreement": 1.0,
            "probability_change_sign_agreement": 1.0,
            "margin_change_sign_agreement": 1.0,
            "functional_distance_to_full_oracle": 0.0,
            "functional_distance_reduction": 1.0,
        }
    )

    path_df = pd.DataFrame(path_rows)
    functional_df = pd.DataFrame(functional_rows)

    path_df.to_csv(config.path_output, index=False)
    functional_df.to_csv(config.functional_output, index=False)

    print("\nCommon-distance summary at fraction 1.0:")
    common_summary_columns = [
        "method",
        "smooth_eo_surrogate_before",
        "smooth_eo_surrogate_after",
        "equalized_odds_before",
        "equalized_odds_after",
        "accuracy_before",
        "accuracy_after",
        "changed_predictions",
        "changed_predictions_A",
        "changed_predictions_B",
    ]
    common_summary = path_df[
        (path_df["comparison_mode"] == "common_distance")
        & np.isclose(path_df["path_fraction"], 1.0)
    ]
    print(common_summary[common_summary_columns].to_string(index=False))

    print("\nNative-update summary at each method's largest tested fraction:")
    native_rows: List[pd.Series] = []
    native_df = path_df[path_df["comparison_mode"] == "native_update"]
    for method in native_df["method"].unique():
        method_df = native_df[native_df["method"] == method]
        native_rows.append(
            method_df.loc[method_df["path_fraction"].idxmax()]
        )
    native_summary = pd.DataFrame(native_rows)
    print(native_summary[common_summary_columns + ["path_fraction", "parameter_distance"]].to_string(index=False))

    print("\nNative functional alignment at each method's largest tested fraction:")
    native_functional_df = functional_df[
        functional_df["comparison_mode"] == "native_update"
    ]
    native_functional_rows: List[pd.Series] = []
    for method in native_functional_df["method"].unique():
        method_df = native_functional_df[
            native_functional_df["method"] == method
        ]
        native_functional_rows.append(
            method_df.loc[method_df["path_fraction"].idxmax()]
        )
    native_functional_summary = pd.DataFrame(native_functional_rows)
    print(native_functional_summary.to_string(index=False))

    metadata = {
        "config": asdict(config),
        "device": str(DEVICE),
        "runtime_seconds": time.time() - start_time,
        "parameter_names": parameter_names,
        "n_parameter_tensors": len(parameter_names),
        "n_parameter_elements": int(
            sum(tensor.numel() for tensor in influence_direction)
        ),
        "comparison_modes": [
            "common_distance",
            "native_update",
            "reference_model",
        ],
        "common_distance_reference_radius_definition": (
            "native norm of -H^{-1} g_fair"
        ),
        "common_distance_reference_radius": reference_radius,
        "oracle_used_for_construction": False,
        "oracle_role": "evaluation reference only",
        "oracle_selected_subspace_method_name": (
            "oracle_selected_subspace_direction"
        ),
        "full_oracle_method_name": "full_oracle_model",
        "phase2_probe_sample_ids": [int(x) for x in saved_probe_ids],
        "phase2_fairness_loss": phase2_fairness_loss,
        "phase3_reproduced_fairness_loss": phase3_fairness_loss,
        "fairness_baseline_absolute_difference": (
            fairness_baseline_difference
        ),
        "deletion_baseline_included": deletion_direction is not None,
        "phase2_predicted_native_fairness_change": payload.get(
            "predicted_first_order_fairness_change"
        ),
        "outputs": {
            "parameter_alignment": config.parameter_output,
            "path_metrics": config.path_output,
            "functional_alignment": config.functional_output,
        },
    }

    with open(config.metadata_output, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSaved outputs:")
    print(f"- {config.parameter_output}")
    print(f"- {config.path_output}")
    print(f"- {config.functional_output}")
    print(f"- {config.metadata_output}")
    print(f"\nPhase 3 completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()