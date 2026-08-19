"""
Phase 4: Probe-select and save fairness repair interventions.

This phase uses only the exact Phase-2 fairness-probe subset to select one
step fraction for each repair direction. The untouched test split is never
loaded or used here.

Corresponds to Section 4.5 of the thesis.

Methods
-------
- influence_hessian: proposed oracle-free direction, -H^{-1} g_fair
- raw_fairness_gradient: baseline, -g_fair
- deletion_baseline: influence-guided deletion update, when available
- oracle_selected_subspace_direction: diagnostic reference only

Selection rule
--------------
For each method, select the candidate with the lowest smooth equalized-odds
surrogate subject to an accuracy-loss constraint relative to the biased model.
Ties are resolved by: higher accuracy, smaller parameter distance, then smaller
fraction.

Outputs
-------
- models/m1_influence_repaired/
- models/m1_raw_gradient_baseline/
- models/m1_deletion_baseline/                (if available)
- models/m1_oracle_direction_reference/       (reference only)
- data/repair_selection.csv
- data/repair_selection_metadata.json
- data/selected_repair_updates.pt
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
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
    biased_model_path: str = "models/m0_biased"
    oracle_model_path: str = "models/mr_oracle_reference"
    direction_path: str = "data/influence_fairness_direction.pt"
    deletion_update_path: str = "data/estimated_fairness_update.pt"
    fairness_probe_path: str = "data/fairness_probe.csv"

    influence_output_dir: str = "models/m1_influence_repaired"
    raw_gradient_output_dir: str = "models/m1_raw_gradient_baseline"
    deletion_output_dir: str = "models/m1_deletion_baseline"
    oracle_direction_output_dir: str = "models/m1_oracle_direction_reference"

    selection_output: str = "./data/repair_selection.csv"
    metadata_output: str = "./data/repair_selection_metadata.json"
    update_output: str = "./data/selected_repair_updates.pt"

    seed: int = 42
    max_length: int = 96
    batch_size: int = 32
    threshold: float = 0.5
    accuracy_loss_tolerance: float = 0.02
    fairness_consistency_tolerance: float = 1e-5
    overwrite_existing_outputs: bool = True

    influence_fractions: Tuple[float, ...] = (
        0.0, 0.10, 0.25, 0.50, 0.75, 1.00
    )
    raw_gradient_fractions: Tuple[float, ...] = (
        0.0, 0.001, 0.005, 0.010, 0.025, 0.050
    )
    deletion_fractions: Tuple[float, ...] = (
        0.0, 0.10, 0.25, 0.50, 0.75, 1.00
    )
    oracle_direction_fractions: Tuple[float, ...] = (
        0.0, 0.10, 0.25, 0.50, 0.75, 1.00
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


def vector_norm(vector: Sequence[Tensor]) -> float:
    total = sum(torch.sum(tensor.detach().float() ** 2) for tensor in vector)
    return float(torch.sqrt(total).item())


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
    result: List[Tensor] = []
    for name, tensor in zip(names, tensors):
        if name not in parameters:
            raise KeyError(f"{label}: parameter not found: {name}")
        if tuple(tensor.shape) != tuple(parameters[name].shape):
            raise ValueError(
                f"{label}: shape mismatch for {name}: "
                f"{tuple(tensor.shape)} vs {tuple(parameters[name].shape)}"
            )
        value = tensor.detach().cpu().to(dtype=parameters[name].dtype)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{label}: non-finite values for {name}")
        result.append(value)
    return result


def build_oracle_direction(
    biased_model: BertForSequenceClassification,
    oracle_model: BertForSequenceClassification,
    names: Sequence[str],
) -> List[Tensor]:
    biased = dict(biased_model.named_parameters())
    oracle = dict(oracle_model.named_parameters())
    return [
        oracle[name].detach().cpu() - biased[name].detach().cpu()
        for name in names
    ]


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
    return prepare_vector(
        model,
        target_names,
        [by_name[name] for name in target_names],
        "deletion_baseline",
    )


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


def predict(
    model: BertForSequenceClassification,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_list: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    indices_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            labels_list.append(batch["labels"].numpy())
            logits_list.append(output.logits.detach().cpu().numpy())
            indices_list.append(batch["row_index"].numpy())
    labels = np.concatenate(labels_list).astype(int)
    logits = np.concatenate(logits_list).astype(float)
    order = np.concatenate(indices_list).astype(int)
    inverse = np.argsort(order)
    return labels[inverse], logits[inverse]


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()[:, 1]


def smooth_eo_surrogate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    means: Dict[Tuple[str, int], float] = {}
    for group in ("A", "B"):
        for label in (0, 1):
            mask = (groups == group) & (labels == label)
            if not np.any(mask):
                raise ValueError(f"Missing fairness cell: {group}_{label}")
            means[(group, label)] = float(np.mean(probabilities[mask]))
    gap_0 = means[("A", 0)] - means[("B", 0)]
    gap_1 = means[("A", 1)] - means[("B", 1)]
    return float(gap_0 ** 2 + gap_1 ** 2)


def safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> float:
    n = int(np.sum(denominator))
    return float(np.sum(numerator & denominator) / n) if n else float("nan")


def compute_metrics(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    logits: np.ndarray,
) -> Dict[str, float]:
    probabilities = probabilities_from_logits(logits)
    predictions = (probabilities >= config.threshold).astype(int)
    groups = dataframe["demo"].astype(str).to_numpy()

    result: Dict[str, float] = {
        "smooth_eo_surrogate": smooth_eo_surrogate(probabilities, labels, groups),
        "equalized_odds": float(
            equalized_odds_difference(
                labels, predictions, sensitive_features=groups
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mean_positive_probability": float(np.mean(probabilities)),
    }

    for group in ("A", "B"):
        group_mask = groups == group
        positive = labels == 1
        negative = labels == 0
        result[f"TPR_{group}"] = safe_rate(predictions == 1, group_mask & positive)
        result[f"FPR_{group}"] = safe_rate(predictions == 1, group_mask & negative)
        result[f"pass_rate_{group}"] = float(np.mean(predictions[group_mask]))

    result["TPR_gap"] = abs(result["TPR_A"] - result["TPR_B"])
    result["FPR_gap"] = abs(result["FPR_A"] - result["FPR_B"])
    return result


def apply_direction(
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
    model.eval()
    return model


def remove_existing_output(path: str) -> None:
    if not os.path.exists(path):
        return
    if not config.overwrite_existing_outputs:
        raise FileExistsError(
            f"Output already exists: {path}. Set overwrite_existing_outputs=True "
            "or remove it manually."
        )
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def save_model_and_tokenizer(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    output_dir: str,
) -> None:
    remove_existing_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def select_candidate(
    rows: pd.DataFrame,
    baseline_accuracy: float,
) -> Tuple[pd.Series, str]:
    minimum_accuracy = baseline_accuracy - config.accuracy_loss_tolerance
    feasible = rows[rows["accuracy"] >= minimum_accuracy - EPS].copy()
    rule = (
        "minimum smooth_eo_surrogate subject to accuracy >= "
        f"{minimum_accuracy:.8f}"
    )

    if feasible.empty:
        feasible = rows.copy()
        rule = (
            "fallback: no candidate satisfied utility constraint; selected maximum "
            "accuracy, then minimum smooth_eo_surrogate"
        )
        feasible = feasible.sort_values(
            ["accuracy", "smooth_eo_surrogate", "parameter_distance", "fraction"],
            ascending=[False, True, True, True],
            kind="stable",
        )
    else:
        feasible = feasible.sort_values(
            ["smooth_eo_surrogate", "accuracy", "parameter_distance", "fraction"],
            ascending=[True, False, True, True],
            kind="stable",
        )

    return feasible.iloc[0], rule


def dataframe_hash(dataframe: pd.DataFrame) -> str:
    canonical = dataframe.sort_values("sample_id").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def get_saved_phase2_fairness(payload: Mapping[str, object]) -> float | None:
    candidate_keys = (
        "fairness_loss",
        "fairness_surrogate_loss",
        "baseline_fairness_loss",
        "phase2_fairness_loss",
    )
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Tensor) and value.numel() == 1:
            return float(value.item())
    return None


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
    os.makedirs("./models", exist_ok=True)

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
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("Duplicate parameter names in Phase 2 payload.")

    influence_direction = prepare_vector(
        biased_model, parameter_names, payload["direction"], "influence_hessian"
    )
    fairness_gradient = prepare_vector(
        biased_model,
        parameter_names,
        payload["fairness_gradient"],
        "fairness_gradient",
    )
    raw_direction = [-tensor for tensor in fairness_gradient]
    oracle_direction = build_oracle_direction(
        biased_model, oracle_model, parameter_names
    )
    deletion_direction = load_optional_deletion_direction(
        config.deletion_update_path, biased_model, parameter_names
    )

    directions: Dict[str, List[Tensor]] = {
        "influence_hessian": influence_direction,
        "raw_fairness_gradient": raw_direction,
        "oracle_selected_subspace_direction": oracle_direction,
    }
    fraction_grids: Dict[str, Tuple[float, ...]] = {
        "influence_hessian": config.influence_fractions,
        "raw_fairness_gradient": config.raw_gradient_fractions,
        "oracle_selected_subspace_direction": config.oracle_direction_fractions,
    }
    output_dirs: Dict[str, str] = {
        "influence_hessian": config.influence_output_dir,
        "raw_fairness_gradient": config.raw_gradient_output_dir,
        "oracle_selected_subspace_direction": config.oracle_direction_output_dir,
    }
    is_oracle_free: Dict[str, bool] = {
        "influence_hessian": True,
        "raw_fairness_gradient": True,
        "oracle_selected_subspace_direction": False,
    }

    if deletion_direction is not None:
        directions["deletion_baseline"] = deletion_direction
        fraction_grids["deletion_baseline"] = config.deletion_fractions
        output_dirs["deletion_baseline"] = config.deletion_output_dir
        is_oracle_free["deletion_baseline"] = True

    probe_df = pd.read_csv(config.fairness_probe_path)
    required_columns = {"sample_id", "essay_text", "true_quality", "demo"}
    missing = sorted(required_columns - set(probe_df.columns))
    if missing:
        raise ValueError(f"Fairness probe is missing columns: {missing}")

    saved_probe_ids = payload.get("probe_sample_ids")
    if not isinstance(saved_probe_ids, Sequence) or len(saved_probe_ids) == 0:
        raise ValueError(
            "Phase 2 payload does not contain a non-empty probe_sample_ids list."
        )
    saved_probe_ids = [int(value) for value in saved_probe_ids]
    if len(saved_probe_ids) != len(set(saved_probe_ids)):
        raise ValueError("Phase 2 probe_sample_ids contains duplicates.")

    probe_by_id = probe_df.set_index("sample_id", drop=False)
    absent_ids = [sample_id for sample_id in saved_probe_ids if sample_id not in probe_by_id.index]
    if absent_ids:
        raise ValueError(
            f"{len(absent_ids)} Phase 2 probe IDs are missing from fairness_probe.csv."
        )
    selected_probe = probe_by_id.loc[saved_probe_ids].reset_index(drop=True)

    observed_cells = set(
        zip(
            selected_probe["demo"].astype(str),
            selected_probe["true_quality"].astype(int),
        )
    )
    required_cells = {("A", 0), ("A", 1), ("B", 0), ("B", 1)}
    if observed_cells != required_cells:
        raise ValueError(
            f"Exact Phase 2 probe does not contain all required cells: {observed_cells}"
        )

    loader = DataLoader(
        ProbeDataset(selected_probe, tokenizer),
        batch_size=config.batch_size,
        shuffle=False,
    )
    labels, base_logits = predict(biased_model, loader)
    base_metrics = compute_metrics(selected_probe, labels, base_logits)

    phase2_fairness = get_saved_phase2_fairness(payload)
    if phase2_fairness is not None:
        difference = abs(base_metrics["smooth_eo_surrogate"] - phase2_fairness)
        print("\nPhase-2/Phase-4 fairness consistency check:")
        print(f"Phase 2 saved fairness loss : {phase2_fairness:.8f}")
        print(
            "Phase 4 reproduced loss     : "
            f"{base_metrics['smooth_eo_surrogate']:.8f}"
        )
        print(f"Absolute difference         : {difference:.10f}")
        if difference > config.fairness_consistency_tolerance:
            raise ValueError(
                "Phase 2 and Phase 4 fairness baselines do not match. "
                "Do not select a repair until the mismatch is resolved."
            )
    else:
        difference = None
        print(
            "Warning: no scalar Phase-2 fairness loss was found in the payload; "
            "probe IDs are still reproduced exactly."
        )

    all_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    selected_updates: Dict[str, Dict[str, object]] = {}

    print("\nEvaluating candidate fractions on the exact Phase 2 probe...")
    for method, direction in directions.items():
        direction_norm = vector_norm(direction)
        method_rows: List[Dict[str, object]] = []

        for fraction in fraction_grids[method]:
            candidate_model = apply_direction(
                biased_model, parameter_names, direction, fraction
            )
            candidate_labels, candidate_logits = predict(candidate_model, loader)
            if not np.array_equal(candidate_labels, labels):
                raise RuntimeError("Probe labels changed across evaluations.")
            metrics = compute_metrics(selected_probe, candidate_labels, candidate_logits)
            probabilities = probabilities_from_logits(candidate_logits)
            predictions = (probabilities >= config.threshold).astype(int)
            base_predictions = (
                probabilities_from_logits(base_logits) >= config.threshold
            ).astype(int)

            row: Dict[str, object] = {
                "method": method,
                "fraction": float(fraction),
                "parameter_distance": float(direction_norm * fraction),
                "direction_norm": float(direction_norm),
                "is_oracle_free": bool(is_oracle_free[method]),
                "changed_predictions": int(np.sum(predictions != base_predictions)),
                "meets_accuracy_constraint": bool(
                    metrics["accuracy"]
                    >= base_metrics["accuracy"] - config.accuracy_loss_tolerance - EPS
                ),
            }
            for key, value in metrics.items():
                row[key] = float(value)
                row[f"{key}_change"] = float(value - base_metrics[key])
            method_rows.append(row)
            all_rows.append(row)

            del candidate_model
            clear_device_cache()

        method_df = pd.DataFrame(method_rows)
        chosen, selection_rule = select_candidate(
            method_df, base_metrics["accuracy"]
        )
        chosen_fraction = float(chosen["fraction"])
        selected_model = apply_direction(
            biased_model, parameter_names, direction, chosen_fraction
        )
        save_model_and_tokenizer(
            selected_model, tokenizer, output_dirs[method]
        )

        selected_record = chosen.to_dict()
        selected_record.update(
            {
                "selected": True,
                "selection_rule": selection_rule,
                "output_model_dir": output_dirs[method],
                "selection_split": "fairness_probe_exact_phase2_subset",
                "test_set_used_for_selection": False,
            }
        )
        selected_rows.append(selected_record)
        selected_updates[method] = {
            "fraction": chosen_fraction,
            "parameter_names": parameter_names,
            "direction": [tensor.detach().cpu() for tensor in direction],
            "scaled_update": [
                tensor.detach().cpu() * chosen_fraction for tensor in direction
            ],
            "direction_norm": direction_norm,
            "parameter_distance": float(direction_norm * chosen_fraction),
            "output_model_dir": output_dirs[method],
            "is_oracle_free": is_oracle_free[method],
        }

        print(
            f"{method:38s} fraction={chosen_fraction:>7g}  "
            f"smooth_EO={float(chosen['smooth_eo_surrogate']):.8f}  "
            f"accuracy={float(chosen['accuracy']):.4f}  "
            f"saved={output_dirs[method]}"
        )

        del selected_model
        clear_device_cache()

    all_df = pd.DataFrame(all_rows)
    selected_df = pd.DataFrame(selected_rows)
    all_df["selected"] = False
    for _, selected in selected_df.iterrows():
        mask = (
            (all_df["method"] == selected["method"])
            & np.isclose(all_df["fraction"], float(selected["fraction"]))
        )
        all_df.loc[mask, "selected"] = True

    all_df = all_df.sort_values(
        ["method", "fraction"], kind="stable"
    ).reset_index(drop=True)
    all_df.to_csv(config.selection_output, index=False)

    torch.save(
        {
            "selection_split": "fairness_probe_exact_phase2_subset",
            "test_set_used_for_selection": False,
            "parameter_names": parameter_names,
            "selected_updates": selected_updates,
            "phase2_probe_sample_ids": saved_probe_ids,
            "base_metrics": base_metrics,
        },
        config.update_output,
    )

    primary = next(
        row for row in selected_rows if row["method"] == "influence_hessian"
    )
    metadata = {
        "phase": 4,
        "purpose": "probe_select_and_save_repairs",
        "config": asdict(config),
        "device": str(DEVICE),
        "selection_split": "fairness_probe_exact_phase2_subset",
        "test_set_used_for_selection": False,
        "selection_rule": (
            "Minimize smooth_eo_surrogate subject to accuracy loss <= "
            f"{config.accuracy_loss_tolerance:.4f}; ties: higher accuracy, "
            "smaller parameter distance, smaller fraction."
        ),
        "primary_method": "influence_hessian",
        "primary_selected_fraction": float(primary["fraction"]),
        "primary_output_model_dir": primary["output_model_dir"],
        "base_probe_metrics": base_metrics,
        "phase2_saved_fairness_loss": phase2_fairness,
        "phase2_phase4_fairness_difference": difference,
        "phase2_probe_sample_ids": saved_probe_ids,
        "probe_size": int(len(selected_probe)),
        "probe_sha256": dataframe_hash(selected_probe),
        "selected_models": selected_rows,
        "full_oracle_model_path": config.oracle_model_path,
        "full_oracle_used_for_selection": False,
        "elapsed_seconds": float(time.time() - start_time),
    }
    with open(config.metadata_output, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print("\nSelected repairs:")
    print(
        selected_df[
            [
                "method",
                "fraction",
                "parameter_distance",
                "smooth_eo_surrogate",
                "accuracy",
                "changed_predictions",
                "output_model_dir",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    print(f"- {config.selection_output}")
    print(f"- {config.metadata_output}")
    print(f"- {config.update_output}")
    for row in selected_rows:
        print(f"- {row['output_model_dir']}")
    print(f"\nPhase 4 completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()