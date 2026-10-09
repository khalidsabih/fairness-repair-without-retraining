"""
Phase 2: Fairness-oriented influence diagnostics.

This script ranks biased-training examples according to their estimated effect on
an equalized-odds surrogate measured on a separate fairness-probe split.

Corresponds to Section 4.4 of the thesis.

Algorithmic inputs
------------------
- models/m0_biased
- data/train_biased.csv
- data/fairness_probe.csv

Algorithmic signal
------------------
The influence ranking uses only:
- observed training labels;
- clean labels in the separate fairness-probe set;
- demographic attributes in the fairness-probe set.

Synthetic validation only
-------------------------
`was_corrupted` is never used to construct the fairness gradient or calculate
influence scores. It is used only after ranking to quantify whether deliberately
corrupted examples were recovered.

Outputs
-------
- data/influence_scores.csv
- data/topk_influential_samples.csv
- data/influence_detection_metrics.csv
- data/influence_detection_by_group.csv
- data/fairness_probe_gradient.pt
- data/influence_fairness_direction.pt
- data/influence_run_metadata.json
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


# =====================================================
# 1. Configuration
# =====================================================


@dataclass(frozen=True)
class Config:
    model_path: str = "models/m0_biased"
    train_path: str = "data/train_biased.csv"
    fairness_probe_path: str = "data/fairness_probe.csv"

    influence_output: str = "./data/influence_scores.csv"
    topk_output: str = "./data/topk_influential_samples.csv"
    detection_metrics_output: str = "./data/influence_detection_metrics.csv"
    group_metrics_output: str = "./data/influence_detection_by_group.csv"
    probe_gradient_output: str = "./data/fairness_probe_gradient.pt"
    direction_output: str = "./data/influence_fairness_direction.pt"
    metadata_output: str = "./data/influence_run_metadata.json"

    seed: int = 42
    max_length: int = 96

    # Influence is restricted to the classifier and final encoder layer.
    target_parameter_patterns: Tuple[str, ...] = (
        "classifier",
        "bert.encoder.layer.11",
    )

    # LiSSA approximation settings.
    hessian_batch_size: int = 4
    recursion_depth: int = 100
    damping: float = 0.01
    scale: float = 1_000.0
    lissa_repetitions: int = 3

    # Use a balanced fairness probe to prevent a large stratum from dominating.
    probe_per_stratum: int = 24

    # None means score every training example.
    score_limit: int | None = None
    top_k: int = 50

    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


config = Config()
DEVICE = torch.device(config.device)


# =====================================================
# 2. Reproducibility and validation
# =====================================================


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
            "Missing Phase 1 outputs:\n- " + "\n- ".join(missing)
            + "\nRun 01_bias_injection.py first."
        )


def validate_columns(dataframe: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# =====================================================
# 3. Training dataset
# =====================================================


class TrainingDataset(Dataset):
    """Tokenized biased-training examples with stable IDs and local positions."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.encodings = tokenizer(
            self.dataframe["essay_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            self.dataframe["observed_label"].astype(int).to_numpy(),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
            "local_index": torch.tensor(index, dtype=torch.long),
            "sample_id": torch.tensor(
                int(self.dataframe.iloc[index]["sample_id"]), dtype=torch.long
            ),
        }


# =====================================================
# 4. Parameter selection and vector utilities
# =====================================================


def get_target_named_parameters(
    model: BertForSequenceClassification,
) -> List[Tuple[str, torch.nn.Parameter]]:
    selected: List[Tuple[str, torch.nn.Parameter]] = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and any(
            pattern in name for pattern in config.target_parameter_patterns
        ):
            selected.append((name, parameter))

    if not selected:
        raise RuntimeError(
            "No parameters matched target_parameter_patterns: "
            f"{config.target_parameter_patterns}"
        )

    return selected


def vector_dot(vector_a: Sequence[Tensor], vector_b: Sequence[Tensor]) -> Tensor:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vector lengths do not match.")
    return sum(torch.sum(a * b) for a, b in zip(vector_a, vector_b))


def vector_norm(vector: Sequence[Tensor]) -> float:
    squared = sum(torch.sum(tensor.detach().float() ** 2) for tensor in vector)
    return float(torch.sqrt(squared).item())


def detach_to_cpu(vector: Sequence[Tensor]) -> List[Tensor]:
    return [tensor.detach().cpu() for tensor in vector]


# =====================================================
# 5. Fairness-probe construction
# =====================================================


def balanced_probe_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sample equally from A0, A1, B0, and B1 fairness-probe strata."""

    rng = np.random.default_rng(config.seed)
    pieces: List[pd.DataFrame] = []

    for group in ("A", "B"):
        for label in (0, 1):
            stratum = dataframe[
                (dataframe["demo"] == group)
                & (dataframe["true_quality"].astype(int) == label)
            ]

            if stratum.empty:
                raise ValueError(
                    f"Fairness-probe split contains no samples for stratum {group}_{label}."
                )

            n = min(config.probe_per_stratum, len(stratum))
            selected_positions = rng.choice(len(stratum), size=n, replace=False)
            pieces.append(stratum.iloc[selected_positions])

    subset = pd.concat(pieces, ignore_index=True)
    return subset.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)


def compute_fairness_probe_gradient(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    probe_dataframe: pd.DataFrame,
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
) -> Tuple[List[Tensor], float, Dict[str, float]]:
    """
    Differentiate a smooth equalized-odds surrogate.

    For each clean outcome y in {0, 1}, compare the mean predicted positive
    probability between groups A and B:

        L_fair = sum_y (E[p(y_hat=1)|A,y] - E[p(y_hat=1)|B,y])^2

    The clean labels and group attributes come only from the held-out fairness
    probe. No training corruption indicator is used.
    """

    encodings = tokenizer(
        probe_dataframe["essay_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)

    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    positive_probabilities = torch.softmax(outputs.logits, dim=1)[:, 1]

    fairness_terms: List[Tensor] = []
    cell_means: Dict[str, float] = {}

    demo = probe_dataframe["demo"].to_numpy()
    clean_labels = probe_dataframe["true_quality"].astype(int).to_numpy()

    for clean_label in (0, 1):
        group_means: Dict[str, Tensor] = {}

        for group in ("A", "B"):
            mask_np = (demo == group) & (clean_labels == clean_label)
            mask = torch.tensor(mask_np, dtype=torch.bool, device=DEVICE)

            if not bool(mask.any()):
                raise ValueError(f"Empty fairness-probe cell: {group}_{clean_label}")

            mean_probability = positive_probabilities[mask].mean()
            group_means[group] = mean_probability
            cell_means[f"mean_probability_{group}_y{clean_label}"] = float(
                mean_probability.detach().item()
            )

        fairness_terms.append((group_means["A"] - group_means["B"]) ** 2)

    fairness_loss = torch.stack(fairness_terms).sum()
    parameters = [parameter for _, parameter in named_parameters]

    fairness_gradient = grad(
        fairness_loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )

    return (
        [gradient.detach() for gradient in fairness_gradient],
        float(fairness_loss.detach().item()),
        cell_means,
    )


# =====================================================
# 6. Per-example gradients and LiSSA inverse-HVP
# =====================================================


def move_training_batch(batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
        "labels": batch["labels"].to(DEVICE),
    }


def compute_training_gradient(
    model: BertForSequenceClassification,
    batch: Dict[str, Tensor],
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    *,
    create_graph: bool,
) -> List[Tensor]:
    model.zero_grad(set_to_none=True)
    device_batch = move_training_batch(batch)

    outputs = model(**device_batch)
    parameters = [parameter for _, parameter in named_parameters]

    gradients = grad(
        outputs.loss,
        parameters,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )

    if create_graph:
        return list(gradients)
    return [gradient.detach() for gradient in gradients]


def estimate_inverse_hvp_lissa(
    model: BertForSequenceClassification,
    source_vector: Sequence[Tensor],
    train_loader: DataLoader,
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
) -> List[Tensor]:
    """Approximate H^{-1}v using repeated stochastic LiSSA recursions."""

    repetition_estimates: List[List[Tensor]] = []
    parameters = [parameter for _, parameter in named_parameters]

    for repetition in range(config.lissa_repetitions):
        estimate = [tensor.clone().detach() for tensor in source_vector]
        loader_iterator = cycle(train_loader)

        progress = tqdm(
            range(config.recursion_depth),
            desc=f"LiSSA repetition {repetition + 1}/{config.lissa_repetitions}",
        )

        for _ in progress:
            batch = next(loader_iterator)
            device_batch = move_training_batch(batch)

            model.zero_grad(set_to_none=True)
            outputs = model(**device_batch)
            training_gradients = grad(
                outputs.loss,
                parameters,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )

            gradient_estimate_product = vector_dot(training_gradients, estimate)
            hessian_vector_product = grad(
                gradient_estimate_product,
                parameters,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )

            with torch.no_grad():
                estimate = [
                    source
                    + (1.0 - config.damping) * current
                    - hessian_component / config.scale
                    for source, current, hessian_component in zip(
                        source_vector,
                        estimate,
                        hessian_vector_product,
                    )
                ]

        # LiSSA's recursive estimate is scaled by `scale`.
        repetition_estimates.append(
            [tensor.detach() / config.scale for tensor in estimate]
        )

    averaged: List[Tensor] = []
    with torch.no_grad():
        for parameter_index in range(len(source_vector)):
            stacked = torch.stack(
                [estimate[parameter_index] for estimate in repetition_estimates]
            )
            averaged.append(stacked.mean(dim=0))

    return averaged


# =====================================================
# 7. Diagnostic evaluation
# =====================================================


def precision_recall_at_k(
    sorted_corruption_labels: np.ndarray,
    k: int,
) -> Tuple[float, float]:
    effective_k = min(k, len(sorted_corruption_labels))
    if effective_k == 0:
        return float("nan"), float("nan")

    positives_in_top_k = int(sorted_corruption_labels[:effective_k].sum())
    total_positives = int(sorted_corruption_labels.sum())

    precision = positives_in_top_k / effective_k
    recall = positives_in_top_k / total_positives if total_positives else float("nan")
    return precision, recall


def calculate_detection_metrics(ranked: pd.DataFrame) -> pd.DataFrame:
    labels = ranked["was_corrupted"].astype(int).to_numpy()
    scores = ranked["harmful_influence"].astype(float).to_numpy()

    metrics: Dict[str, float | int] = {
        "n_scored": len(ranked),
        "n_corrupted": int(labels.sum()),
        "corruption_prevalence": float(labels.mean()),
    }

    if len(np.unique(labels)) == 2:
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics["average_precision"] = float("nan")
        metrics["roc_auc"] = float("nan")

    for k in sorted({10, 20, config.top_k, 100}):
        if k > len(ranked):
            continue
        precision, recall = precision_recall_at_k(labels, k)
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = recall

    corrupted_ranks = ranked.loc[ranked["was_corrupted"] == 1, "rank"]
    metrics["mean_corrupted_rank"] = (
        float(corrupted_ranks.mean()) if not corrupted_ranks.empty else float("nan")
    )
    metrics["median_corrupted_rank"] = (
        float(corrupted_ranks.median()) if not corrupted_ranks.empty else float("nan")
    )

    return pd.DataFrame([metrics])


def calculate_group_detection_metrics(ranked: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []

    for group in sorted(ranked["demo"].unique()):
        group_df = ranked[ranked["demo"] == group].copy()
        labels = group_df["was_corrupted"].astype(int).to_numpy()
        scores = group_df["harmful_influence"].astype(float).to_numpy()

        row: Dict[str, float | int | str] = {
            "demo": group,
            "n_scored": len(group_df),
            "n_corrupted": int(labels.sum()),
            "corruption_prevalence": float(labels.mean()),
        }

        if len(np.unique(labels)) == 2:
            row["average_precision"] = float(average_precision_score(labels, scores))
            row["roc_auc"] = float(roc_auc_score(labels, scores))
        else:
            row["average_precision"] = float("nan")
            row["roc_auc"] = float("nan")

        local_ranked = group_df.sort_values(
            "harmful_influence", ascending=False
        ).reset_index(drop=True)
        local_labels = local_ranked["was_corrupted"].astype(int).to_numpy()
        local_k = min(config.top_k, len(local_ranked))
        precision, recall = precision_recall_at_k(local_labels, local_k)
        row[f"precision_at_{local_k}"] = precision
        row[f"recall_at_{local_k}"] = recall
        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================
# 8. Main execution
# =====================================================


def main() -> None:
    start_time = time.time()
    os.makedirs("./data", exist_ok=True)

    require_paths(
        [
            config.model_path,
            config.train_path,
            config.fairness_probe_path,
        ]
    )

    print(f"Using device: {DEVICE}")
    print("Loading M0 and Phase 1 datasets...")

    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    model = BertForSequenceClassification.from_pretrained(config.model_path).to(DEVICE)
    model.eval()

    train_df = pd.read_csv(config.train_path)
    fairness_probe_df = pd.read_csv(config.fairness_probe_path)

    shared_columns = [
        "sample_id",
        "essay_text",
        "observed_label",
        "true_quality",
        "demo",
        "bias_eligible",
        "was_corrupted",
        "split",
    ]
    validate_columns(train_df, shared_columns, "train_biased.csv")
    validate_columns(fairness_probe_df, shared_columns, "fairness_probe.csv")

    if train_df["sample_id"].duplicated().any():
        raise ValueError("train_biased.csv contains duplicate sample_id values.")

    training_dataset = TrainingDataset(train_df, tokenizer)
    hessian_loader = DataLoader(
        training_dataset,
        batch_size=config.hessian_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    named_parameters = get_target_named_parameters(model)
    parameter_names = [name for name, _ in named_parameters]
    print(f"Selected {len(parameter_names)} parameter tensors.")
    print(f"Selected parameter elements: {sum(p.numel() for _, p in named_parameters):,}")

    balanced_probe = balanced_probe_subset(fairness_probe_df)
    print(
        "Fairness-probe composition:\n"
        + balanced_probe.groupby(["demo", "true_quality"]).size().to_string()
    )

    fairness_gradient, fairness_loss, probe_cell_means = compute_fairness_probe_gradient(
        model=model,
        tokenizer=tokenizer,
        probe_dataframe=balanced_probe,
        named_parameters=named_parameters,
    )

    print(f"Fairness surrogate loss: {fairness_loss:.8f}")
    print(f"Fairness-gradient norm: {vector_norm(fairness_gradient):.8f}")

    # Save the probe gradient for reproducibility and Phase 3 diagnostics.
    torch.save(
        {
            "parameter_names": parameter_names,
            "gradient": detach_to_cpu(fairness_gradient),
            "fairness_loss": fairness_loss,
            "probe_sample_ids": balanced_probe["sample_id"].astype(int).tolist(),
            "probe_cell_means": probe_cell_means,
            "objective": (
                "sum_y (mean P(class=1 | demo=A, true_quality=y) - "
                "mean P(class=1 | demo=B, true_quality=y))^2"
            ),
            "uses_training_corruption_metadata": False,
        },
        config.probe_gradient_output,
    )

    inverse_fairness_hvp = estimate_inverse_hvp_lissa(
        model=model,
        source_vector=fairness_gradient,
        train_loader=hessian_loader,
        named_parameters=named_parameters,
    )
    inverse_hvp_norm = vector_norm(inverse_fairness_hvp)
    influence_repair_direction = [
        -tensor.detach() for tensor in inverse_fairness_hvp
    ]
    direction_norm = vector_norm(influence_repair_direction)

    # First-order prediction for applying theta_new = theta + d_IF, where
    # d_IF = -H^{-1} g_fair. A negative value predicts a local reduction
    # of the smooth fairness objective.
    with torch.no_grad():
        predicted_fairness_change = float(
            vector_dot(fairness_gradient, influence_repair_direction).item()
        )

    print(f"Inverse-HVP norm: {inverse_hvp_norm:.8f}")
    print(f"Influence repair-direction norm: {direction_norm:.8f}")
    print(
        "Predicted first-order fairness change: "
        f"{predicted_fairness_change:.8f}"
    )

    torch.save(
        {
            "parameter_names": parameter_names,
            "direction": detach_to_cpu(influence_repair_direction),
            "inverse_fairness_hvp": detach_to_cpu(inverse_fairness_hvp),
            "fairness_gradient": detach_to_cpu(fairness_gradient),
            "fairness_loss": fairness_loss,
            "direction_norm": direction_norm,
            "inverse_hvp_norm": inverse_hvp_norm,
            "predicted_first_order_fairness_change": predicted_fairness_change,
            "direction_definition": "-H^{-1} g_fair",
            "objective": (
                "sum_y (mean P(class=1 | demo=A, true_quality=y) - "
                "mean P(class=1 | demo=B, true_quality=y))^2"
            ),
            "target_parameter_patterns": list(config.target_parameter_patterns),
            "probe_sample_ids": balanced_probe["sample_id"].astype(int).tolist(),
            "probe_cell_means": probe_cell_means,
            "lissa_settings": {
                "hessian_batch_size": config.hessian_batch_size,
                "recursion_depth": config.recursion_depth,
                "damping": config.damping,
                "scale": config.scale,
                "repetitions": config.lissa_repetitions,
            },
            "uses_oracle_information": False,
            "uses_training_corruption_metadata": False,
        },
        config.direction_output,
    )

    if config.score_limit is None:
        score_count = len(training_dataset)
    else:
        score_count = min(config.score_limit, len(training_dataset))

    rows: List[Dict[str, object]] = []
    n_train = len(training_dataset)

    for local_index in tqdm(range(score_count), desc="Scoring training examples"):
        item = training_dataset[local_index]
        one_example_batch = {
            key: value.unsqueeze(0)
            for key, value in item.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }

        example_gradient = compute_training_gradient(
            model=model,
            batch=one_example_batch,
            named_parameters=named_parameters,
            create_graph=False,
        )

        with torch.no_grad():
            # Upweighting influence on validation fairness:
            # dL_fair/d epsilon_i = -g_fair^T H^{-1} g_i.
            # By symmetry, this equals -g_i^T H^{-1} g_fair.
            harmful_influence = float(
                (-vector_dot(example_gradient, inverse_fairness_hvp) / n_train).item()
            )

        source_row = train_df.iloc[local_index]
        rows.append(
            {
                "local_train_index": local_index,
                "sample_id": int(source_row["sample_id"]),
                "harmful_influence": harmful_influence,
                "observed_label": int(source_row["observed_label"]),
                "true_quality": int(source_row["true_quality"]),
                "demo": str(source_row["demo"]),
                "bias_eligible": int(source_row["bias_eligible"]),
                "was_corrupted": int(source_row["was_corrupted"]),
                "essay_text": str(source_row["essay_text"]),
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        "harmful_influence", ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    ranked.to_csv(config.influence_output, index=False)

    effective_top_k = min(config.top_k, len(ranked))
    top_k = ranked.head(effective_top_k).copy()
    top_k.to_csv(config.topk_output, index=False)

    detection_metrics = calculate_detection_metrics(ranked)
    detection_metrics.to_csv(config.detection_metrics_output, index=False)

    group_metrics = calculate_group_detection_metrics(ranked)
    group_metrics.to_csv(config.group_metrics_output, index=False)

    runtime_seconds = time.time() - start_time
    metadata = {
        **asdict(config),
        "runtime_seconds": runtime_seconds,
        "n_training_examples": len(training_dataset),
        "n_scored_examples": score_count,
        "effective_top_k": effective_top_k,
        "selected_parameter_names": parameter_names,
        "selected_parameter_count": sum(p.numel() for _, p in named_parameters),
        "fairness_surrogate_loss": fairness_loss,
        "fairness_gradient_norm": vector_norm(fairness_gradient),
        "inverse_hvp_norm": inverse_hvp_norm,
        "influence_repair_direction_norm": direction_norm,
        "predicted_first_order_fairness_change": predicted_fairness_change,
        "direction_definition": "-H^{-1} g_fair",
        "probe_sample_ids": balanced_probe["sample_id"].astype(int).tolist(),
        "probe_cell_means": probe_cell_means,
        "corruption_metadata_used_for_ranking": False,
        "corruption_metadata_used_for_evaluation_only": True,
    }

    with open(config.metadata_output, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTop influential training examples:")
    print(
        top_k[
            [
                "rank",
                "sample_id",
                "harmful_influence",
                "demo",
                "observed_label",
                "true_quality",
                "was_corrupted",
            ]
        ].to_string(index=False)
    )

    print("\nDetection metrics (synthetic validation only):")
    print(detection_metrics.to_string(index=False))

    print("\nSaved outputs:")
    for path in (
        config.influence_output,
        config.topk_output,
        config.detection_metrics_output,
        config.group_metrics_output,
        config.probe_gradient_output,
        config.direction_output,
        config.metadata_output,
    ):
        print(f"- {path}")

    print(f"\nPhase 2 completed in {runtime_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()