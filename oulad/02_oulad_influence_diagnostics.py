"""
Phase 2 - OULAD influence diagnostics and fairness-unlearning direction

This phase loads the natural-disparity OULAD baseline from Phase 1 and:

Corresponds to Section 4.4 of the thesis.

1. constructs a balanced fairness-probe subset;
2. differentiates a smooth Equalized Odds surrogate;
3. estimates H^{-1} g_fair with LiSSA over the training loss;
4. scores training registrations by their estimated influence on probe unfairness;
5. saves both the raw fairness gradient and the Hessian-preconditioned repair
   direction for the next phase.

No artificial corruption labels or oracle model are used. Influence rankings are
therefore diagnostics of model behaviour, not ground-truth corruption recovery.

Expected layout
---------------
project/
├── 01_oulad_data_and_training.py
├── 02_oulad_influence_diagnostics.py
└── oulad_phase1/
    ├── m0_oulad.pt
    ├── train.csv
    ├── train_features.npz
    ├── fairness_probe.csv
    └── fairness_probe_features.npz

Run
---
python 02_oulad_influence_diagnostics.py

Useful lightweight test:
python 02_oulad_influence_diagnostics.py --recursion-depth 50 --score-limit 500
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class Config:
    phase1_dir: str = ""
    output_dir: str = ""
    seed: int = 42
    device: str = "auto"

    # Balanced probe cells: A0, A1, B0, B1.
    probe_per_stratum: int = 100

    # The Hessian is estimated from stochastic mini-batches.
    hessian_batch_size: int = 64
    recursion_depth: int = 200
    damping: float = 0.01
    scale: float = 100.0
    lissa_repetitions: int = 1

    # None scores every training example.
    score_limit: int | None = None
    top_k: int = 100

    # Phase 1 stores these names in the model checkpoint. Keeping only the
    # upper MLP layers makes second-order computations feasible on a normal PC.
    target_parameter_names: Tuple[str, ...] = (
        "encoder.3.weight",
        "encoder.3.bias",
        "classifier.weight",
        "classifier.bias",
    )


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


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: Iterable[int]) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(list(labels)), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "features": self.features[index],
            "labels": self.labels[index],
            "local_index": torch.tensor(index, dtype=torch.long),
        }


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
        raise FileNotFoundError("Missing Phase 1 files:\n- " + "\n- ".join(missing))


def validate_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return (
            np.asarray(data["features"], dtype=np.float32),
            np.asarray(data["labels"], dtype=np.int64),
            np.asarray(data["demo"]).astype(str),
            np.asarray(data["sample_id"]).astype(str),
        )


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[OULADMLP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    model.eval()  # disables dropout during all influence calculations
    return model, checkpoint


def get_target_named_parameters(
    model: nn.Module,
    requested_names: Sequence[str],
) -> List[Tuple[str, nn.Parameter]]:
    parameter_map = dict(model.named_parameters())
    missing = [name for name in requested_names if name not in parameter_map]
    if missing:
        raise ValueError(f"Target parameters not found in model: {missing}")

    selected = [(name, parameter_map[name]) for name in requested_names]
    if not selected:
        raise RuntimeError("No target parameters were selected.")
    return selected


def vector_dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    if len(a) != len(b):
        raise ValueError("Vector lengths do not match.")
    return sum(torch.sum(x * y) for x, y in zip(a, b))


def vector_norm(vector: Sequence[Tensor]) -> float:
    total = sum(torch.sum(x.detach().float() ** 2) for x in vector)
    return float(torch.sqrt(total).item())


def cosine_similarity(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    numerator = float(vector_dot(a, b).detach().item())
    denominator = vector_norm(a) * vector_norm(b)
    return numerator / denominator if denominator > 0 else float("nan")


def to_cpu(vector: Sequence[Tensor]) -> List[Tensor]:
    return [x.detach().cpu() for x in vector]


def balanced_probe_indices(
    labels: np.ndarray,
    demo: np.ndarray,
    per_stratum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []

    for group in ("A", "B"):
        for label in (0, 1):
            candidates = np.flatnonzero((demo == group) & (labels == label))
            if len(candidates) == 0:
                raise ValueError(f"Fairness probe has no examples in cell {group}_{label}.")
            n = min(per_stratum, len(candidates))
            selected.append(rng.choice(candidates, size=n, replace=False))

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def smooth_equalized_odds_loss(
    logits: Tensor,
    labels: np.ndarray,
    demo: np.ndarray,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, float]]:
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    terms: List[Tensor] = []
    cell_means: Dict[str, float] = {}

    for label in (0, 1):
        means: Dict[str, Tensor] = {}
        for group in ("A", "B"):
            mask_np = (labels == label) & (demo == group)
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
            if not bool(mask.any()):
                raise ValueError(f"Empty fairness probe cell {group}_{label}.")
            mean_probability = probabilities[mask].mean()
            means[group] = mean_probability
            cell_means[f"mean_probability_{group}_y{label}"] = float(
                mean_probability.detach().item()
            )
        terms.append((means["A"] - means["B"]) ** 2)

    return torch.stack(terms).sum(), cell_means


def compute_fairness_gradient(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    device: torch.device,
) -> Tuple[List[Tensor], float, Dict[str, float]]:
    model.zero_grad(set_to_none=True)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    logits = model(x)
    fairness_loss, cell_means = smooth_equalized_odds_loss(
        logits, labels, demo, device
    )
    parameters = [parameter for _, parameter in named_parameters]
    gradients = grad(
        fairness_loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return (
        [g.detach() for g in gradients],
        float(fairness_loss.detach().item()),
        cell_means,
    )


def batch_training_gradient(
    model: nn.Module,
    features: Tensor,
    labels: Tensor,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    create_graph: bool,
) -> List[Tensor]:
    model.zero_grad(set_to_none=True)
    logits = model(features)
    loss = F.cross_entropy(logits, labels)
    parameters = [parameter for _, parameter in named_parameters]
    gradients = grad(
        loss,
        parameters,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    return list(gradients) if create_graph else [g.detach() for g in gradients]


def estimate_inverse_hvp_lissa(
    model: nn.Module,
    source_vector: Sequence[Tensor],
    train_loader: DataLoader,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    config: Config,
    device: torch.device,
) -> List[Tensor]:
    parameters = [parameter for _, parameter in named_parameters]
    repetition_results: List[List[Tensor]] = []

    for repetition in range(config.lissa_repetitions):
        estimate = [v.clone().detach() for v in source_vector]
        iterator = cycle(train_loader)
        progress = tqdm(
            range(config.recursion_depth),
            desc=f"LiSSA {repetition + 1}/{config.lissa_repetitions}",
        )

        for _ in progress:
            batch = next(iterator)
            x = batch["features"].to(device)
            y = batch["labels"].to(device)

            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss_gradients = grad(
                loss,
                parameters,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )
            hvp_scalar = vector_dot(loss_gradients, estimate)
            hvp = grad(
                hvp_scalar,
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
                        source_vector, estimate, hvp
                    )
                ]

            if any(not torch.isfinite(x).all() for x in estimate):
                raise FloatingPointError(
                    "LiSSA became non-finite. Increase --scale, increase --damping, "
                    "or use --device cpu."
                )

        repetition_results.append([x.detach() / config.scale for x in estimate])

    averaged: List[Tensor] = []
    with torch.no_grad():
        for parameter_index in range(len(source_vector)):
            stacked = torch.stack(
                [result[parameter_index] for result in repetition_results]
            )
            averaged.append(stacked.mean(dim=0))
    return averaged


def score_examples(
    model: nn.Module,
    train_features: np.ndarray,
    train_df: pd.DataFrame,
    inverse_hvp: Sequence[Tensor],
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    score_count: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n_train = len(train_df)

    for index in tqdm(range(score_count), desc="Scoring training examples"):
        x = torch.as_tensor(
            train_features[index : index + 1], dtype=torch.float32, device=device
        )
        y = torch.as_tensor(
            [int(train_df.iloc[index]["observed_label"])],
            dtype=torch.long,
            device=device,
        )
        example_gradient = batch_training_gradient(
            model, x, y, named_parameters, create_graph=False
        )

        with torch.no_grad():
            # Influence of infinitesimally upweighting example i on probe fairness:
            # d L_fair / d epsilon_i = -g_fair^T H^{-1} g_i.
            # Positive values indicate estimated worsening of the smooth probe
            # fairness objective when the example is upweighted.
            harmful = float((-vector_dot(example_gradient, inverse_hvp) / n_train).item())

        source = train_df.iloc[index]
        rows.append(
            {
                "local_train_index": index,
                "sample_id": str(source["sample_id"]),
                "harmful_influence": harmful,
                "observed_label": int(source["observed_label"]),
                "label": int(source["label"]),
                "demo": str(source["demo"]),
                "id_student": int(source["id_student"]),
                "code_module": str(source["code_module"]),
                "code_presentation": str(source["code_presentation"]),
                "final_result": str(source["final_result"]),
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        "harmful_influence", ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def summarize_rankings(ranked: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    effective_k = min(top_k, len(ranked))
    top = ranked.head(effective_k).copy()

    overall = pd.DataFrame(
        [
            {
                "n_scored": len(ranked),
                "top_k": effective_k,
                "mean_influence": float(ranked["harmful_influence"].mean()),
                "std_influence": float(ranked["harmful_influence"].std(ddof=0)),
                "min_influence": float(ranked["harmful_influence"].min()),
                "max_influence": float(ranked["harmful_influence"].max()),
                "positive_influence_fraction": float(
                    (ranked["harmful_influence"] > 0).mean()
                ),
                "top_k_group_A_fraction": float((top["demo"] == "A").mean()),
                "top_k_positive_label_fraction": float(
                    (top["observed_label"] == 1).mean()
                ),
            }
        ]
    )

    group_rows: List[Dict[str, object]] = []
    for group in sorted(ranked["demo"].unique()):
        full_group = ranked[ranked["demo"] == group]
        top_group = top[top["demo"] == group]
        group_rows.append(
            {
                "demo": group,
                "n_scored": len(full_group),
                "share_of_scored": float(len(full_group) / len(ranked)),
                "mean_influence": float(full_group["harmful_influence"].mean()),
                "median_influence": float(full_group["harmful_influence"].median()),
                "positive_influence_fraction": float(
                    (full_group["harmful_influence"] > 0).mean()
                ),
                "n_in_top_k": len(top_group),
                "share_of_top_k": float(len(top_group) / effective_k),
            }
        )
    return overall, pd.DataFrame(group_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OULAD Phase 2: influence diagnostics and fairness direction"
    )
    parser.add_argument("--phase1-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--probe-per-stratum", type=int, default=100)
    parser.add_argument("--hessian-batch-size", type=int, default=64)
    parser.add_argument("--recursion-depth", type=int, default=200)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--lissa-repetitions", type=int, default=1)
    parser.add_argument("--score-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    phase1_dir = Path(args.phase1_dir).expanduser().resolve() if args.phase1_dir else script_dir / "oulad_phase1"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else script_dir / "oulad_phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        phase1_dir=str(phase1_dir),
        output_dir=str(output_dir),
        seed=args.seed,
        device=args.device,
        probe_per_stratum=args.probe_per_stratum,
        hessian_batch_size=args.hessian_batch_size,
        recursion_depth=args.recursion_depth,
        damping=args.damping,
        scale=args.scale,
        lissa_repetitions=args.lissa_repetitions,
        score_limit=args.score_limit,
        top_k=args.top_k,
    )

    if config.recursion_depth <= 0 or config.hessian_batch_size <= 0:
        raise ValueError("recursion depth and Hessian batch size must be positive.")
    if not 0 <= config.damping < 1:
        raise ValueError("damping must be in [0, 1).")
    if config.scale <= 0:
        raise ValueError("scale must be positive.")

    set_seed(config.seed)
    device = choose_device(config.device)
    print(f"Device: {device}")

    paths = {
        "checkpoint": phase1_dir / "m0_oulad.pt",
        "train_csv": phase1_dir / "train.csv",
        "train_npz": phase1_dir / "train_features.npz",
        "probe_csv": phase1_dir / "fairness_probe.csv",
        "probe_npz": phase1_dir / "fairness_probe_features.npz",
    }
    require_files(list(paths.values()))

    start_time = time.time()
    model, checkpoint = load_model(paths["checkpoint"], device)
    train_df = pd.read_csv(paths["train_csv"])
    probe_df = pd.read_csv(paths["probe_csv"])
    validate_columns(
        train_df,
        ["sample_id", "id_student", "observed_label", "label", "demo", "code_module", "code_presentation", "final_result"],
        "train.csv",
    )
    validate_columns(probe_df, ["sample_id", "label", "demo"], "fairness_probe.csv")

    train_features, train_labels, train_demo, train_ids = load_npz(paths["train_npz"])
    probe_features, probe_labels, probe_demo, probe_ids = load_npz(paths["probe_npz"])

    if len(train_df) != len(train_features) or len(probe_df) != len(probe_features):
        raise ValueError("CSV and NPZ row counts do not match.")
    if not np.array_equal(train_df["label"].to_numpy(dtype=np.int64), train_labels):
        raise ValueError("train.csv labels do not match train_features.npz.")
    if not np.array_equal(probe_df["label"].to_numpy(dtype=np.int64), probe_labels):
        raise ValueError("fairness_probe.csv labels do not match its NPZ file.")

    checkpoint_names = checkpoint.get("target_parameter_names_for_unlearning")
    target_names = tuple(checkpoint_names) if checkpoint_names else config.target_parameter_names
    named_parameters = get_target_named_parameters(model, target_names)
    parameter_names = [name for name, _ in named_parameters]
    selected_elements = sum(parameter.numel() for _, parameter in named_parameters)
    print(f"Selected parameter tensors: {parameter_names}")
    print(f"Selected parameter elements: {selected_elements:,}")

    probe_indices = balanced_probe_indices(
        probe_labels, probe_demo, config.probe_per_stratum, config.seed
    )
    balanced_features = probe_features[probe_indices]
    balanced_labels = probe_labels[probe_indices]
    balanced_demo = probe_demo[probe_indices]
    balanced_ids = probe_ids[probe_indices]

    probe_counts = pd.DataFrame({"demo": balanced_demo, "label": balanced_labels}).groupby(["demo", "label"]).size()
    print("Balanced fairness-probe composition:\n" + probe_counts.to_string())

    fairness_gradient, fairness_loss, cell_means = compute_fairness_gradient(
        model,
        balanced_features,
        balanced_labels,
        balanced_demo,
        named_parameters,
        device,
    )
    print(f"Smooth EO surrogate: {fairness_loss:.8f}")
    print(f"Fairness-gradient norm: {vector_norm(fairness_gradient):.8f}")

    torch.save(
        {
            "parameter_names": parameter_names,
            "gradient": to_cpu(fairness_gradient),
            "fairness_loss": fairness_loss,
            "probe_sample_ids": balanced_ids.tolist(),
            "probe_cell_means": cell_means,
            "objective": "sum_y (mean P(class=1|A,y) - mean P(class=1|B,y))^2",
            "uses_corruption_metadata": False,
        },
        output_dir / "fairness_probe_gradient.pt",
    )

    training_dataset = FeatureDataset(train_features, train_labels)
    generator = torch.Generator().manual_seed(config.seed)
    hessian_loader = DataLoader(
        training_dataset,
        batch_size=config.hessian_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    try:
        inverse_hvp = estimate_inverse_hvp_lissa(
            model,
            fairness_gradient,
            hessian_loader,
            named_parameters,
            config,
            device,
        )
    except RuntimeError as error:
        if device.type == "mps":
            raise RuntimeError(
                f"LiSSA failed on MPS: {error}\nRetry with: "
                "python 02_oulad_influence_diagnostics.py --device cpu"
            ) from error
        raise

    influence_direction = [-x for x in inverse_hvp]
    raw_gradient_direction = [-x for x in fairness_gradient]
    print(f"Inverse-HVP norm: {vector_norm(inverse_hvp):.8f}")
    print(
        "Cosine(raw gradient, Hessian-preconditioned direction): "
        f"{cosine_similarity(raw_gradient_direction, influence_direction):.6f}"
    )

    torch.save(
        {
            "parameter_names": parameter_names,
            "inverse_hvp": to_cpu(inverse_hvp),
            "influence_repair_direction": to_cpu(influence_direction),
            "raw_fairness_repair_direction": to_cpu(raw_gradient_direction),
            "sign_convention": "repair update is theta_new = theta + alpha * direction",
            "fairness_loss": fairness_loss,
            "uses_oracle": False,
        },
        output_dir / "fairness_unlearning_directions.pt",
    )

    score_count = len(train_df) if config.score_limit is None else min(config.score_limit, len(train_df))
    ranked = score_examples(
        model,
        train_features,
        train_df,
        inverse_hvp,
        named_parameters,
        score_count,
        device,
    )
    ranked.to_csv(output_dir / "influence_scores.csv", index=False)
    effective_top_k = min(config.top_k, len(ranked))
    ranked.head(effective_top_k).to_csv(
        output_dir / "topk_influential_samples.csv", index=False
    )

    ranking_summary, group_summary = summarize_rankings(ranked, config.top_k)
    ranking_summary.to_csv(output_dir / "influence_summary.csv", index=False)
    group_summary.to_csv(output_dir / "influence_summary_by_group.csv", index=False)

    runtime = time.time() - start_time
    metadata = {
        **asdict(config),
        "resolved_device": str(device),
        "runtime_seconds": runtime,
        "n_training_examples": len(train_df),
        "n_scored_examples": len(ranked),
        "selected_parameter_names": parameter_names,
        "selected_parameter_elements": selected_elements,
        "fairness_surrogate_loss": fairness_loss,
        "fairness_gradient_norm": vector_norm(fairness_gradient),
        "inverse_hvp_norm": vector_norm(inverse_hvp),
        "raw_vs_influence_direction_cosine": cosine_similarity(
            raw_gradient_direction, influence_direction
        ),
        "probe_cell_means": cell_means,
        "probe_sample_ids": balanced_ids.tolist(),
        "natural_disparity_experiment": True,
        "bias_injection_applied": False,
        "oracle_available": False,
        "interpretation_warning": (
            "Influence ranks estimate contribution to the smooth fairness objective; "
            "they do not identify objectively corrupted records."
        ),
    }
    with (output_dir / "influence_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)

    print("\nTop influential registrations:")
    print(
        ranked.head(min(10, len(ranked)))[
            ["rank", "sample_id", "demo", "observed_label", "harmful_influence"]
        ].to_string(index=False)
    )
    print("\nInfluence summary:")
    print(ranking_summary.to_string(index=False))
    print("\nGroup summary:")
    print(group_summary.to_string(index=False))
    print(f"\nPhase 2 completed in {runtime:.2f} seconds.")
    print(f"Artefacts saved in: {output_dir}")
    print("No corruption-detection AP/AUC is reported because OULAD was not artificially corrupted.")


if __name__ == "__main__":
    main()