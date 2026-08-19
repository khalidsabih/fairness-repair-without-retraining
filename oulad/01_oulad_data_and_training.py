"""
Phase 1 - OULAD data preparation and baseline training

Corresponds to Section 5.3 of the thesis.

This OULAD experiment performs NO artificial bias injection.
It prepares a real educational prediction task, trains a lightweight MLP,
and creates four disjoint splits:

    train / validation / fairness_probe / untouched_test

The validation split is used for early stopping. The fairness probe remains
untouched during model training and is reserved for the later unlearning and
repair-selection phases.

Default task
------------
Target:
    1 = Pass or Distinction
    0 = Fail or Withdrawn
Protected attribute:
    gender, mapped as M -> A and F -> B
Module:
    BBB, to keep later influence/Hessian computations manageable

Dataset source
--------------
This version uses a locally downloaded OULAD dataset stored in a folder
named "archive" next to this Python file. The folder may contain
studentInfo.csv directly or inside a nested subfolder.

Examples
--------
python 01_oulad_data_and_training.py

Optional override:
python 01_oulad_data_and_training.py --oulad-dir /path/to/oulad
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fairlearn.metrics import equalized_odds_difference
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    output_dir: str = ""
    oulad_dir: str = ""

    experiment_type: str = "natural_disparity"
    dataset_name: str = "OULAD"
    module: str = "BBB"

    protected_attribute: str = "gender"
    group_a_value: str = "M"
    group_b_value: str = "F"

    test_size: float = 0.20
    probe_size: float = 0.15
    validation_size: float = 0.15

    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    dropout: float = 0.20

    batch_size: int = 128
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8

    seed: int = 42
    decision_threshold: float = 0.50


REQUIRED_COLUMNS = {
    "code_module",
    "code_presentation",
    "id_student",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "final_result",
}

BASE_CATEGORICAL_FEATURES = [
    "code_presentation",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

NUMERICAL_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def locate_student_info(root: Path) -> Path:
    direct = root / "studentInfo.csv"
    if direct.exists():
        return direct

    matches = list(root.rglob("studentInfo.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find studentInfo.csv inside {root.resolve()}"
        )
    if len(matches) > 1:
        print(f"Warning: multiple studentInfo.csv files found; using {matches[0]}")
    return matches[0]


def resolve_dataset_dir(config: Config) -> Path:
    script_dir = Path(__file__).resolve().parent
    root = (
        Path(config.oulad_dir).expanduser().resolve()
        if config.oulad_dir
        else script_dir / "archive"
    )

    if not root.exists():
        raise FileNotFoundError(
            "Could not find the local OULAD dataset folder. Expected it at: "
            f"{root}. Place the folder named 'archive' next to this Python file "
            "or pass --oulad-dir PATH."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"OULAD path must point to a directory: {root}")
    return root.resolve()


def categorical_features_for(config: Config) -> List[str]:
    features = list(BASE_CATEGORICAL_FEATURES)

    # Do not feed the protected attribute directly to the predictor. This keeps
    # the experiment focused on disparity encoded through the remaining data.
    features = [f for f in features if f != config.protected_attribute]

    if config.module.upper() == "ALL":
        features.insert(0, "code_module")
    return features


def load_oulad_student_info(
    config: Config,
) -> Tuple[pd.DataFrame, Path, List[str]]:
    dataset_root = resolve_dataset_dir(config)
    csv_path = locate_student_info(dataset_root)
    print(f"Using dataset file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = set(REQUIRED_COLUMNS) | {config.protected_attribute}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"studentInfo.csv is missing required columns: {missing}")

    if config.module.upper() != "ALL":
        available_modules = sorted(df["code_module"].dropna().unique().tolist())
        df = df.loc[df["code_module"] == config.module].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for module {config.module!r}. "
                f"Available modules: {available_modules}"
            )

    target_map = {
        "Pass": 1,
        "Distinction": 1,
        "Fail": 0,
        "Withdrawn": 0,
    }
    df["label"] = df["final_result"].map(target_map)
    df = df.dropna(subset=["label", config.protected_attribute]).copy()
    df["label"] = df["label"].astype(np.int64)

    group_map = {
        config.group_a_value: "A",
        config.group_b_value: "B",
    }
    df = df.loc[df[config.protected_attribute].isin(group_map)].copy()
    df["demo"] = df[config.protected_attribute].map(group_map)

    df["sample_id"] = (
        df["code_module"].astype(str)
        + "_"
        + df["code_presentation"].astype(str)
        + "_"
        + df["id_student"].astype(str)
    )

    # Compatibility fields for later phases. They must not be interpreted as
    # evidence of real corruption in this natural-disparity experiment.
    df["observed_label"] = df["label"]
    df["was_corrupted"] = 0

    categorical_features = categorical_features_for(config)
    keep_columns = [
        "sample_id",
        "id_student",
        "code_module",
        "code_presentation",
        config.protected_attribute,
        "demo",
        "final_result",
        "label",
        "observed_label",
        "was_corrupted",
        *categorical_features,
        *NUMERICAL_FEATURES,
    ]
    keep_columns = list(dict.fromkeys(keep_columns))

    result = df[keep_columns].drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    if result["sample_id"].duplicated().any():
        raise RuntimeError("sample_id is not unique after preprocessing.")

    return result, csv_path, categorical_features


def make_stratification_key(df: pd.DataFrame) -> pd.Series:
    return df["demo"].astype(str) + "_" + df["label"].astype(str)


def validate_strata(df: pd.DataFrame, name: str) -> None:
    expected = pd.MultiIndex.from_product(
        [["A", "B"], [0, 1]], names=["demo", "label"]
    )
    counts = (
        df.groupby(["demo", "label"])
        .size()
        .reindex(expected, fill_value=0)
    )
    if (counts == 0).any():
        raise ValueError(
            f"{name} lacks at least one protected-group/label stratum:\n{counts}"
        )


def split_data(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_holdout = config.test_size + config.probe_size + config.validation_size
    if not 0 < total_holdout < 1:
        raise ValueError("test_size + probe_size + validation_size must be between 0 and 1.")

    train_df, holdout_df = train_test_split(
        df,
        test_size=total_holdout,
        random_state=config.seed,
        stratify=make_stratification_key(df),
    )

    test_fraction_of_holdout = config.test_size / total_holdout
    remaining_df, test_df = train_test_split(
        holdout_df,
        test_size=test_fraction_of_holdout,
        random_state=config.seed + 1,
        stratify=make_stratification_key(holdout_df),
    )

    probe_fraction_of_remaining = config.probe_size / (
        config.probe_size + config.validation_size
    )
    validation_df, probe_df = train_test_split(
        remaining_df,
        test_size=probe_fraction_of_remaining,
        random_state=config.seed + 2,
        stratify=make_stratification_key(remaining_df),
    )

    splits = {
        "train": train_df,
        "validation": validation_df,
        "fairness_probe": probe_df,
        "test": test_df,
    }
    for name, split in splits.items():
        validate_strata(split, name)

    return tuple(split.reset_index(drop=True) for split in splits.values())


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(categorical_features: List[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = make_preprocessor(categorical_features)
    x_train = preprocessor.fit_transform(train_df)
    x_validation = preprocessor.transform(validation_df)
    x_probe = preprocessor.transform(probe_df)
    x_test = preprocessor.transform(test_df)

    return (
        preprocessor,
        np.asarray(x_train, dtype=np.float32),
        np.asarray(x_validation, dtype=np.float32),
        np.asarray(x_probe, dtype=np.float32),
        np.asarray(x_test, dtype=np.float32),
    )


class OULADDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: Iterable[int]):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(list(labels)), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "labels": self.labels[index],
        }


class OULADMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        dropout: float,
    ):
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"].to(device))
            labels = batch["labels"].to(device)
            losses.append(float(criterion(logits, labels).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(
    model: OULADMLP,
    train_dataset: OULADDataset,
    validation_dataset: OULADDataset,
    config: Config,
    device: torch.device,
) -> Tuple[OULADMLP, Dict[str, List[float]]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.to(device)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "validation_loss": [],
    }

    start = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses))
        validation_loss = evaluate_loss(model, validation_loader, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.4f} "
            f"| validation loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    model.to(device)
    print(f"Training runtime: {time.time() - start:.2f} seconds")
    return model, history


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        torch.as_tensor(features, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) != 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def group_rates(
    y_true: np.ndarray,
    predictions: np.ndarray,
    demo: np.ndarray,
) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for group in ["A", "B"]:
        mask = demo == group
        y_group = y_true[mask]
        p_group = predictions[mask]

        tp = int(np.sum((y_group == 1) & (p_group == 1)))
        fn = int(np.sum((y_group == 1) & (p_group == 0)))
        fp = int(np.sum((y_group == 0) & (p_group == 1)))
        tn = int(np.sum((y_group == 0) & (p_group == 0)))

        rates[f"n_{group}"] = int(mask.sum())
        rates[f"TPR_{group}"] = tp / (tp + fn) if tp + fn else float("nan")
        rates[f"FPR_{group}"] = fp / (fp + tn) if fp + tn else float("nan")
        rates[f"FNR_{group}"] = fn / (tp + fn) if tp + fn else float("nan")
        rates[f"selection_rate_{group}"] = (
            float(np.mean(p_group)) if len(p_group) else float("nan")
        )

    rates["TPR_gap"] = abs(rates["TPR_A"] - rates["TPR_B"])
    rates["FPR_gap"] = abs(rates["FPR_A"] - rates["FPR_B"])
    rates["FNR_gap"] = abs(rates["FNR_A"] - rates["FNR_B"])
    rates["selection_rate_gap"] = abs(
        rates["selection_rate_A"] - rates["selection_rate_B"]
    )
    return rates


def audit_model(
    model: OULADMLP,
    features: np.ndarray,
    dataframe: pd.DataFrame,
    config: Config,
    device: torch.device,
    split_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    probabilities = predict_probabilities(
        model, features, config.batch_size, device
    )
    predictions = (probabilities >= config.decision_threshold).astype(np.int64)
    y_true = dataframe["label"].to_numpy(dtype=np.int64)
    demo = dataframe["demo"].to_numpy()

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "precision_positive": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": safe_roc_auc(y_true, probabilities),
        "equalized_odds_difference": float(
            equalized_odds_difference(
                y_true,
                predictions,
                sensitive_features=demo,
            )
        ),
        "threshold": float(config.decision_threshold),
    }
    metrics.update(group_rates(y_true, predictions, demo))

    prediction_df = dataframe[
        [
            "sample_id",
            "id_student",
            "code_module",
            "code_presentation",
            "demo",
            "label",
        ]
    ].copy()
    prediction_df["probability_positive"] = probabilities
    prediction_df["prediction"] = predictions

    print(f"\n{'=' * 68}\n{split_name.upper()} AUDIT\n{'=' * 68}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")

    return metrics, prediction_df


def save_split(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    df.to_csv(output_dir / f"{name}.csv", index=False)


def save_numpy_split(
    features: np.ndarray,
    df: pd.DataFrame,
    name: str,
    output_dir: Path,
) -> None:
    np.savez_compressed(
        output_dir / f"{name}_features.npz",
        features=features,
        labels=df["label"].to_numpy(dtype=np.int64),
        demo=df["demo"].to_numpy(),
        sample_id=df["sample_id"].to_numpy(),
    )


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OULAD Phase 1: natural-disparity baseline training"
    )
    parser.add_argument(
        "--oulad-dir",
        default=None,
        help=(
            "Optional local OULAD directory. By default, the script uses a folder "
            "named 'archive' next to this Python file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. By default, artefacts are saved in "
            "'oulad_phase1' next to this Python file."
        ),
    )
    parser.add_argument("--module", default="BBB", help='Module code or "ALL"')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--protected-attribute", default="gender")
    parser.add_argument("--group-a-value", default="M")
    parser.add_argument("--group-b-value", default="F")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config = Config(
        oulad_dir=args.oulad_dir or str(script_dir / "archive"),
        output_dir=args.output_dir or str(script_dir / "oulad_phase1"),
        module=args.module,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        protected_attribute=args.protected_attribute,
        group_a_value=args.group_a_value,
        group_b_value=args.group_b_value,
    )

    set_seed(config.seed)
    device = get_device()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    df, source_csv, categorical_features = load_oulad_student_info(config)

    print(f"Usable registrations: {len(df)}")
    print("Outcome distribution:")
    print(df["label"].value_counts(normalize=True).sort_index())
    print("Protected-group distribution:")
    print(df["demo"].value_counts(normalize=True).sort_index())
    print("Group/outcome counts:")
    print(df.groupby(["demo", "label"]).size())

    train_df, validation_df, probe_df, test_df = split_data(df, config)
    print(
        "Split sizes | "
        f"train={len(train_df)} | "
        f"validation={len(validation_df)} | "
        f"fairness probe={len(probe_df)} | "
        f"untouched test={len(test_df)}"
    )

    (
        preprocessor,
        x_train,
        x_validation,
        x_probe,
        x_test,
    ) = transform_splits(
        train_df,
        validation_df,
        probe_df,
        test_df,
        categorical_features,
    )

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Encoded feature dimension: {len(feature_names)}")

    model = OULADMLP(
        input_dim=x_train.shape[1],
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
        dropout=config.dropout,
    )
    model, history = train_model(
        model,
        OULADDataset(x_train, train_df["label"]),
        OULADDataset(x_validation, validation_df["label"]),
        config,
        device,
    )

    validation_metrics, validation_predictions = audit_model(
        model,
        x_validation,
        validation_df,
        config,
        device,
        "validation",
    )
    probe_metrics, probe_predictions = audit_model(
        model,
        x_probe,
        probe_df,
        config,
        device,
        "fairness probe",
    )
    test_metrics, test_predictions = audit_model(
        model,
        x_test,
        test_df,
        config,
        device,
        "untouched test",
    )

    split_objects = {
        "train": (train_df, x_train),
        "validation": (validation_df, x_validation),
        "fairness_probe": (probe_df, x_probe),
        "test": (test_df, x_test),
    }
    for name, (frame, features) in split_objects.items():
        save_split(frame, name, output_dir)
        save_numpy_split(features, frame, name, output_dir)

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    write_json(output_dir / "feature_names.json", feature_names)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "OULADMLP",
        "input_dim": int(x_train.shape[1]),
        "hidden_dim_1": config.hidden_dim_1,
        "hidden_dim_2": config.hidden_dim_2,
        "dropout": config.dropout,
        "target_parameter_names_for_unlearning": [
            "encoder.3.weight",
            "encoder.3.bias",
            "classifier.weight",
            "classifier.bias",
        ],
        "config": asdict(config),
    }
    torch.save(checkpoint, output_dir / "m0_oulad.pt")

    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    probe_predictions.to_csv(output_dir / "probe_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    write_json(output_dir / "training_history.json", history)
    write_json(output_dir / "validation_metrics.json", validation_metrics)
    write_json(output_dir / "probe_metrics.json", probe_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)

    experiment_metadata = {
        **asdict(config),
        "source_student_info": str(source_csv),
        "target_definition": {
            "positive": ["Pass", "Distinction"],
            "negative": ["Fail", "Withdrawn"],
        },
        "protected_group_mapping": {
            "A": config.group_a_value,
            "B": config.group_b_value,
        },
        "categorical_features": categorical_features,
        "numerical_features": NUMERICAL_FEATURES,
        "bias_injection_applied": False,
        "oracle_model_available": False,
        "fairness_probe_used_for_training": False,
        "test_used_for_model_selection": False,
    }
    write_json(output_dir / "config.json", experiment_metadata)

    print(f"\nPhase 1 completed. Artefacts saved in: {output_dir}")
    print("No artificial bias was injected.")
    print("The fairness probe was not used for training or early stopping.")
    print("The untouched test split must not be used until the final audit.")


if __name__ == "__main__":
    main()