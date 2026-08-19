"""
Phase 1 - xAPI-Edu-Data preparation and baseline training

Corresponds to Section 5.3 of the thesis.

This real-data experiment performs NO artificial bias injection.
It trains a lightweight MLP for binary student-performance prediction and
creates four disjoint splits:

    train / validation / fairness_probe / untouched_test

Default task
------------
Target:
    1 = medium or high performance (Class in {M, H})
    0 = low performance (Class == L)
Protected attribute:
    gender, mapped as M -> A and F -> B

Dataset layout
--------------
Place the downloaded file next to this script as:

    archive/xAPI-Edu-Data.csv

The validation split is used for early stopping. The fairness probe is not
used during training and is reserved for later repair estimation/selection.
The untouched test split must not be used until the final audit.
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
    dataset_dir: str = ""
    dataset_name: str = "xAPI-Edu-Data"
    experiment_type: str = "natural_disparity"

    protected_attribute: str = "gender"
    group_a_value: str = "M"
    group_b_value: str = "F"

    positive_classes: Tuple[str, ...] = ("M", "H")
    negative_classes: Tuple[str, ...] = ("L",)

    test_size: float = 0.20
    probe_size: float = 0.15
    validation_size: float = 0.15

    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    dropout: float = 0.20

    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12

    seed: int = 42
    decision_threshold: float = 0.50


REQUIRED_COLUMNS = {
    "gender",
    "NationalITy",
    "PlaceofBirth",
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
    "Class",
}

CATEGORICAL_FEATURES_BASE = [
    "NationalITy",
    "PlaceofBirth",
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
]

NUMERICAL_FEATURES = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
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


def resolve_dataset_root(config: Config) -> Path:
    script_dir = Path(__file__).resolve().parent
    root = (
        Path(config.dataset_dir).expanduser().resolve()
        if config.dataset_dir
        else script_dir / "archive"
    )
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. Place 'archive' next to this script "
            "or pass --dataset-dir PATH."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")
    return root


def locate_xapi_csv(root: Path) -> Path:
    preferred = root / "xAPI-Edu-Data.csv"
    if preferred.exists():
        return preferred

    exact_matches = list(root.rglob("xAPI-Edu-Data.csv"))
    if exact_matches:
        return exact_matches[0]

    csv_matches = list(root.rglob("*.csv"))
    if len(csv_matches) == 1:
        print(
            "Warning: xAPI-Edu-Data.csv was not found by name; "
            f"using the only CSV present: {csv_matches[0]}"
        )
        return csv_matches[0]
    if not csv_matches:
        raise FileNotFoundError(f"No CSV file found inside {root}")
    raise FileNotFoundError(
        "Could not uniquely identify xAPI-Edu-Data.csv. CSV candidates:\n- "
        + "\n- ".join(str(p) for p in csv_matches)
    )


def categorical_features_for(config: Config) -> List[str]:
    # The protected attribute is deliberately excluded from model inputs.
    return [
        name
        for name in CATEGORICAL_FEATURES_BASE
        if name != config.protected_attribute
    ]


def load_xapi_data(
    config: Config,
) -> Tuple[pd.DataFrame, Path, List[str]]:
    root = resolve_dataset_root(config)
    csv_path = locate_xapi_csv(root)
    print(f"Using dataset file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["Class"] = df["Class"].astype(str).str.strip().str.upper()
    allowed_classes = set(config.positive_classes) | set(config.negative_classes)
    df = df.loc[df["Class"].isin(allowed_classes)].copy()

    df["label"] = df["Class"].isin(config.positive_classes).astype(np.int64)

    group_map = {
        str(config.group_a_value): "A",
        str(config.group_b_value): "B",
    }
    df[config.protected_attribute] = (
        df[config.protected_attribute].astype(str).str.strip()
    )
    df = df.loc[df[config.protected_attribute].isin(group_map)].copy()
    df["demo"] = df[config.protected_attribute].map(group_map)

    # Stable row identifier because the public file does not include student IDs.
    # This identifies records, not individual learners.
    df = df.reset_index(drop=False).rename(columns={"index": "source_row"})
    df["sample_id"] = "xapi_" + df["source_row"].astype(str)

    # Compatibility fields for later phases. No corruption is implied.
    df["observed_label"] = df["label"]
    df["was_corrupted"] = 0

    categorical_features = categorical_features_for(config)
    keep_columns = [
        "sample_id",
        "source_row",
        config.protected_attribute,
        "demo",
        "Class",
        "label",
        "observed_label",
        "was_corrupted",
        *categorical_features,
        *NUMERICAL_FEATURES,
    ]
    keep_columns = list(dict.fromkeys(keep_columns))

    result = df[keep_columns].copy().reset_index(drop=True)
    if result["sample_id"].duplicated().any():
        raise RuntimeError("sample_id is not unique after preprocessing.")

    return result, csv_path, categorical_features


def make_stratification_key(df: pd.DataFrame) -> pd.Series:
    return df["demo"].astype(str) + "_" + df["label"].astype(str)


def validate_strata(df: pd.DataFrame, name: str) -> None:
    expected = pd.MultiIndex.from_product(
        [["A", "B"], [0, 1]], names=["demo", "label"]
    )
    counts = df.groupby(["demo", "label"]).size().reindex(expected, fill_value=0)
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
        raise ValueError(
            "test_size + probe_size + validation_size must be between 0 and 1."
        )

    stratum_counts = make_stratification_key(df).value_counts()
    if (stratum_counts < 8).any():
        raise ValueError(
            "At least one gender/label stratum is too small for a four-way split:\n"
            f"{stratum_counts}"
        )

    train_df, holdout_df = train_test_split(
        df,
        test_size=total_holdout,
        random_state=config.seed,
        stratify=make_stratification_key(df),
    )

    test_fraction = config.test_size / total_holdout
    remaining_df, test_df = train_test_split(
        holdout_df,
        test_size=test_fraction,
        random_state=config.seed + 1,
        stratify=make_stratification_key(holdout_df),
    )

    probe_fraction = config.probe_size / (config.probe_size + config.validation_size)
    validation_df, probe_df = train_test_split(
        remaining_df,
        test_size=probe_fraction,
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


class XAPIDataset(Dataset):
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


class XAPIMLP(nn.Module):
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
    model: XAPIMLP,
    train_dataset: XAPIDataset,
    validation_dataset: XAPIDataset,
    config: Config,
    device: torch.device,
) -> Tuple[XAPIMLP, Dict[str, List[float]]]:
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
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            outputs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outputs)


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
    model: XAPIMLP,
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
        "average_precision": float(average_precision_score(y_true, probabilities)),
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
        ["sample_id", "source_row", config.protected_attribute, "demo", "Class", "label"]
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
        description="xAPI Phase 1: natural-disparity baseline training"
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Optional dataset directory. By default, use 'archive' next to this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. By default, use 'xapi_phase1' next to this script."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--protected-attribute", default="gender")
    parser.add_argument("--group-a-value", default="M")
    parser.add_argument("--group-b-value", default="F")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config = Config(
        dataset_dir=args.dataset_dir or str(script_dir / "archive"),
        output_dir=args.output_dir or str(script_dir / "xapi_phase1"),
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
    df, source_csv, categorical_features = load_xapi_data(config)

    print(f"Usable records: {len(df)}")
    print("Original class distribution:")
    print(df["Class"].value_counts(normalize=True).sort_index())
    print("Binary outcome distribution:")
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

    model = XAPIMLP(
        input_dim=x_train.shape[1],
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
        dropout=config.dropout,
    )
    model, history = train_model(
        model,
        XAPIDataset(x_train, train_df["label"]),
        XAPIDataset(x_validation, validation_df["label"]),
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
        "model_type": "XAPIMLP",
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
    torch.save(checkpoint, output_dir / "m0_xapi.pt")

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
        "source_csv": str(source_csv),
        "target_definition": {
            "positive": list(config.positive_classes),
            "negative": list(config.negative_classes),
            "interpretation": "medium/high performance versus low performance",
        },
        "protected_group_mapping": {
            "A": config.group_a_value,
            "B": config.group_b_value,
        },
        "categorical_features": categorical_features,
        "numerical_features": NUMERICAL_FEATURES,
        "protected_attribute_excluded_from_model_input": True,
        "bias_injection_applied": False,
        "oracle_model_available": False,
        "fairness_probe_used_for_training": False,
        "test_used_for_model_selection": False,
        "important_scope_note": (
            "The public xAPI file has no stable student identifier. Rows are treated "
            "as educational records, and random splitting therefore cannot guarantee "
            "student-level separation."
        ),
    }
    write_json(output_dir / "config.json", experiment_metadata)

    print(f"\nPhase 1 completed. Artefacts saved in: {output_dir}")
    print("No artificial bias was injected.")
    print("Gender was excluded from the model input and used only for fairness auditing.")
    print("The fairness probe was not used for training or early stopping.")
    print("The untouched test split must not be used until the final audit.")


if __name__ == "__main__":
    main()