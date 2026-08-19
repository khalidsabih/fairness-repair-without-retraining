"""
Phase 1 - Synthetic benchmark generation and biased baseline training.

Generates 2,500 templated educational essays with a latent quality label and a
protected-group marker, injects label corruption into two group-label strata,
fine-tunes bert-base-uncased on the corrupted labels, and trains a clean
oracle-reference model on the uncorrupted labels for evaluation only.

Corresponds to Section 5.3 of the thesis.

Splits: train (70%) / fairness_probe (15%) / untouched_test (15%).

Corruption is confined to group A with high latent quality and group B with low
latent quality. The corruption indicator is recorded but is never used during
repair; it is read only in Phase 2 to evaluate ranking quality.
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from fairlearn.metrics import equalized_odds_difference
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


# =====================================================
# 1. Configuration
# =====================================================

class Config:
    model_name = "bert-base-uncased"
    num_labels = 2

    batch_size = 16
    epochs = 4
    learning_rate = 2e-5
    max_length = 96

    seed = 42

    bias_probability = 0.85
    ambiguous_phrase_probability = 0.20

    n_samples = 2500

    train_fraction = 0.70
    fairness_probe_fraction = 0.15
    test_fraction = 0.15

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_dir = "data"
    models_dir = "models"

    biased_output_dir = "models/m0_biased"
    oracle_output_dir = "models/mr_oracle_reference"

    train_biased_path = "data/train_biased.csv"
    fairness_probe_path = "data/fairness_probe.csv"
    test_path = "data/test.csv"
    train_oracle_path = "data/train_oracle.csv"

    corruption_summary_path = "data/corruption_summary.csv"
    phase1_metrics_path = "data/phase1_metrics.csv"
    experiment_config_path = "data/experiment_config.json"


config = Config()


# =====================================================
# 2. Reproducibility
# =====================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(config.seed)


# =====================================================
# 3. Validation
# =====================================================

def validate_split_fractions():
    total = (
        config.train_fraction
        + config.fairness_probe_fraction
        + config.test_fraction
    )

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Train, fairness-probe, and test fractions must sum to 1.0. "
            f"Received: {total}"
        )


validate_split_fractions()


# =====================================================
# 4. Synthetic Dataset
# =====================================================

def generate_sensitive_biased_data(n_samples):
    """
    Generates a controlled synthetic educational dataset.

    Each row contains:
    - stable global sample identifier;
    - essay text;
    - observed, potentially corrupted label;
    - clean latent quality;
    - demographic group;
    - corruption eligibility;
    - explicit corruption indicator.

    The corruption metadata is intended only for validating
    the later influence-ranking stage. It must not be used by
    the proposed intervention algorithm.
    """

    high_quality_phrases = [
        "the argument is mostly coherent and supported by relevant examples",
        "the analysis develops a clear position with some supporting evidence",
        "the response shows adequate reasoning and connects ideas effectively",
        "the essay presents a structured interpretation with mostly consistent logic",
        "the submission contains relevant claims and explains them with moderate depth",
    ]

    low_quality_phrases = [
        "the argument is only partially developed and examples are limited",
        "the analysis includes relevant points but lacks consistent explanation",
        "the response shows some reasoning but leaves important claims unsupported",
        "the essay presents a position but the structure is uneven",
        "the submission contains claims that are only weakly connected to evidence",
    ]

    ambiguous_phrases = [
        "the writing includes both clear and unclear parts",
        "some parts of the response are convincing while others remain underdeveloped",
        "the essay shows partial understanding with mixed evidence quality",
        "the structure is generally visible but not always sustained",
    ]

    neutral_contexts = [
        "The student response discusses a social issue in relation to the prompt.",
        "The essay addresses the question and attempts to organize a response.",
        "The submission reflects an attempt to compare multiple ideas.",
        "The response includes an introduction, body discussion, and conclusion.",
        "The essay refers to the reading material and develops a short argument.",
    ]

    rows = []

    for sample_id in range(n_samples):
        demo = np.random.choice(["A", "B"])
        true_quality = int(np.random.choice([0, 1]))

        if np.random.rand() < config.ambiguous_phrase_probability:
            quality_text = np.random.choice(ambiguous_phrases)
        elif true_quality == 1:
            quality_text = np.random.choice(high_quality_phrases)
        else:
            quality_text = np.random.choice(low_quality_phrases)

        neutral_text = np.random.choice(neutral_contexts)

        essay_text = (
            f"GROUP_{demo}: {neutral_text} "
            f"Evaluation note: {quality_text}. "
            f"Essay ID {sample_id}."
        )

        bias_eligible = int(
            (demo == "A" and true_quality == 1)
            or
            (demo == "B" and true_quality == 0)
        )

        observed_label = true_quality

        if bias_eligible and np.random.rand() < config.bias_probability:
            observed_label = 1 - true_quality

        was_corrupted = int(observed_label != true_quality)

        rows.append(
            {
                "sample_id": sample_id,
                "essay_text": essay_text,
                "observed_label": int(observed_label),
                "true_quality": true_quality,
                "demo": demo,
                "bias_eligible": bias_eligible,
                "was_corrupted": was_corrupted,
            }
        )

    dataframe = pd.DataFrame(rows)

    dataframe["stratum"] = (
        dataframe["demo"].astype(str)
        + "_"
        + dataframe["true_quality"].astype(str)
    )

    return dataframe


print("Generating controlled synthetic educational dataset...")

df_all = generate_sensitive_biased_data(
    n_samples=config.n_samples
)


# =====================================================
# 5. Three-Way Stratified Split
# =====================================================

train_df, temporary_df = train_test_split(
    df_all,
    test_size=(
        config.fairness_probe_fraction
        + config.test_fraction
    ),
    random_state=config.seed,
    stratify=df_all["stratum"],
)

temporary_test_fraction = (
    config.test_fraction
    /
    (
        config.fairness_probe_fraction
        + config.test_fraction
    )
)

fairness_probe_df, test_df = train_test_split(
    temporary_df,
    test_size=temporary_test_fraction,
    random_state=config.seed,
    stratify=temporary_df["stratum"],
)

train_df = train_df.reset_index(drop=True)
fairness_probe_df = fairness_probe_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df["split"] = "train"
fairness_probe_df["split"] = "fairness_probe"
test_df["split"] = "test"

train_df = train_df.drop(columns=["stratum"])
fairness_probe_df = fairness_probe_df.drop(columns=["stratum"])
test_df = test_df.drop(columns=["stratum"])


# =====================================================
# 6. Oracle Training Data
# =====================================================

train_oracle_df = train_df.copy()

train_oracle_df["observed_label"] = (
    train_oracle_df["true_quality"]
)

train_oracle_df["split"] = "train_oracle_reference"


# =====================================================
# 7. Save Data Splits
# =====================================================

os.makedirs(
    config.data_dir,
    exist_ok=True
)

os.makedirs(
    config.models_dir,
    exist_ok=True
)

train_df.to_csv(
    config.train_biased_path,
    index=False
)

fairness_probe_df.to_csv(
    config.fairness_probe_path,
    index=False
)

test_df.to_csv(
    config.test_path,
    index=False
)

train_oracle_df.to_csv(
    config.train_oracle_path,
    index=False
)

print("\nSaved dataset splits:")
print(f"Biased training set : {len(train_df)}")
print(f"Fairness probe set  : {len(fairness_probe_df)}")
print(f"Final test set      : {len(test_df)}")
print(f"Oracle training set : {len(train_oracle_df)}")


# =====================================================
# 8. Corruption Summary
# =====================================================

def build_corruption_summary(dataframe):
    summary = (
        dataframe
        .groupby(
            ["demo", "true_quality"],
            as_index=False
        )
        .agg(
            count=("sample_id", "count"),
            eligible=("bias_eligible", "sum"),
            corrupted=("was_corrupted", "sum"),
        )
    )

    summary["corruption_rate_all"] = (
        summary["corrupted"]
        /
        summary["count"]
    )

    summary["corruption_rate_eligible"] = np.where(
        summary["eligible"] > 0,
        summary["corrupted"] / summary["eligible"],
        0.0,
    )

    return summary


corruption_summary = build_corruption_summary(
    df_all
)

corruption_summary.to_csv(
    config.corruption_summary_path,
    index=False
)

print("\nCorruption summary:")
print(
    corruption_summary.to_string(
        index=False
    )
)

print(
    "\nOverall corrupted samples: "
    f"{int(df_all['was_corrupted'].sum())}"
    f"/{len(df_all)} "
    f"({df_all['was_corrupted'].mean():.4f})"
)

eligible_mask = (
    df_all["bias_eligible"] == 1
)

eligible_corruption_rate = (
    df_all.loc[
        eligible_mask,
        "was_corrupted"
    ].mean()
)

print(
    "Corruption rate among eligible samples: "
    f"{eligible_corruption_rate:.4f}"
)


# =====================================================
# 9. Dataset Class
# =====================================================

class EssayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
    ):
        self.encodings = tokenizer(
            dataframe["essay_text"].tolist(),
            truncation=True,
            padding=True,
            max_length=config.max_length,
        )

        self.labels = (
            dataframe["observed_label"]
            .astype(int)
            .tolist()
        )

    def __getitem__(self, index):
        item = {
            key: torch.tensor(
                values[index]
            )
            for key, values
            in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item

    def __len__(self):
        return len(self.labels)


tokenizer = BertTokenizer.from_pretrained(
    config.model_name
)


# =====================================================
# 10. Training Function
# =====================================================

def train_model(
    train_dataframe,
    output_dir,
    model_display_name,
):
    print(
        f"\nTraining {model_display_name}..."
    )

    start_time = time.time()

    model = (
        BertForSequenceClassification
        .from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        )
        .to(config.device)
    )

    train_dataset = EssayDataset(
        dataframe=train_dataframe,
        tokenizer=tokenizer,
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        logging_steps=50,
        seed=config.seed,
        data_seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
    )

    trainer.train()

    runtime_seconds = (
        time.time() - start_time
    )

    model.save_pretrained(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    print(
        f"{model_display_name} saved to: "
        f"{output_dir}"
    )

    print(
        f"{model_display_name} runtime: "
        f"{runtime_seconds:.2f} seconds"
    )

    return model, runtime_seconds


# =====================================================
# 11. Train Models
# =====================================================

m0_model, m0_runtime = train_model(
    train_dataframe=train_df,
    output_dir=config.biased_output_dir,
    model_display_name="M0 Biased",
)

oracle_model, oracle_runtime = train_model(
    train_dataframe=train_oracle_df,
    output_dir=config.oracle_output_dir,
    model_display_name="MR Oracle Reference",
)


# =====================================================
# 12. Group Metrics
# =====================================================

def compute_group_rates(
    y_true,
    predictions,
    demographic_groups,
):
    rates = {}

    for group in ["A", "B"]:
        mask = (
            demographic_groups == group
        )

        group_true = y_true[mask]
        group_predictions = predictions[mask]

        true_positives = np.sum(
            (group_true == 1)
            &
            (group_predictions == 1)
        )

        false_negatives = np.sum(
            (group_true == 1)
            &
            (group_predictions == 0)
        )

        false_positives = np.sum(
            (group_true == 0)
            &
            (group_predictions == 1)
        )

        true_negatives = np.sum(
            (group_true == 0)
            &
            (group_predictions == 0)
        )

        true_positive_rate = (
            true_positives
            /
            (
                true_positives
                + false_negatives
            )
            if (
                true_positives
                + false_negatives
            ) > 0
            else 0.0
        )

        false_positive_rate = (
            false_positives
            /
            (
                false_positives
                + true_negatives
            )
            if (
                false_positives
                + true_negatives
            ) > 0
            else 0.0
        )

        rates[f"TPR_{group}"] = (
            true_positive_rate
        )

        rates[f"FPR_{group}"] = (
            false_positive_rate
        )

    rates["TPR_Gap"] = abs(
        rates["TPR_A"]
        - rates["TPR_B"]
    )

    rates["FPR_Gap"] = abs(
        rates["FPR_A"]
        - rates["FPR_B"]
    )

    return rates


# =====================================================
# 13. Audit Function
# =====================================================

def audit_model(
    model,
    dataframe,
    model_name,
    evaluation_split,
):
    model.eval()

    texts = dataframe[
        "essay_text"
    ].tolist()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    ).to(config.device)

    with torch.no_grad():
        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )[:, 1].cpu().numpy()

        predictions = (
            probabilities > 0.5
        ).astype(int)

    clean_labels = dataframe[
        "true_quality"
    ].to_numpy()

    demographic_groups = dataframe[
        "demo"
    ].to_numpy()

    accuracy = accuracy_score(
        clean_labels,
        predictions,
    )

    f1 = f1_score(
        clean_labels,
        predictions,
        zero_division=0,
    )

    equalized_odds = (
        equalized_odds_difference(
            clean_labels,
            predictions,
            sensitive_features=(
                demographic_groups
            ),
        )
    )

    average_probability_a = np.mean(
        probabilities[
            demographic_groups == "A"
        ]
    )

    average_probability_b = np.mean(
        probabilities[
            demographic_groups == "B"
        ]
    )

    pass_rate_a = np.mean(
        predictions[
            demographic_groups == "A"
        ]
    )

    pass_rate_b = np.mean(
        predictions[
            demographic_groups == "B"
        ]
    )

    group_rates = compute_group_rates(
        y_true=clean_labels,
        predictions=predictions,
        demographic_groups=(
            demographic_groups
        ),
    )

    result = {
        "model": model_name,
        "evaluation_split": evaluation_split,
        "n_examples": len(dataframe),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "equalized_odds_difference": float(
            equalized_odds
        ),
        "average_positive_probability_A": float(
            average_probability_a
        ),
        "average_positive_probability_B": float(
            average_probability_b
        ),
        "pass_rate_A": float(
            pass_rate_a
        ),
        "pass_rate_B": float(
            pass_rate_b
        ),
        "TPR_A": float(
            group_rates["TPR_A"]
        ),
        "TPR_B": float(
            group_rates["TPR_B"]
        ),
        "FPR_A": float(
            group_rates["FPR_A"]
        ),
        "FPR_B": float(
            group_rates["FPR_B"]
        ),
        "TPR_gap": float(
            group_rates["TPR_Gap"]
        ),
        "FPR_gap": float(
            group_rates["FPR_Gap"]
        ),
    }

    print("\n==============================")
    print(
        f"{model_name} — "
        f"{evaluation_split}"
    )
    print("==============================")

    print(
        f"Accuracy              : "
        f"{accuracy:.4f}"
    )

    print(
        f"F1 Score              : "
        f"{f1:.4f}"
    )

    print(
        f"EO Difference         : "
        f"{equalized_odds:.4f}"
    )

    print(
        f"Average Probability A : "
        f"{average_probability_a:.4f}"
    )

    print(
        f"Average Probability B : "
        f"{average_probability_b:.4f}"
    )

    print(
        f"Pass Rate A           : "
        f"{pass_rate_a:.4f}"
    )

    print(
        f"Pass Rate B           : "
        f"{pass_rate_b:.4f}"
    )

    print(
        f"TPR Gap               : "
        f"{group_rates['TPR_Gap']:.4f}"
    )

    print(
        f"FPR Gap               : "
        f"{group_rates['FPR_Gap']:.4f}"
    )

    return result


# =====================================================
# 14. Initial Audits
# =====================================================

phase1_metrics = []

for model_name, model in [
    ("M0_Biased", m0_model),
    ("MR_Oracle_Reference", oracle_model),
]:
    phase1_metrics.append(
        audit_model(
            model=model,
            dataframe=fairness_probe_df,
            model_name=model_name,
            evaluation_split=(
                "fairness_probe"
            ),
        )
    )

    phase1_metrics.append(
        audit_model(
            model=model,
            dataframe=test_df,
            model_name=model_name,
            evaluation_split="test",
        )
    )


# =====================================================
# 15. Save Metrics
# =====================================================

metrics_df = pd.DataFrame(
    phase1_metrics
)

metrics_df["training_runtime_seconds"] = (
    metrics_df["model"].map(
        {
            "M0_Biased": m0_runtime,
            "MR_Oracle_Reference": (
                oracle_runtime
            ),
        }
    )
)

metrics_df.to_csv(
    config.phase1_metrics_path,
    index=False,
)

print(
    "\nSaved Phase 1 metrics to: "
    f"{config.phase1_metrics_path}"
)


# =====================================================
# 16. Save Experiment Configuration
# =====================================================

experiment_configuration = {
    "seed": config.seed,
    "model_name": config.model_name,
    "num_labels": config.num_labels,
    "n_samples": config.n_samples,
    "bias_probability": (
        config.bias_probability
    ),
    "ambiguous_phrase_probability": (
        config.ambiguous_phrase_probability
    ),
    "train_fraction": (
        config.train_fraction
    ),
    "fairness_probe_fraction": (
        config.fairness_probe_fraction
    ),
    "test_fraction": (
        config.test_fraction
    ),
    "epochs": config.epochs,
    "learning_rate": (
        config.learning_rate
    ),
    "batch_size": config.batch_size,
    "max_length": config.max_length,
    "device": str(config.device),
    "biased_model_path": (
        config.biased_output_dir
    ),
    "oracle_reference_path": (
        config.oracle_output_dir
    ),
    "algorithmic_note": (
        "The proposed intervention must use the fairness-probe "
        "split and must not use was_corrupted or training-set "
        "true_quality to rank harmful training samples."
    ),
    "oracle_note": (
        "The oracle is an experimental reference model and is "
        "not required by the proposed influence-guided intervention."
    ),
}

with open(
    config.experiment_config_path,
    "w",
    encoding="utf-8",
) as configuration_file:
    json.dump(
        experiment_configuration,
        configuration_file,
        indent=2,
    )

print(
    "Saved experiment configuration to: "
    f"{config.experiment_config_path}"
)

print(
    "Saved corruption summary to: "
    f"{config.corruption_summary_path}"
)

print("\nPhase 1 completed successfully.")