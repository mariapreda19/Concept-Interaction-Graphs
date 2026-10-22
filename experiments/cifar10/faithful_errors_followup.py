import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon, spearmanr
from tqdm.auto import tqdm
TIE_EPS = 1e-4
PRIMARY_QUANTILE = 0.75
STRICT_QUANTILE = 0.90
SURROGATE_C = 1.0
FOLLOWUP_DIR = os.path.join(
    OUTPUT_DIR,
    "followup")

os.makedirs(FOLLOWUP_DIR,exist_ok=True)
print("CIG FOLLOW-UP ANALYSIS")
def add_outcome_columns(df, eps=TIE_EPS):

    df = df.copy()

    advantage = df[
        "cig_advantage"
    ].values

    df["outcome"] = np.where(

        advantage > eps,

        "win",

        np.where(
            advantage < -eps,
            "loss",
            "tie"
        )
    )

    df["same_selection"] = (
        df["selection_jaccard"]
        >= 0.999999
    )

    df["changed_selection"] = (
        ~df["same_selection"]
    )

    return df


results2 = add_outcome_columns(
    results_df
)


def outcome_summary(df, label):

    rows = []

    for n in sorted(
        df["N"].unique()
    ):

        x = df[
            df["N"] == n
        ]

        if len(x) == 0:
            continue

        counts = (
            x["outcome"]
            .value_counts(
                normalize=True
            )
        )

        rows.append({

            "subset":
                label,

            "N":
                n,

            "images":
                len(x),

            "win_rate":
                counts.get(
                    "win",
                    0.0
                ),

            "tie_rate":
                counts.get(
                    "tie",
                    0.0
                ),

            "loss_rate":
                counts.get(
                    "loss",
                    0.0
                ),

            "mean_advantage":
                x[
                    "cig_advantage"
                ].mean(),

            "median_advantage":
                x[
                    "cig_advantage"
                ].median(),

            "mean_jaccard":
                x[
                    "selection_jaccard"
                ].mean(),

            "changed_selection_rate":
                x[
                    "changed_selection"
                ].mean(),

            "list_correction":
                x[
                    "list_corrected"
                ].mean(),

            "cig_correction":
                x[
                    "cig_corrected"
                ].mean()
        })

    return pd.DataFrame(
        rows
    )


overall_outcomes = outcome_summary(
    results2,
    "all errors"
)

print("\n" + "=" * 90)
print("1. TRUE WIN / TIE / LOSS")
print("=" * 90)

print(
    overall_outcomes
    .round(4)
    .to_string(
        index=False
    )
)

changed_df = results2[
    results2[
        "changed_selection"
    ]
].copy()

changed_summary = outcome_summary(
    changed_df,
    "CIG changed selection"
)

print("\n" + "=" * 90)
print("2. CASES WHERE CIG ACTUALLY CHANGES THE SELECTED CONCEPTS")
print("=" * 90)

print(
    changed_summary
    .round(4)
    .to_string(
        index=False
    )
)

def interaction_dominant_subset(
    df,
    quantile=0.75
):

    selected = []

    thresholds = []

    for n in sorted(
        df["N"].unique()
    ):

        x = df[
            df["N"] == n
        ].copy()

        if len(x) == 0:
            continue

        threshold = x[
            "interaction_energy_ratio"
        ].quantile(
            quantile
        )

        y = x[
            x[
                "interaction_energy_ratio"
            ]
            >= threshold
        ].copy()

        y[
            "interaction_ratio_threshold"
        ] = threshold

        selected.append(
            y
        )

        thresholds.append({

            "N":
                n,

            "quantile":
                quantile,

            "threshold":
                threshold,

            "selected_images":
                len(y),

            "total_images":
                len(x)
        })

    if len(selected) == 0:

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    return (
        pd.concat(
            selected,
            ignore_index=True
        ),
        pd.DataFrame(
            thresholds
        )
    )


interaction_q75, thresholds75 = (
    interaction_dominant_subset(
        results2,
        PRIMARY_QUANTILE
    )
)

interaction_q90, thresholds90 = (
    interaction_dominant_subset(
        results2,
        STRICT_QUANTILE
    )
)


q75_summary = outcome_summary(
    interaction_q75,
    "top-25% interaction/node ratio"
)

q90_summary = outcome_summary(
    interaction_q90,
    "top-10% interaction/node ratio"
)


print("\n" + "=" * 90)
print("3A. TOP 25% MOST INTERACTION-DRIVEN CASES")
print("=" * 90)

print(
    q75_summary
    .round(4)
    .to_string(
        index=False
    )
)

print("\nThresholds:")

print(
    thresholds75
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("3B. TOP 10% MOST INTERACTION-DRIVEN CASES")
print("=" * 90)

print(
    q90_summary
    .round(4)
    .to_string(
        index=False
    )
)

@torch.no_grad()
def build_concept_dataset(
    model,
    loader,
    n_concepts
):

    model.eval()

    concept_features = []

    cnn_predictions = []

    true_labels = []

    image_indices = []

    current_index = 0

    for images, labels in tqdm(
        loader,
        desc="Building concept surrogate dataset"
    ):

        B = images.size(0)

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        feat = (
            model
            .get_layer3_features(
                images
            )
        )

        assignments = assign_concepts(
            feat
        )

        logits = (
            model
            .forward_from_layer3(
                feat
            )
        )

        preds = (
            logits
            .argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

        B, H, W = (
            assignments.shape
        )

        batch_vectors = np.zeros(
            (
                B,
                n_concepts
            ),
            dtype=np.float32
        )

        for b in range(B):

            counts = torch.bincount(

                assignments[b]
                .flatten(),

                minlength=n_concepts
            ).float()

            counts = (
                counts
                /
                counts.sum()
            )

            batch_vectors[b] = (
                counts
                .cpu()
                .numpy()
            )

        concept_features.append(
            batch_vectors
        )

        cnn_predictions.append(
            preds
        )

        true_labels.append(
            labels.numpy()
        )

        image_indices.extend(
            range(
                current_index,
                current_index + B
            )
        )

        current_index += B

    return (
        np.concatenate(
            concept_features,
            axis=0
        ),
        np.concatenate(
            cnn_predictions
        ),
        np.concatenate(
            true_labels
        ),
        np.asarray(
            image_indices
        )
    )


surrogate_train_dataset = (
    torchvision.datasets.CIFAR10(

        root="/kaggle/working/cifar_data",

        train=True,

        download=False,

        transform=test_transform
    )
)

surrogate_train_loader = DataLoader(

    surrogate_train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)

surrogate_test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)


print("\n" + "=" * 90)
print("4. TRAINING NODE-ONLY CONCEPT SURROGATE")
print("=" * 90)


X_train, cnn_train_pred, y_train, train_indices = (
    build_concept_dataset(

        intervention_model,

        surrogate_train_loader,

        N_CONCEPTS
    )
)


X_test, cnn_test_pred, y_test, test_indices = (
    build_concept_dataset(

        intervention_model,

        surrogate_test_loader,

        N_CONCEPTS
    )
)


print(
    "\nConcept matrix train:",
    X_train.shape
)

print(
    "Concept matrix test :",
    X_test.shape
)


surrogate = LogisticRegression(

    C=SURROGATE_C,

    max_iter=3000,

    solver="lbfgs"
)

surrogate.fit(
    X_train,
    cnn_train_pred
)

surrogate_train_pred = (
    surrogate.predict(
        X_train
    )
)

surrogate_test_pred = (
    surrogate.predict(
        X_test
    )
)


train_fidelity = accuracy_score(
    cnn_train_pred,
    surrogate_train_pred
)

test_fidelity = accuracy_score(
    cnn_test_pred,
    surrogate_test_pred
)


surrogate_true_accuracy = accuracy_score(
    y_test,
    surrogate_test_pred
)

cnn_true_accuracy = accuracy_score(
    y_test,
    cnn_test_pred
)


print(
    f"\nCNN test accuracy:        "
    f"{cnn_true_accuracy:.4f}"
)

print(
    f"Surrogate test accuracy:  "
    f"{surrogate_true_accuracy:.4f}"
)

print(
    f"Surrogate train fidelity: "
    f"{train_fidelity:.4f}"
)

print(
    f"Surrogate test fidelity:  "
    f"{test_fidelity:.4f}"
)


cnn_error_mask = (
    cnn_test_pred
    != y_test
)

faithful_error_mask = (

    cnn_error_mask

    &

    (
        surrogate_test_pred
        == cnn_test_pred
    )
)

all_error_indices = (
    test_indices[
        cnn_error_mask
    ]
)

faithful_error_indices = (
    test_indices[
        faithful_error_mask
    ]
)


faithful_error_coverage = (

    faithful_error_mask.sum()

    /

    max(
        cnn_error_mask.sum(),
        1
    )
)


print("\n" + "=" * 90)
print("5. FAITHFUL ERROR COVERAGE")
print("=" * 90)

print(
    "All CNN test errors:",
    cnn_error_mask.sum()
)

print(
    "Faithful CNN errors:",
    faithful_error_mask.sum()
)

print(
    f"Faithful error coverage: "
    f"{faithful_error_coverage:.4f}"
)


faithful_set = set(
    faithful_error_indices.tolist()
)

results_faithful = results2[

    results2[
        "image_index"
    ]
    .isin(
        faithful_set
    )

].copy()


print(
    "\nFaithful errors among the 300 analysed:",
    results_faithful[
        "error_id"
    ].nunique()
)


faithful_summary = outcome_summary(
    results_faithful,
    "faithful errors"
)


print("\n" + "=" * 90)
print("6. CIG VS LIST — FAITHFUL ERRORS ONLY")
print("=" * 90)

print(
    faithful_summary
    .round(4)
    .to_string(
        index=False
    )
)


faithful_stats = []

for n in sorted(
    results_faithful[
        "N"
    ].unique()
):

    x = results_faithful[
        results_faithful[
            "N"
        ]
        == n
    ]

    if len(x) < 2:
        continue

    try:

        stat = wilcoxon(

            x[
                "cig_margin_improvement"
            ],

            x[
                "list_margin_improvement"
            ],

            alternative="greater"
        )

        p = stat.pvalue

    except Exception:

        p = np.nan

    faithful_stats.append({

        "N":
            n,

        "images":
            len(x),

        "mean_advantage":
            x[
                "cig_advantage"
            ].mean(),

        "median_advantage":
            x[
                "cig_advantage"
            ].median(),

        "win_rate":
            (
                x[
                    "outcome"
                ]
                == "win"
            ).mean(),

        "tie_rate":
            (
                x[
                    "outcome"
                ]
                == "tie"
            ).mean(),

        "loss_rate":
            (
                x[
                    "outcome"
                ]
                == "loss"
            ).mean(),

        "wilcoxon_one_sided_p":
            p
    })


faithful_stats_df = pd.DataFrame(
    faithful_stats
)

print("\nFaithful statistical comparison:")

print(
    faithful_stats_df
    .round(5)
    .to_string(
        index=False
    )
)

faithful_interaction_q75, faithful_thresholds75 = (
    interaction_dominant_subset(

        results_faithful,

        PRIMARY_QUANTILE
    )
)

faithful_interaction_summary = (
    outcome_summary(

        faithful_interaction_q75,

        "faithful + top-25% interaction-driven"
    )
)


print("\n" + "=" * 90)
print("7. FAITHFUL + TOP 25% INTERACTION-DRIVEN")
print("=" * 90)

print(
    faithful_interaction_summary
    .round(4)
    .to_string(
        index=False
    )
)


focused = faithful_interaction_q75[

    faithful_interaction_q75[
        "changed_selection"
    ]

].copy()


focused_summary = outcome_summary(
    focused,
    "faithful + interaction-driven + changed selection"
)


print("\n" + "=" * 90)
print("8. DIAGNOSTIC: WHERE THE GRAPH SHOULD ACTUALLY MATTER")
print("=" * 90)

print(
    focused_summary
    .round(4)
    .to_string(
        index=False
    )
)


corr_rows = []

for n in sorted(
    results_faithful[
        "N"
    ].unique()
):

    x = results_faithful[
        results_faithful[
            "N"
        ]
        == n
    ]

    if len(x) < 5:
        continue

    rho, p = spearmanr(

        x[
            "interaction_energy_ratio"
        ],

        x[
            "cig_advantage"
        ]
    )

    corr_rows.append({

        "N":
            n,

        "images":
            len(x),

        "rho":
            rho,

        "p":
            p
    })


faithful_corr_df = pd.DataFrame(
    corr_rows
)


print("\n" + "=" * 90)
print("9. INTERACTION/NODE RATIO -> CIG ADVANTAGE, FAITHFUL ERRORS")
print("=" * 90)

print(
    faithful_corr_df
    .round(6)
    .to_string(
        index=False
    )
)


overall_outcomes.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "01_all_win_tie_loss.csv"
    ),

    index=False
)

changed_summary.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "02_changed_selection.csv"
    ),

    index=False
)

q75_summary.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "03_top25_interaction_ratio.csv"
    ),

    index=False
)

q90_summary.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "04_top10_interaction_ratio.csv"
    ),

    index=False
)

faithful_summary.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "05_faithful_errors.csv"
    ),

    index=False
)

faithful_stats_df.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "06_faithful_stats.csv"
    ),

    index=False
)

faithful_interaction_summary.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "07_faithful_interaction_top25.csv"
    ),

    index=False
)

focused_summary.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "08_focused_diagnostic.csv"
    ),

    index=False
)

faithful_corr_df.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "09_faithful_interaction_correlation.csv"
    ),

    index=False
)

results_faithful.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "faithful_detailed_results.csv"
    ),

    index=False
)


tables = [

    overall_outcomes,

    changed_summary,

    q75_summary,

    faithful_summary,

    faithful_interaction_summary,

    focused_summary
]

tables = [
    x
    for x in tables
    if len(x) > 0
]

final_comparison = pd.concat(
    tables,
    ignore_index=True
)


print("FINAL COMPARISON")

cols = [

    "subset",
    "N",
    "images",
    "mean_advantage",
    "win_rate",
    "tie_rate",
    "loss_rate",
    "changed_selection_rate",
    "list_correction",
    "cig_correction"
]

print(
    final_comparison[
        cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)


final_comparison.to_csv(

    os.path.join(
        FOLLOWUP_DIR,
        "FINAL_COMPARISON.csv"
    ),

    index=False
)


print("\nSaved to:")
print(FOLLOWUP_DIR)

print("\nDONE.")