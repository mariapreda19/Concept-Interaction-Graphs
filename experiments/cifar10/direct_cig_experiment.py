import os
import gc
import math
import time
import random
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

from sklearn.cluster import MiniBatchKMeans
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

BATCH_SIZE = 128
NUM_WORKERS = 2

EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

MODEL_PATH = "/kaggle/working/cifar10_resnet18_cig.pth"
LOAD_IF_EXISTS = True
N_CONCEPTS = 32
DISCOVERY_IMAGES = 5000
MAX_CLUSTER_FEATURES = 100_000
MAX_ERRORS = 300
TOP_CANDIDATE_CONCEPTS = 12
N_VALUES = [2, 4, 6, 8]
LAMBDA_INTERACTION = 1.0
ONLY_POSITIVE_NODE_EFFECTS = True
OUTPUT_DIR = "/kaggle/working/cig_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "cig_causal_results.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "cig_summary.csv"
)

INTERACTION_CSV = os.path.join(
    OUTPUT_DIR,
    "cig_all_interactions.csv"
)

CONCEPT_CSV = os.path.join(
    OUTPUT_DIR,
    "concept_statistics.csv"
)

print("=" * 80)
print("CAUSAL CONCEPT INTERACTION GRAPH EXPERIMENT")
print("=" * 80)
print("Device:", DEVICE)
print("Seed:", SEED)
print("Concepts:", N_CONCEPTS)
print("Max errors:", MAX_ERRORS)
print("Candidate concepts:", TOP_CANDIDATE_CONCEPTS)
print("N:", N_VALUES)
print("=" * 80)

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

CIFAR_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR_STD = (
    0.2470,
    0.2435,
    0.2616
)

train_transform = T.Compose([

    T.RandomCrop(
        32,
        padding=4
    ),

    T.RandomHorizontalFlip(),

    T.ToTensor(),

    T.Normalize(
        CIFAR_MEAN,
        CIFAR_STD
    )
])

test_transform = T.Compose([

    T.ToTensor(),

    T.Normalize(
        CIFAR_MEAN,
        CIFAR_STD
    )
])

train_dataset = torchvision.datasets.CIFAR10(

    root="/kaggle/working/cifar_data",

    train=True,

    download=True,

    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(

    root="/kaggle/working/cifar_data",

    train=False,

    download=True,

    transform=test_transform
)

CLASS_NAMES = train_dataset.classes

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)

print("\nDataset loaded.")
print("Train:", len(train_dataset))
print("Test :", len(test_dataset))
print("Classes:", CLASS_NAMES)

class CIFARResNet18(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()

        m = resnet18(
            weights=None
        )

        m.conv1 = nn.Conv2d(

            3,

            64,

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False
        )

        m.maxpool = nn.Identity()

        m.fc = nn.Linear(

            m.fc.in_features,

            num_classes
        )

        self.model = m

    def forward(
        self,
        x
    ):

        return self.model(x)


model = CIFARResNet18(
    NUM_CLASSES
).to(DEVICE)

@torch.no_grad()
def evaluate_accuracy(
    model,
    loader
):

    model.eval()

    total = 0

    correct = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(images)

        preds = logits.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            preds == labels
        ).sum().item()

    return correct / max(
        total,
        1
    )

if (
    LOAD_IF_EXISTS
    and os.path.exists(MODEL_PATH)
):

    print("\nLoading model:", MODEL_PATH)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state
    )

else:

    print("\nTraining ResNet18...")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS
    )

    best_acc = -1

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for images, labels in pbar:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        scheduler.step()

        test_acc = evaluate_accuracy(
            model,
            test_loader
        )

        avg_loss = (
            running_loss
            / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"loss={avg_loss:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )


test_accuracy = evaluate_accuracy(
    model,
    test_loader
)

print(
    "\nFinal classifier accuracy:",
    round(test_accuracy, 4)
)

class ResNetInterventionModel(nn.Module):

    def __init__(
        self,
        original_model
    ):

        super().__init__()

        m = original_model.model

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool

        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.avgpool = m.avgpool

        self.fc = m.fc

    def get_layer3_features(
        self,
        x
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)

        return x

    def forward_from_layer3(
        self,
        feat
    ):

        x = self.layer4(
            feat
        )

        x = self.avgpool(
            x
        )

        x = torch.flatten(
            x,
            1
        )

        logits = self.fc(
            x
        )

        return logits

    def forward(
        self,
        x
    ):

        feat = self.get_layer3_features(
            x
        )

        return self.forward_from_layer3(
            feat
        )


intervention_model = ResNetInterventionModel(
    model
).to(DEVICE)

intervention_model.eval()

@torch.no_grad()
def collect_feature_vectors(
    model,
    loader,
    max_images,
    max_vectors
):

    model.eval()

    all_vectors = []

    collected_images = 0

    for images, _ in tqdm(
        loader,
        desc="Collecting layer3 features"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        feat = model.get_layer3_features(
            images
        )

        B, C, H, W = feat.shape

        vectors = (
            feat
            .permute(
                0,
                2,
                3,
                1
            )
            .reshape(
                -1,
                C
            )
        )

        vectors = F.normalize(
            vectors,
            p=2,
            dim=1
        )

        all_vectors.append(
            vectors.cpu().numpy()
        )

        collected_images += B

        if (
            collected_images
            >= max_images
        ):
            break

    X = np.concatenate(
        all_vectors,
        axis=0
    )

    if len(X) > max_vectors:

        ids = np.random.choice(

            len(X),

            size=max_vectors,

            replace=False
        )

        X = X[ids]

    return X


cluster_vectors = collect_feature_vectors(

    intervention_model,

    train_loader,

    DISCOVERY_IMAGES,

    MAX_CLUSTER_FEATURES
)

print(
    "\nFeature matrix:",
    cluster_vectors.shape
)

print("\nRunning MiniBatchKMeans...")

kmeans = MiniBatchKMeans(

    n_clusters=N_CONCEPTS,

    batch_size=4096,

    random_state=SEED,

    n_init=10,

    max_iter=200
)

kmeans.fit(
    cluster_vectors
)

concept_centers = torch.tensor(

    kmeans.cluster_centers_,

    dtype=torch.float32,

    device=DEVICE
)

concept_centers = F.normalize(

    concept_centers,

    p=2,

    dim=1
)

print(
    "Concept centers:",
    concept_centers.shape
)

@torch.no_grad()
def assign_concepts(
    feature_map
):

    B, C, H, W = (
        feature_map.shape
    )

    flat = (
        feature_map
        .permute(
            0,
            2,
            3,
            1
        )
        .reshape(
            -1,
            C
        )
    )

    flat_norm = F.normalize(

        flat,

        p=2,

        dim=1
    )

    similarity = (

        flat_norm

        @ concept_centers.T
    )

    assignments = similarity.argmax(
        dim=1
    )

    assignments = assignments.reshape(

        B,
        H,
        W
    )

    return assignments


@torch.no_grad()
def compute_concept_statistics(
    model,
    loader,
    max_images=3000
):

    counts = np.zeros(
        N_CONCEPTS,
        dtype=np.int64
    )

    image_counts = np.zeros(
        N_CONCEPTS,
        dtype=np.int64
    )

    total_positions = 0

    total_images = 0

    for images, _ in tqdm(
        loader,
        desc="Concept statistics"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        feat = model.get_layer3_features(
            images
        )

        assignments = assign_concepts(
            feat
        )

        B, H, W = (
            assignments.shape
        )

        for b in range(B):

            a = assignments[b]

            binc = torch.bincount(

                a.flatten(),

                minlength=N_CONCEPTS
            )

            counts += (
                binc.cpu().numpy()
            )

            present = torch.unique(
                a
            ).cpu().numpy()

            image_counts[present] += 1

        total_positions += (
            B * H * W
        )

        total_images += B

        if total_images >= max_images:
            break

    df = pd.DataFrame({

        "concept_id":
            np.arange(
                N_CONCEPTS
            ),

        "spatial_frequency":
            counts
            / max(
                total_positions,
                1
            ),

        "image_frequency":
            image_counts
            / max(
                total_images,
                1
            ),

        "raw_count":
            counts,

        "image_count":
            image_counts
    })

    return df


concept_stats = compute_concept_statistics(

    intervention_model,

    train_loader,

    max_images=min(
        DISCOVERY_IMAGES,
        3000
    )
)

concept_stats.to_csv(
    CONCEPT_CSV,
    index=False
)

print("\nConcept statistics:")
print(
    concept_stats.head(10)
)

@torch.no_grad()
def suppress_concepts(
    feature_map,
    concept_ids,
    assignments=None
):

    original = feature_map

    modified = (
        feature_map
        .clone()
    )

    B, C, H, W = (
        modified.shape
    )

    assert B == 1, (
        "Intervention currently expects "
        "one image at a time."
    )

    if assignments is None:

        assignments = assign_concepts(
            original
        )[0]

    spatial = (
        modified[0]
        .permute(
            1,
            2,
            0
        )
        .clone()
    )

    for cid in concept_ids:

        cid = int(cid)

        mask = (
            assignments
            == cid
        )

        if (
            mask.sum().item()
            == 0
        ):
            continue

        center = concept_centers[
            cid
        ]

        selected = spatial[
            mask
        ]

        coeff = (
            selected
            @ center
        )

        projection = (

            coeff[:, None]

            * center[None, :]
        )

        selected_new = (
            selected
            - projection
        )

        spatial[
            mask
        ] = selected_new

    modified[0] = spatial.permute(
        2,
        0,
        1
    )

    energy = torch.norm(
        original
        - modified,
        p=2
    ).item()

    return (
        modified,
        energy
    )

def classification_margin(
    logits,
    wrong_class,
    true_class
):

    return (

        logits[
            0,
            wrong_class
        ]

        -

        logits[
            0,
            true_class
        ]

    ).item()


@torch.no_grad()
def collect_errors(
    model,
    dataset,
    max_errors
):

    model.eval()

    errors = []

    for idx in tqdm(
        range(
            len(dataset)
        ),
        desc="Finding CNN errors"
    ):

        image, true_class = (
            dataset[idx]
        )

        x = (
            image
            .unsqueeze(0)
            .to(DEVICE)
        )

        logits = model(
            x
        )

        pred = (
            logits
            .argmax(
                dim=1
            )
            .item()
        )

        if pred != true_class:

            probs = F.softmax(
                logits,
                dim=1
            )

            confidence = (
                probs[
                    0,
                    pred
                ].item()
            )

            margin = (
                logits[
                    0,
                    pred
                ]
                -
                logits[
                    0,
                    true_class
                ]
            ).item()

            errors.append({

                "index":
                    idx,

                "true_class":
                    int(
                        true_class
                    ),

                "wrong_class":
                    int(
                        pred
                    ),

                "confidence":
                    confidence,

                "original_margin":
                    margin
            })

        if (
            len(errors)
            >= max_errors
        ):
            break

    return errors


errors = collect_errors(

    intervention_model,

    test_dataset,

    MAX_ERRORS
)

print(
    "\nClassification errors collected:",
    len(errors)
)


@torch.no_grad()
def analyse_node_effects(
    image,
    true_class,
    wrong_class
):

    x = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    feat = (
        intervention_model
        .get_layer3_features(
            x
        )
    )

    assignments = assign_concepts(
        feat
    )[0]

    original_logits = (
        intervention_model
        .forward_from_layer3(
            feat
        )
    )

    original_margin = (
        classification_margin(

            original_logits,

            wrong_class,

            true_class
        )
    )

    original_prediction = (
        original_logits
        .argmax(
            dim=1
        )
        .item()
    )

    present_concepts = (
        torch
        .unique(
            assignments
        )
        .cpu()
        .tolist()
    )

    node_effects = {}

    node_energies = {}

    node_new_margins = {}

    for cid in present_concepts:

        modified, energy = (
            suppress_concepts(

                feat,

                [cid],

                assignments=assignments
            )
        )

        logits = (
            intervention_model
            .forward_from_layer3(
                modified
            )
        )

        new_margin = (
            classification_margin(

                logits,

                wrong_class,

                true_class
            )
        )

        delta = (
            original_margin
            -
            new_margin
        )

        node_effects[
            int(cid)
        ] = float(
            delta
        )

        node_energies[
            int(cid)
        ] = float(
            energy
        )

        node_new_margins[
            int(cid)
        ] = float(
            new_margin
        )

    return {

        "feat":
            feat,

        "assignments":
            assignments,

        "original_logits":
            original_logits,

        "original_margin":
            float(
                original_margin
            ),

        "original_prediction":
            int(
                original_prediction
            ),

        "node_effects":
            node_effects,

        "node_energies":
            node_energies,

        "node_new_margins":
            node_new_margins,

        "present_concepts":
            present_concepts
    }


@torch.no_grad()
def compute_pair_interactions(

    feat,

    assignments,

    original_margin,

    node_effects,

    candidates,

    true_class,

    wrong_class
):

    interactions = {}

    pair_effects = {}

    pair_energies = {}

    rows = []

    for i, j in itertools.combinations(
        candidates,
        2
    ):

        modified, energy = (
            suppress_concepts(

                feat,

                [i, j],

                assignments=assignments
            )
        )

        logits = (
            intervention_model
            .forward_from_layer3(
                modified
            )
        )

        margin_ij = (
            classification_margin(

                logits,

                wrong_class,

                true_class
            )
        )

        delta_ij = (

            original_margin

            - margin_ij
        )

        delta_i = (
            node_effects[i]
        )

        delta_j = (
            node_effects[j]
        )

        interaction = (

            delta_ij

            - delta_i

            - delta_j
        )

        key = tuple(
            sorted(
                (
                    int(i),
                    int(j)
                )
            )
        )

        interactions[
            key
        ] = float(
            interaction
        )

        pair_effects[
            key
        ] = float(
            delta_ij
        )

        pair_energies[
            key
        ] = float(
            energy
        )

        rows.append({

            "concept_i":
                int(i),

            "concept_j":
                int(j),

            "delta_i":
                float(delta_i),

            "delta_j":
                float(delta_j),

            "delta_ij":
                float(delta_ij),

            "interaction":
                float(
                    interaction
                ),

            "abs_interaction":
                float(
                    abs(
                        interaction
                    )
                ),

            "pair_energy":
                float(
                    energy
                )
        })

    return (
        interactions,
        pair_effects,
        pair_energies,
        rows
    )


def select_top_nodes(
    node_effects,
    candidates,
    n
):

    candidate_scores = [

        (
            cid,
            node_effects[cid]
        )

        for cid in candidates
    ]

    candidate_scores = sorted(

        candidate_scores,

        key=lambda x: x[1],

        reverse=True
    )

    return [

        int(cid)

        for cid, _
        in candidate_scores[:n]
    ]


def graph_score(

    subset,

    node_effects,

    interactions,

    lambda_interaction=1.0
):

    node_score = sum(

        node_effects[cid]

        for cid in subset
    )

    interaction_score = 0.0

    for i, j in itertools.combinations(
        subset,
        2
    ):

        key = tuple(
            sorted(
                (
                    int(i),
                    int(j)
                )
            )
        )

        interaction_score += (
            interactions.get(
                key,
                0.0
            )
        )

    total = (

        node_score

        +

        lambda_interaction
        * interaction_score
    )

    return (
        float(total),
        float(node_score),
        float(interaction_score)
    )


def select_cig_nodes(

    candidates,

    node_effects,

    interactions,

    n,

    lambda_interaction=1.0
):

    if len(candidates) <= n:

        subset = tuple(
            candidates
        )

        score, ns, es = graph_score(

            subset,

            node_effects,

            interactions,

            lambda_interaction
        )

        return (
            list(subset),
            score,
            ns,
            es
        )

    best_subset = None

    best_score = -float(
        "inf"
    )

    best_node_score = None

    best_edge_score = None

    for subset in itertools.combinations(
        candidates,
        n
    ):

        score, node_score, edge_score = (
            graph_score(

                subset,

                node_effects,

                interactions,

                lambda_interaction
            )
        )

        if score > best_score:

            best_score = score

            best_subset = subset

            best_node_score = (
                node_score
            )

            best_edge_score = (
                edge_score
            )

    return (

        list(
            best_subset
        ),

        float(
            best_score
        ),

        float(
            best_node_score
        ),

        float(
            best_edge_score
        )
    )


@torch.no_grad()
def evaluate_intervention(

    feat,

    assignments,

    selected_concepts,

    true_class,

    wrong_class,

    original_margin
):

    modified, energy = (
        suppress_concepts(

            feat,

            selected_concepts,

            assignments=assignments
        )
    )

    logits = (
        intervention_model
        .forward_from_layer3(
            modified
        )
    )

    new_margin = (
        classification_margin(

            logits,

            wrong_class,

            true_class
        )
    )

    new_prediction = (
        logits
        .argmax(
            dim=1
        )
        .item()
    )

    margin_improvement = (

        original_margin

        - new_margin
    )

    positive_margin_result = int(
        new_margin < 0
    )

    corrected = int(
        new_prediction
        == true_class
    )

    normalized_recovery = (

        margin_improvement

        / (
            abs(
                original_margin
            )

            + 1e-8
        )
    )

    energy_normalized = (

        margin_improvement

        / (
            energy
            + 1e-8
        )
    )

    probs = F.softmax(
        logits,
        dim=1
    )

    true_prob = (
        probs[
            0,
            true_class
        ].item()
    )

    wrong_prob = (
        probs[
            0,
            wrong_class
        ].item()
    )

    return {

        "new_margin":
            float(
                new_margin
            ),

        "margin_improvement":
            float(
                margin_improvement
            ),

        "normalized_recovery":
            float(
                normalized_recovery
            ),

        "energy":
            float(
                energy
            ),

        "energy_normalized_improvement":
            float(
                energy_normalized
            ),

        "corrected":
            int(
                corrected
            ),

        "positive_margin_result":
            int(
                positive_margin_result
            ),

        "new_prediction":
            int(
                new_prediction
            ),

        "true_probability":
            float(
                true_prob
            ),

        "wrong_probability":
            float(
                wrong_prob
            )
    }


results = []

all_interaction_rows = []

skipped_too_few = 0

start_time = time.time()

pbar = tqdm(
    errors,
    desc="Causal CIG experiment"
)

for error_id, error in enumerate(
    pbar
):

    idx = int(
        error["index"]
    )

    true_class = int(
        error["true_class"]
    )

    wrong_class = int(
        error["wrong_class"]
    )

    image, _ = (
        test_dataset[idx]
    )

    analysis = (
        analyse_node_effects(

            image,

            true_class,

            wrong_class
        )
    )

    feat = analysis[
        "feat"
    ]

    assignments = analysis[
        "assignments"
    ]

    original_margin = analysis[
        "original_margin"
    ]

    node_effects = analysis[
        "node_effects"
    ]

    node_energies = analysis[
        "node_energies"
    ]

    if ONLY_POSITIVE_NODE_EFFECTS:

        useful_nodes = {

            cid: effect

            for cid, effect
            in node_effects.items()

            if effect > 0
        }

    else:

        useful_nodes = (
            node_effects.copy()
        )

    if len(useful_nodes) < 2:

        skipped_too_few += 1

        continue

    ranked_concepts = sorted(

        useful_nodes.keys(),

        key=lambda cid:
            useful_nodes[cid],

        reverse=True
    )

    candidates = (
        ranked_concepts[
            :TOP_CANDIDATE_CONCEPTS
        ]
    )

    (
        interactions,
        pair_effects,
        pair_energies,
        interaction_rows
    ) = compute_pair_interactions(

        feat,

        assignments,

        original_margin,

        node_effects,

        candidates,

        true_class,

        wrong_class
    )

    for r in interaction_rows:

        r.update({

            "error_id":
                int(
                    error_id
                ),

            "image_index":
                int(
                    idx
                ),

            "true_class":
                int(
                    true_class
                ),

            "wrong_class":
                int(
                    wrong_class
                ),

            "true_name":
                CLASS_NAMES[
                    true_class
                ],

            "wrong_name":
                CLASS_NAMES[
                    wrong_class
                ],

            "original_margin":
                float(
                    original_margin
                )
        })

        all_interaction_rows.append(
            r
        )

    interaction_values = np.array(
        list(
            interactions.values()
        ),
        dtype=float
    )

    if (
        len(
            interaction_values
        )
        > 0
    ):

        mean_abs_interaction = (
            np.mean(
                np.abs(
                    interaction_values
                )
            )
        )

        max_abs_interaction = (
            np.max(
                np.abs(
                    interaction_values
                )
            )
        )

        mean_signed_interaction = (
            np.mean(
                interaction_values
            )
        )

        positive_interaction_fraction = (
            np.mean(
                interaction_values
                > 0
            )
        )

        interaction_energy_ratio = (

            np.sum(
                np.abs(
                    interaction_values
                )
            )

            /

            (
                np.sum(
                    np.abs(
                        [
                            node_effects[c]
                            for c in candidates
                        ]
                    )
                )

                + 1e-8
            )
        )

    else:

        mean_abs_interaction = 0.0
        max_abs_interaction = 0.0
        mean_signed_interaction = 0.0
        positive_interaction_fraction = 0.0
        interaction_energy_ratio = 0.0


    for n in N_VALUES:

        if len(candidates) < n:
            continue


        list_nodes = (
            select_top_nodes(

                node_effects,

                candidates,

                n
            )
        )

        list_result = (
            evaluate_intervention(

                feat,

                assignments,

                list_nodes,

                true_class,

                wrong_class,

                original_margin
            )
        )

        (
            cig_nodes,
            cig_selection_score,
            cig_node_component,
            cig_interaction_component
        ) = select_cig_nodes(

            candidates,

            node_effects,

            interactions,

            n,

            LAMBDA_INTERACTION
        )

        cig_result = (
            evaluate_intervention(

                feat,

                assignments,

                cig_nodes,

                true_class,

                wrong_class,

                original_margin
            )
        )

        overlap = len(

            set(
                list_nodes
            )

            &

            set(
                cig_nodes
            )
        )

        jaccard = (

            overlap

            /

            max(
                len(
                    set(
                        list_nodes
                    )
                    |
                    set(
                        cig_nodes
                    )
                ),
                1
            )
        )

        cig_advantage = (

            cig_result[
                "margin_improvement"
            ]

            -

            list_result[
                "margin_improvement"
            ]
        )

        cig_normalized_advantage = (

            cig_result[
                "normalized_recovery"
            ]

            -

            list_result[
                "normalized_recovery"
            ]
        )

        energy_normalized_advantage = (

            cig_result[
                "energy_normalized_improvement"
            ]

            -

            list_result[
                "energy_normalized_improvement"
            ]
        )

        results.append({

            "error_id":
                int(
                    error_id
                ),

            "image_index":
                int(
                    idx
                ),

            "true_class":
                int(
                    true_class
                ),

            "wrong_class":
                int(
                    wrong_class
                ),

            "true_name":
                CLASS_NAMES[
                    true_class
                ],

            "wrong_name":
                CLASS_NAMES[
                    wrong_class
                ],

            "N":
                int(
                    n
                ),

            "original_margin":
                float(
                    original_margin
                ),

            "num_present_concepts":
                int(
                    len(
                        node_effects
                    )
                ),

            "num_positive_concepts":
                int(
                    len(
                        useful_nodes
                    )
                ),

            "num_candidates":
                int(
                    len(
                        candidates
                    )
                ),

            "interaction_strength_mean_abs":
                float(
                    mean_abs_interaction
                ),

            "interaction_strength_max_abs":
                float(
                    max_abs_interaction
                ),

            "interaction_mean_signed":
                float(
                    mean_signed_interaction
                ),

            "positive_interaction_fraction":
                float(
                    positive_interaction_fraction
                ),

            "interaction_energy_ratio":
                float(
                    interaction_energy_ratio
                ),


            "list_margin_improvement":
                list_result[
                    "margin_improvement"
                ],

            "list_normalized_recovery":
                list_result[
                    "normalized_recovery"
                ],

            "list_energy":
                list_result[
                    "energy"
                ],

            "list_energy_normalized_improvement":
                list_result[
                    "energy_normalized_improvement"
                ],

            "list_corrected":
                list_result[
                    "corrected"
                ],

            "list_positive_margin":
                list_result[
                    "positive_margin_result"
                ],

            "list_new_margin":
                list_result[
                    "new_margin"
                ],

            "list_true_probability":
                list_result[
                    "true_probability"
                ],

            "list_wrong_probability":
                list_result[
                    "wrong_probability"
                ],


            "cig_margin_improvement":
                cig_result[
                    "margin_improvement"
                ],

            "cig_normalized_recovery":
                cig_result[
                    "normalized_recovery"
                ],

            "cig_energy":
                cig_result[
                    "energy"
                ],

            "cig_energy_normalized_improvement":
                cig_result[
                    "energy_normalized_improvement"
                ],

            "cig_corrected":
                cig_result[
                    "corrected"
                ],

            "cig_positive_margin":
                cig_result[
                    "positive_margin_result"
                ],

            "cig_new_margin":
                cig_result[
                    "new_margin"
                ],

            "cig_true_probability":
                cig_result[
                    "true_probability"
                ],

            "cig_wrong_probability":
                cig_result[
                    "wrong_probability"
                ],


            "cig_advantage":
                float(
                    cig_advantage
                ),

            "cig_normalized_advantage":
                float(
                    cig_normalized_advantage
                ),

            "energy_normalized_advantage":
                float(
                    energy_normalized_advantage
                ),

            "cig_win":
                int(
                    cig_advantage > 0
                ),

            "same_correction_status":
                int(
                    cig_result[
                        "corrected"
                    ]
                    ==
                    list_result[
                        "corrected"
                    ]
                ),

            "selection_overlap":
                int(
                    overlap
                ),

            "selection_jaccard":
                float(
                    jaccard
                ),

            "cig_selection_score":
                float(
                    cig_selection_score
                ),

            "cig_node_component":
                float(
                    cig_node_component
                ),

            "cig_interaction_component":
                float(
                    cig_interaction_component
                ),


            "list_nodes":
                ",".join(
                    map(
                        str,
                        list_nodes
                    )
                ),

            "cig_nodes":
                ",".join(
                    map(
                        str,
                        cig_nodes
                    )
                )
        })

    if (
        torch.cuda.is_available()
        and error_id % 25 == 0
    ):

        torch.cuda.empty_cache()


results_df = pd.DataFrame(
    results
)

interaction_df = pd.DataFrame(
    all_interaction_rows
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)

interaction_df.to_csv(
    INTERACTION_CSV,
    index=False
)

print("\n" + "=" * 80)
print("MAIN EXPERIMENT FINISHED")
print("=" * 80)

print(
    "Rows:",
    len(
        results_df
    )
)

print(
    "Errors skipped because too few useful concepts:",
    skipped_too_few
)

print(
    "Runtime (minutes):",
    round(
        (
            time.time()
            -
            start_time
        )
        / 60,
        2
    )
)


def bootstrap_mean_ci(

    values,

    n_bootstrap=5000,

    confidence=0.95,

    seed=42
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        n_bootstrap
    )

    for b in range(
        n_bootstrap
    ):

        sample = rng.choice(

            values,

            size=len(
                values
            ),

            replace=True
        )

        bootstrap_means[b] = (
            np.mean(
                sample
            )
        )

    alpha = (
        1
        -
        confidence
    ) / 2

    lower = np.quantile(
        bootstrap_means,
        alpha
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha
    )

    return (

        float(
            np.mean(
                values
            )
        ),

        float(
            lower
        ),

        float(
            upper
        )
    )


summary_rows = []

if len(results_df) > 0:

    for n in N_VALUES:

        subset = (
            results_df[
                results_df[
                    "N"
                ]
                == n
            ]
            .copy()
        )

        if len(subset) == 0:
            continue

        adv_mean, adv_lo, adv_hi = (
            bootstrap_mean_ci(
                subset[
                    "cig_advantage"
                ].values
            )
        )

        norm_mean, norm_lo, norm_hi = (
            bootstrap_mean_ci(
                subset[
                    "cig_normalized_advantage"
                ].values
            )
        )

        energy_mean, energy_lo, energy_hi = (
            bootstrap_mean_ci(
                subset[
                    "energy_normalized_advantage"
                ].values
            )
        )

        try:

            wilcoxon_result = wilcoxon(

                subset[
                    "cig_margin_improvement"
                ],

                subset[
                    "list_margin_improvement"
                ],

                alternative="two-sided"
            )

            wilcoxon_p = float(
                wilcoxon_result.pvalue
            )

        except Exception:

            wilcoxon_p = np.nan

        summary_rows.append({

            "N":
                n,

            "num_images":
                len(
                    subset
                ),

            "list_margin_improvement":
                subset[
                    "list_margin_improvement"
                ].mean(),

            "cig_margin_improvement":
                subset[
                    "cig_margin_improvement"
                ].mean(),

            "cig_advantage":
                adv_mean,

            "cig_advantage_CI_low":
                adv_lo,

            "cig_advantage_CI_high":
                adv_hi,

            "list_normalized_recovery":
                subset[
                    "list_normalized_recovery"
                ].mean(),

            "cig_normalized_recovery":
                subset[
                    "cig_normalized_recovery"
                ].mean(),

            "normalized_advantage":
                norm_mean,

            "normalized_advantage_CI_low":
                norm_lo,

            "normalized_advantage_CI_high":
                norm_hi,

            "list_energy_normalized":
                subset[
                    "list_energy_normalized_improvement"
                ].mean(),

            "cig_energy_normalized":
                subset[
                    "cig_energy_normalized_improvement"
                ].mean(),

            "energy_normalized_advantage":
                energy_mean,

            "energy_normalized_CI_low":
                energy_lo,

            "energy_normalized_CI_high":
                energy_hi,

            "list_correction_rate":
                subset[
                    "list_corrected"
                ].mean(),

            "cig_correction_rate":
                subset[
                    "cig_corrected"
                ].mean(),

            "list_positive_margin_rate":
                subset[
                    "list_positive_margin"
                ].mean(),

            "cig_positive_margin_rate":
                subset[
                    "cig_positive_margin"
                ].mean(),

            "cig_win_rate":
                subset[
                    "cig_win"
                ].mean(),

            "mean_selection_jaccard":
                subset[
                    "selection_jaccard"
                ].mean(),

            "wilcoxon_p":
                wilcoxon_p
        })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)

print("\n" + "=" * 80)
print("MAIN SUMMARY")
print("=" * 80)

if len(summary_df) > 0:

    print(
        summary_df.round(
            4
        ).to_string(
            index=False
        )
    )



quartile_tables = []

for n in N_VALUES:

    subset = (
        results_df[
            results_df[
                "N"
            ]
            == n
        ]
        .copy()
    )

    if len(subset) < 8:
        continue

    try:

        subset[
            "interaction_quartile"
        ] = pd.qcut(

            subset[
                "interaction_strength_mean_abs"
            ],

            q=4,

            labels=[
                "Q1 weak",
                "Q2",
                "Q3",
                "Q4 strong"
            ],

            duplicates="drop"
        )

    except Exception:

        continue

    table = (

        subset

        .groupby(
            "interaction_quartile",
            observed=False
        )

        .agg(

            images=(
                "cig_advantage",
                "size"
            ),

            interaction_strength=(
                "interaction_strength_mean_abs",
                "mean"
            ),

            list_margin_improvement=(
                "list_margin_improvement",
                "mean"
            ),

            cig_margin_improvement=(
                "cig_margin_improvement",
                "mean"
            ),

            cig_advantage=(
                "cig_advantage",
                "mean"
            ),

            cig_win_rate=(
                "cig_win",
                "mean"
            ),

            list_correction_rate=(
                "list_corrected",
                "mean"
            ),

            cig_correction_rate=(
                "cig_corrected",
                "mean"
            ),

            normalized_advantage=(
                "cig_normalized_advantage",
                "mean"
            )
        )

        .reset_index()
    )

    table.insert(
        0,
        "N",
        n
    )

    quartile_tables.append(
        table
    )


if len(quartile_tables) > 0:

    quartile_df = pd.concat(
        quartile_tables,
        ignore_index=True
    )

    quartile_path = os.path.join(
        OUTPUT_DIR,
        "interaction_quartiles.csv"
    )

    quartile_df.to_csv(
        quartile_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("INTERACTION QUARTILES")
    print("=" * 80)

    print(
        quartile_df.round(
            4
        ).to_string(
            index=False
        )
    )


strong_rows = []

for n in N_VALUES:

    subset = (
        results_df[
            results_df[
                "N"
            ]
            == n
        ]
        .copy()
    )

    if len(subset) == 0:
        continue

    threshold = (
        subset[
            "interaction_strength_mean_abs"
        ]
        .quantile(
            0.75
        )
    )

    strong = subset[

        subset[
            "interaction_strength_mean_abs"
        ]
        >= threshold
    ]

    adv_mean, adv_lo, adv_hi = (
        bootstrap_mean_ci(

            strong[
                "cig_advantage"
            ].values
        )
    )

    strong_rows.append({

        "N":
            n,

        "threshold":
            threshold,

        "num_images":
            len(
                strong
            ),

        "list_margin_improvement":
            strong[
                "list_margin_improvement"
            ].mean(),

        "cig_margin_improvement":
            strong[
                "cig_margin_improvement"
            ].mean(),

        "cig_advantage":
            adv_mean,

        "CI_low":
            adv_lo,

        "CI_high":
            adv_hi,

        "cig_win_rate":
            strong[
                "cig_win"
            ].mean(),

        "list_correction_rate":
            strong[
                "list_corrected"
            ].mean(),

        "cig_correction_rate":
            strong[
                "cig_corrected"
            ].mean(),

        "list_normalized_recovery":
            strong[
                "list_normalized_recovery"
            ].mean(),

        "cig_normalized_recovery":
            strong[
                "cig_normalized_recovery"
            ].mean()
    })


strong_df = pd.DataFrame(
    strong_rows
)

strong_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "strong_interaction_results.csv"
    ),

    index=False
)

print("\n" + "=" * 80)
print("TOP-QUARTILE STRONG-INTERACTION RESULTS")
print("=" * 80)

if len(strong_df) > 0:

    print(
        strong_df.round(
            4
        ).to_string(
            index=False
        )
    )


correlation_rows = []

print("\n" + "=" * 80)
print("INTERACTION STRENGTH -> CIG ADVANTAGE")
print("=" * 80)

for n in N_VALUES:

    subset = (
        results_df[
            results_df[
                "N"
            ]
            == n
        ]
        .dropna(
            subset=[
                "interaction_strength_mean_abs",
                "cig_advantage"
            ]
        )
    )

    if len(subset) < 3:
        continue

    rho, p = spearmanr(

        subset[
            "interaction_strength_mean_abs"
        ],

        subset[
            "cig_advantage"
        ]
    )

    rho_ratio, p_ratio = spearmanr(

        subset[
            "interaction_energy_ratio"
        ],

        subset[
            "cig_advantage"
        ]
    )

    correlation_rows.append({

        "N":
            n,

        "spearman_mean_abs":
            rho,

        "p_mean_abs":
            p,

        "spearman_interaction_ratio":
            rho_ratio,

        "p_interaction_ratio":
            p_ratio
    })

    print(
        f"N={n}: "
        f"rho={rho:.4f}, p={p:.6g} | "
        f"ratio rho={rho_ratio:.4f}, p={p_ratio:.6g}"
    )


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "interaction_advantage_correlations.csv"
    ),

    index=False
)


if len(summary_df) > 0:

    plt.figure(
        figsize=(
            8,
            5
        )
    )

    x = np.arange(
        len(
            summary_df
        )
    )

    width = 0.35

    plt.bar(

        x - width / 2,

        summary_df[
            "list_margin_improvement"
        ],

        width,

        label="Concept list"
    )

    plt.bar(

        x + width / 2,

        summary_df[
            "cig_margin_improvement"
        ],

        width,

        label="CIG"
    )

    plt.xticks(

        x,

        [
            f"N={n}"
            for n in summary_df[
                "N"
            ]
        ]
    )

    plt.ylabel(
        "Mean wrong-vs-true margin improvement"
    )

    plt.title(
        "Concept list vs Causal Concept Interaction Graph"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "01_list_vs_cig_margin.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()


if (
    "quartile_df"
    in globals()
):

    for n in N_VALUES:

        q = quartile_df[
            quartile_df[
                "N"
            ]
            == n
        ]

        if len(q) == 0:
            continue

        plt.figure(
            figsize=(
                7,
                4.5
            )
        )

        plt.bar(

            q[
                "interaction_quartile"
            ].astype(str),

            q[
                "cig_advantage"
            ]
        )

        plt.axhline(
            0,
            linewidth=1
        )

        plt.xlabel(
            "CNN concept-interaction strength"
        )

        plt.ylabel(
            "Mean CIG advantage over list"
        )

        plt.title(
            f"CIG advantage by interaction regime — N={n}"
        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                OUTPUT_DIR,
                f"02_interaction_quartiles_N{n}.png"
            ),

            dpi=180,

            bbox_inches="tight"
        )

        plt.show()


if (
    "quartile_df"
    in globals()
):

    for n in N_VALUES:

        q = quartile_df[
            quartile_df[
                "N"
            ]
            == n
        ]

        if len(q) == 0:
            continue

        plt.figure(
            figsize=(
                7,
                4.5
            )
        )

        plt.bar(

            q[
                "interaction_quartile"
            ].astype(str),

            q[
                "cig_win_rate"
            ]
        )

        plt.axhline(
            0.5,
            linestyle="--",
            linewidth=1
        )

        plt.ylim(
            0,
            1
        )

        plt.xlabel(
            "CNN concept-interaction strength"
        )

        plt.ylabel(
            "P(CIG beats node list)"
        )

        plt.title(
            f"CIG win rate by interaction regime — N={n}"
        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                OUTPUT_DIR,
                f"03_win_rate_quartiles_N{n}.png"
            ),

            dpi=180,

            bbox_inches="tight"
        )

        plt.show()


for n in N_VALUES:

    subset = (
        results_df[
            results_df[
                "N"
            ]
            == n
        ]
        .copy()
    )

    if len(subset) < 3:
        continue

    plt.figure(
        figsize=(
            7,
            5
        )
    )

    plt.scatter(

        subset[
            "interaction_strength_mean_abs"
        ],

        subset[
            "cig_advantage"
        ],

        alpha=0.55,

        s=20
    )

    plt.axhline(
        0,
        linewidth=1
    )

    rho, p = spearmanr(

        subset[
            "interaction_strength_mean_abs"
        ],

        subset[
            "cig_advantage"
        ]
    )

    plt.xlabel(
        "Mean absolute pairwise interaction"
    )

    plt.ylabel(
        "CIG advantage over list"
    )

    plt.title(
        f"Interaction strength vs CIG advantage — "
        f"N={n}\nSpearman rho={rho:.3f}, p={p:.3g}"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            f"04_scatter_interaction_advantage_N{n}.png"
        ),

        dpi=180,

        bbox_inches="tight"
    )

    plt.show()


if len(summary_df) > 0:

    plt.figure(
        figsize=(
            8,
            5
        )
    )

    x = np.arange(
        len(
            summary_df
        )
    )

    width = 0.35

    plt.bar(

        x - width / 2,

        summary_df[
            "list_correction_rate"
        ],

        width,

        label="Concept list"
    )

    plt.bar(

        x + width / 2,

        summary_df[
            "cig_correction_rate"
        ],

        width,

        label="CIG"
    )

    plt.xticks(

        x,

        [
            f"N={n}"
            for n
            in summary_df[
                "N"
            ]
        ]
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "CNN error correction rate"
    )

    plt.title(
        "Correction rate after feature-space intervention"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "05_correction_rates.png"
        ),

        dpi=180,

        bbox_inches="tight"
    )

    plt.show()


if len(interaction_df) > 0:

    top_interactions = (

        interaction_df

        .sort_values(
            "interaction",
            ascending=False
        )

        .head(
            30
        )
    )

    print("\n" + "=" * 80)
    print("TOP POSITIVE / SYNERGISTIC INTERACTIONS")
    print("=" * 80)

    print(

        top_interactions[[
            "image_index",
            "true_name",
            "wrong_name",
            "concept_i",
            "concept_j",
            "delta_i",
            "delta_j",
            "delta_ij",
            "interaction"
        ]]

        .round(
            4
        )

        .to_string(
            index=False
        )
    )


if len(interaction_df) > 0:

    bottom_interactions = (

        interaction_df

        .sort_values(
            "interaction",
            ascending=True
        )

        .head(
            30
        )
    )

    print("\n" + "=" * 80)
    print("MOST NEGATIVE / REDUNDANT INTERACTIONS")
    print("=" * 80)

    print(

        bottom_interactions[[
            "image_index",
            "true_name",
            "wrong_name",
            "concept_i",
            "concept_j",
            "delta_i",
            "delta_j",
            "delta_ij",
            "interaction"
        ]]

        .round(
            4
        )

        .to_string(
            index=False
        )
    )



if len(interaction_df) > 0:

    error_mode_edges = (

        interaction_df

        .groupby([
            "true_name",
            "wrong_name",
            "concept_i",
            "concept_j"
        ])

        .agg(

            mean_interaction=(
                "interaction",
                "mean"
            ),

            mean_abs_interaction=(
                "abs_interaction",
                "mean"
            ),

            edge_frequency=(
                "interaction",
                "size"
            ),

            mean_joint_effect=(
                "delta_ij",
                "mean"
            )
        )

        .reset_index()
    )

    error_mode_edges.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "error_mode_graph_edges.csv"
        ),

        index=False
    )

    print("\n" + "=" * 80)
    print("STRONGEST ERROR-MODE EDGES")
    print("=" * 80)

    print(

        error_mode_edges

        .sort_values(
            [
                "edge_frequency",
                "mean_interaction"
            ],

            ascending=[
                False,
                False
            ]
        )

        .head(
            30
        )

        .round(
            4
        )

        .to_string(
            index=False
        )
    )


print("\n\n")
print("FINAL REPORT")

print(
    f"\nClassifier test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Classification errors analysed: "
    f"{len(set(results_df['error_id'])) if len(results_df) else 0}"
)

print(
    f"Number of discovered concepts: "
    f"{N_CONCEPTS}"
)

print(
    f"Candidate concepts / image: "
    f"up to {TOP_CANDIDATE_CONCEPTS}"
)

print(
    f"CIG interaction lambda: "
    f"{LAMBDA_INTERACTION}"
)

if len(summary_df) > 0:

    print(
        "\nOVERALL RESULTS"
    )

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            "\n"
            f"N={int(row['N'])}\n"
            f"  List margin improvement : "
            f"{row['list_margin_improvement']:.4f}\n"
            f"  CIG margin improvement  : "
            f"{row['cig_margin_improvement']:.4f}\n"
            f"  CIG advantage           : "
            f"{row['cig_advantage']:.4f} "
            f"[{row['cig_advantage_CI_low']:.4f}, "
            f"{row['cig_advantage_CI_high']:.4f}]\n"
            f"  List correction rate    : "
            f"{row['list_correction_rate']:.4f}\n"
            f"  CIG correction rate     : "
            f"{row['cig_correction_rate']:.4f}\n"
            f"  CIG win rate            : "
            f"{row['cig_win_rate']:.4f}\n"
            f"  Selection Jaccard       : "
            f"{row['mean_selection_jaccard']:.4f}\n"
            f"  Wilcoxon p              : "
            f"{row['wilcoxon_p']:.6g}"
        )


if len(strong_df) > 0:

    print(
        "\n\nTOP-QUARTILE STRONG INTERACTION CASES"
    )

    for _, row in (
        strong_df.iterrows()
    ):

        print(
            "\n"
            f"N={int(row['N'])}\n"
            f"  Images                  : "
            f"{int(row['num_images'])}\n"
            f"  List margin improvement : "
            f"{row['list_margin_improvement']:.4f}\n"
            f"  CIG margin improvement  : "
            f"{row['cig_margin_improvement']:.4f}\n"
            f"  CIG advantage           : "
            f"{row['cig_advantage']:.4f} "
            f"[{row['CI_low']:.4f}, "
            f"{row['CI_high']:.4f}]\n"
            f"  CIG win rate            : "
            f"{row['cig_win_rate']:.4f}\n"
            f"  List correction rate    : "
            f"{row['list_correction_rate']:.4f}\n"
            f"  CIG correction rate     : "
            f"{row['cig_correction_rate']:.4f}"
        )


print("\n")


print("\nSaved files:")

for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    print(
        " -",
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )


print("\nDONE.")