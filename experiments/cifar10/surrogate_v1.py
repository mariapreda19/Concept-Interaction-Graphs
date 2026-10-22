import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
NUM_WORKERS = 2
N_IMAGES = 5000
CONCEPT_LAYER = "layer3"
N_CONCEPTS = 96
PATCHES_PER_IMAGE_FOR_CLUSTERING = 12
KMEANS_BATCH_SIZE = 2048
KMEANS_MAX_ITER = 200
MIN_CONCEPT_ACTIVATION = 0.0
N_REPRESENTATIVE_CROPS = 12
AUTO_LABEL_MODE = "blip"
BLIP_MODEL_PATH = "Salesforce/blip-image-captioning-base"
MAX_CAPTION_CONCEPTS = 4
SURROGATE_TEST_SIZE = 0.25
RIDGE_ALPHA = 2.0
MAX_INTERACTION_PAIRS = 220
INTERACTION_PAIR_CANDIDATES = [0, 25, 50, 100, 150, 220]
INTERACTION_VALIDATION_SIZE = 0.20
MIN_PAIR_CORRELATION = 0.05
TOP_CLASS_NODES = 3
TOP_CONCEPT_NODES = 10
TOP_CONCEPT_CLASS_EDGES = 12
TOP_INTERACTION_EDGES = 10
N_EXPLANATIONS = 50
N_RANDOM_BASELINE_DRAWS = 20
N_BOOTSTRAP_SAMPLES = 1000
TOP_COUNTERFACTUAL_CONCEPTS = 8
TOP_COUNTERFACTUAL_INTERACTIONS = 8
MIN_ABS_CONTRIBUTION = 0.01
MIN_ABS_INTERACTION = 0.01
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
CIFAR10_DATASET_ROOT: Optional[Path] = None

WORKING_DIR = (
    Path("/kaggle/working")
    if Path("/kaggle/working").exists()
    else Path(".")
)

CACHE_DIR = WORKING_DIR / "accig_cache"
OUTPUT_DIR = WORKING_DIR / "accig_outputs"
REPRESENTATIVE_DIR = OUTPUT_DIR / "representative_crops"
EXPLANATION_DIR = OUTPUT_DIR / "local_explanations"

for directory in [
    CACHE_DIR,
    OUTPUT_DIR,
    REPRESENTATIVE_DIR,
    EXPLANATION_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

SUBSET_INDICES_PATH = (
    CACHE_DIR / f"subset_{N_IMAGES}_seed_{SEED}.npy"
)
FEATURE_CACHE_PATH = (
    CACHE_DIR
    / f"resnet18_{CONCEPT_LAYER}_{N_IMAGES}_seed_{SEED}_spatial_features.npz"
)
CENTROID_CACHE_PATH = (
    CACHE_DIR
    / f"kmeans_{CONCEPT_LAYER}_{N_CONCEPTS}_seed_{SEED}.npz"
)
CONCEPT_CACHE_PATH = (
    CACHE_DIR
    / f"rich_concept_activations_{CONCEPT_LAYER}_{N_CONCEPTS}_{N_IMAGES}.npz"
)
CONCEPT_LABELS_PATH = OUTPUT_DIR / "concept_labels.json"
SURROGATE_CACHE_PATH = OUTPUT_DIR / "surrogate_parameters.npz"

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()
class IndexedDataset(Dataset):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def find_cifar10_root() -> Path:
    if CIFAR10_DATASET_ROOT is not None:
        root = Path(CIFAR10_DATASET_ROOT)
        expected = root / "cifar-10-batches-py"
        if not expected.is_dir():
            raise FileNotFoundError(
                f"{root} does not contain cifar-10-batches-py/"
            )
        return root

    candidates = sorted(
        path.parent
        for path in KAGGLE_INPUT_ROOT.rglob("cifar-10-batches-py")
        if path.is_dir()
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find cifar-10-batches-py under /kaggle/input. "
            "Attach the extracted CIFAR-10 Python dataset or set "
            "CIFAR10_DATASET_ROOT manually."
        )

    if len(candidates) > 1:
        print("Multiple CIFAR-10 roots found. Using:", candidates[0])

    return candidates[0]


def load_raw_cifar10() -> torchvision.datasets.CIFAR10:
    root = find_cifar10_root()
    return torchvision.datasets.CIFAR10(
        root=str(root),
        train=False,
        download=False,
        transform=None,
    )


def load_or_create_subset_indices(dataset_size: int) -> np.ndarray:
    subset_size = min(N_IMAGES, dataset_size)

    if SUBSET_INDICES_PATH.exists():
        indices = np.load(SUBSET_INDICES_PATH)
        valid = (
            indices.ndim == 1
            and len(indices) == subset_size
            and np.all(indices >= 0)
            and np.all(indices < dataset_size)
            and len(np.unique(indices)) == len(indices)
        )
        if valid:
            print("Loaded subset indices:", SUBSET_INDICES_PATH)
            return indices.astype(np.int64)

    rng = np.random.default_rng(SEED)
    indices = rng.choice(
        dataset_size,
        size=subset_size,
        replace=False,
    ).astype(np.int64)
    np.save(SUBSET_INDICES_PATH, indices)
    print("Saved subset indices:", SUBSET_INDICES_PATH)
    return indices


def load_data(transform) -> Tuple[DataLoader, np.ndarray]:
    root = find_cifar10_root()
    print("Using local CIFAR-10 from:", root)

    dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=False,
        download=False,
        transform=transform,
    )

    indices = load_or_create_subset_indices(len(dataset))
    subset = IndexedDataset(Subset(dataset, indices.tolist()))

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    return loader, indices

def load_model() -> Tuple[nn.Module, object]:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval().to(DEVICE)
    return model, weights.transforms()


def extract_spatial_features(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if FEATURE_CACHE_PATH.exists():
        with np.load(FEATURE_CACHE_PATH) as cached:
            required = {"features", "logits", "labels", "subset_positions"}
            if required.issubset(cached.files):
                print("Loaded feature cache:", FEATURE_CACHE_PATH)
                return (
                    cached["features"].copy(),
                    cached["logits"].copy(),
                    cached["labels"].copy(),
                    cached["subset_positions"].copy(),
                )

    feature_batches: List[np.ndarray] = []
    logits_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    position_batches: List[np.ndarray] = []

    hook_output: Dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        hook_output["value"] = output.detach()

    handle = getattr(model, CONCEPT_LAYER).register_forward_hook(hook)

    try:
        with torch.no_grad():
            for images, labels, subset_positions in tqdm(
                loader,
                desc="Extracting spatial features",
            ):
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)
                features = hook_output["value"]

                feature_batches.append(
                    features.cpu().numpy().astype(np.float32)
                )
                logits_batches.append(
                    logits.cpu().numpy().astype(np.float32)
                )
                label_batches.append(labels.numpy().astype(np.int64))
                position_batches.append(
                    subset_positions.numpy().astype(np.int64)
                )
    finally:
        handle.remove()

    features_array = np.concatenate(feature_batches)
    logits_array = np.concatenate(logits_batches)
    labels_array = np.concatenate(label_batches)
    positions_array = np.concatenate(position_batches)

    np.savez_compressed(
        FEATURE_CACHE_PATH,
        features=features_array,
        logits=logits_array,
        labels=labels_array,
        subset_positions=positions_array,
    )
    print("Saved feature cache:", FEATURE_CACHE_PATH)

    return (
        features_array,
        logits_array,
        labels_array,
        positions_array,
    )

class CIFARLinearProbe(nn.Module):

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_cifar_probe(
    spatial_features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    pooled = spatial_features.mean(axis=(2, 3))
    x_train, x_test, y_train, y_test = train_test_split(
        pooled,
        labels,
        test_size=SURROGATE_TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_all_scaled = scaler.transform(pooled)
    x_test_scaled = scaler.transform(x_test)

    classifier = Ridge(alpha=1.0)
    y_train_onehot = np.eye(len(CIFAR10_CLASSES))[y_train]
    classifier.fit(x_train_scaled, y_train_onehot)

    all_logits = classifier.predict(x_all_scaled)
    test_logits = classifier.predict(x_test_scaled)
    test_pred = test_logits.argmax(axis=1)

    metrics = {
        "probe_test_accuracy": float(
            accuracy_score(y_test, test_pred)
        )
    }

    np.savez_compressed(
        OUTPUT_DIR / "cifar_linear_probe.npz",
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        logits=all_logits,
    )

    return all_logits.astype(np.float32), metrics

def flatten_and_normalize_spatial_features(
    spatial_features: np.ndarray,
) -> np.ndarray:
    if spatial_features.ndim != 4:
        raise ValueError(
            "Expected spatial_features with shape N x C x H x W, "
            f"but received shape {spatial_features.shape}."
        )

    n_images, channels, height, width = spatial_features.shape

    x = spatial_features.transpose(0, 2, 3, 1)
    x = x.reshape(n_images, height * width, channels)

    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.maximum(norms, 1e-8)

    return x.astype(np.float32)


def sample_patch_vectors(
    normalized_spatial: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_images, n_locations, channels = normalized_spatial.shape

    sample_count = min(
        PATCHES_PER_IMAGE_FOR_CLUSTERING,
        n_locations,
    )

    sampled = []
    for image_index in range(n_images):
        locations = rng.choice(
            n_locations,
            size=sample_count,
            replace=False,
        )
        sampled.append(normalized_spatial[image_index, locations])

    return np.concatenate(sampled, axis=0)


def discover_concepts(
    normalized_spatial: np.ndarray,
) -> np.ndarray:
    if CENTROID_CACHE_PATH.exists():
        with np.load(CENTROID_CACHE_PATH) as cached:
            if "centroids" in cached.files:
                centroids = cached["centroids"].copy()
                if centroids.shape[0] == N_CONCEPTS:
                    print("Loaded concept centroids:", CENTROID_CACHE_PATH)
                    return centroids

    sampled_vectors = sample_patch_vectors(normalized_spatial)

    print(
        f"Clustering {len(sampled_vectors)} patch vectors into "
        f"{N_CONCEPTS} concepts..."
    )

    kmeans = MiniBatchKMeans(
        n_clusters=N_CONCEPTS,
        random_state=SEED,
        batch_size=KMEANS_BATCH_SIZE,
        max_iter=KMEANS_MAX_ITER,
        n_init=10,
        reassignment_ratio=0.01,
    )
    kmeans.fit(sampled_vectors)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroids /= (
        np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    )

    np.savez_compressed(
        CENTROID_CACHE_PATH,
        centroids=centroids,
        inertia=np.array([kmeans.inertia_]),
    )
    print("Saved concept centroids:", CENTROID_CACHE_PATH)
    return centroids


def compute_concept_activations(
    normalized_spatial: np.ndarray,
    centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    expected_width = 4 * N_CONCEPTS

    if CONCEPT_CACHE_PATH.exists():
        with np.load(CONCEPT_CACHE_PATH) as cached:
            if {"activations", "locations"}.issubset(cached.files):
                activations = cached["activations"].copy()
                locations = cached["locations"].copy()
                if (
                    activations.shape[1] == expected_width
                    and locations.shape[1] == N_CONCEPTS
                ):
                    print(
                        "Loaded concept activation cache:",
                        CONCEPT_CACHE_PATH,
                    )
                    return activations, locations

    n_images, n_locations, _ = normalized_spatial.shape
    activations = np.zeros(
        (n_images, expected_width),
        dtype=np.float32,
    )
    locations = np.zeros(
        (n_images, N_CONCEPTS),
        dtype=np.int64,
    )

    top_k = max(1, min(5, n_locations))

    for start in tqdm(
        range(0, n_images, BATCH_SIZE),
        desc="Computing rich concept activations",
    ):
        stop = min(start + BATCH_SIZE, n_images)

        similarities = normalized_spatial[start:stop] @ centroids.T

        max_scores = similarities.max(axis=1)
        best_locations = similarities.argmax(axis=1)

        topk_values = np.partition(
            similarities,
            kth=n_locations - top_k,
            axis=1,
        )[:, -top_k:, :]
        topk_mean = topk_values.mean(axis=1)

        positive_mean = np.maximum(similarities, 0.0).mean(axis=1)

        adaptive_threshold = (
            similarities.mean(axis=1, keepdims=True)
            + similarities.std(axis=1, keepdims=True)
        )
        coverage = (
            similarities >= adaptive_threshold
        ).mean(axis=1)

        feature_block = np.concatenate(
            [
                np.maximum(max_scores, MIN_CONCEPT_ACTIVATION),
                np.maximum(topk_mean, MIN_CONCEPT_ACTIVATION),
                positive_mean,
                coverage,
            ],
            axis=1,
        )

        activations[start:stop] = feature_block.astype(np.float32)
        locations[start:stop] = best_locations.astype(np.int64)

    np.savez_compressed(
        CONCEPT_CACHE_PATH,
        activations=activations,
        locations=locations,
    )
    print("Saved rich concept activations:", CONCEPT_CACHE_PATH)

    return activations, locations


def base_concept_scores(
    rich_activations: np.ndarray,
) -> np.ndarray:
    if rich_activations.shape[1] < N_CONCEPTS:
        raise ValueError("Concept feature matrix is too narrow.")
    return rich_activations[:, :N_CONCEPTS]

@dataclass
class RepresentativeCrop:
    concept_index: int
    image_position: int
    dataset_index: int
    spatial_location: int
    score: float
    path: Path


def spatial_location_to_crop(
    image: Image.Image,
    spatial_location: int,
    feature_h: int,
    feature_w: int,
) -> Image.Image:
    width, height = image.size
    row = spatial_location // feature_w
    col = spatial_location % feature_w

    center_x = (col + 0.5) * width / feature_w
    center_y = (row + 0.5) * height / feature_h

    crop_width = max(width / feature_w * 2.5, width * 0.35)
    crop_height = max(height / feature_h * 2.5, height * 0.35)

    left = max(0, int(center_x - crop_width / 2))
    right = min(width, int(center_x + crop_width / 2))
    top = max(0, int(center_y - crop_height / 2))
    bottom = min(height, int(center_y + crop_height / 2))

    return image.crop((left, top, right, bottom))


def save_representative_crops(
    raw_dataset: torchvision.datasets.CIFAR10,
    subset_indices: np.ndarray,
    concept_activations: np.ndarray,
    best_locations: np.ndarray,
    feature_h: int,
    feature_w: int,
) -> Dict[int, List[RepresentativeCrop]]:
    representatives: Dict[int, List[RepresentativeCrop]] = {}

    for concept_index in range(N_CONCEPTS):
        concept_dir = (
            REPRESENTATIVE_DIR / f"concept_{concept_index:03d}"
        )
        concept_dir.mkdir(parents=True, exist_ok=True)

        top_positions = np.argsort(
            concept_activations[:, concept_index]
        )[-N_REPRESENTATIVE_CROPS:][::-1]

        concept_representatives = []

        for rank, image_position in enumerate(top_positions):
            dataset_index = int(subset_indices[image_position])
            image, _ = raw_dataset[dataset_index]
            location = int(
                best_locations[image_position, concept_index]
            )
            score = float(
                concept_activations[image_position, concept_index]
            )

            crop = spatial_location_to_crop(
                image,
                location,
                feature_h,
                feature_w,
            ).resize((224, 224))

            path = concept_dir / f"rank_{rank:02d}.png"
            crop.save(path)

            concept_representatives.append(
                RepresentativeCrop(
                    concept_index=concept_index,
                    image_position=int(image_position),
                    dataset_index=dataset_index,
                    spatial_location=location,
                    score=score,
                    path=path,
                )
            )

        representatives[concept_index] = concept_representatives

    return representatives


STOP_WORDS = {
    "a", "an", "the", "of", "on", "in", "with", "and", "or",
    "at", "to", "is", "are", "this", "that", "image", "picture",
    "photo", "close", "up", "small", "large", "some", "very",
}


def clean_caption_tokens(caption: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", caption.lower())
    return [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]


def label_concepts_with_blip(
    representatives: Mapping[int, Sequence[RepresentativeCrop]],
) -> Dict[int, Dict[str, object]]:
    if AUTO_LABEL_MODE != "blip":
        return {
            index: {
                "label": f"concept_{index:03d}",
                "captions": [],
                "keywords": [],
            }
            for index in range(N_CONCEPTS)
        }

    try:
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
        )
    except ImportError:
        warnings.warn(
            "transformers is unavailable. Falling back to generic names."
        )
        return {
            index: {
                "label": f"concept_{index:03d}",
                "captions": [],
                "keywords": [],
            }
            for index in range(N_CONCEPTS)
        }

    print("Loading BLIP for automatic concept naming...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_PATH
    ).to(DEVICE)
    model.eval()

    concept_information: Dict[int, Dict[str, object]] = {}

    for concept_index in tqdm(
        range(N_CONCEPTS),
        desc="Naming concepts with BLIP",
    ):
        captions = []
        word_counter: Counter = Counter()

        for representative in representatives[concept_index]:
            image = Image.open(representative.path).convert("RGB")
            inputs = processor(
                images=image,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=20,
                )

            caption = processor.decode(
                generated[0],
                skip_special_tokens=True,
            )
            captions.append(caption)
            word_counter.update(clean_caption_tokens(caption))

        keywords = [
            word
            for word, count in word_counter.most_common(
                MAX_CAPTION_CONCEPTS
            )
        ]

        label = (
            "_".join(keywords[:2])
            if keywords
            else f"concept_{concept_index:03d}"
        )

        concept_information[concept_index] = {
            "label": label,
            "captions": captions,
            "keywords": keywords,
        }

    CONCEPT_LABELS_PATH.write_text(
        json.dumps(concept_information, indent=2),
        encoding="utf-8",
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return concept_information


def load_or_create_concept_labels(
    representatives: Mapping[int, Sequence[RepresentativeCrop]],
) -> Dict[int, Dict[str, object]]:
    if CONCEPT_LABELS_PATH.exists():
        data = json.loads(
            CONCEPT_LABELS_PATH.read_text(encoding="utf-8")
        )
        return {int(key): value for key, value in data.items()}

    labels = label_concepts_with_blip(representatives)

    if not CONCEPT_LABELS_PATH.exists():
        CONCEPT_LABELS_PATH.write_text(
            json.dumps(labels, indent=2),
            encoding="utf-8",
        )

    return labels

@dataclass
class SurrogateModel:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    interaction_pairs: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    concept_feature_count: int
    base_concept_count: int


def select_interaction_pairs(
    standardized_concepts: np.ndarray,
    target_logits: np.ndarray,
) -> np.ndarray:
    base = standardized_concepts[:, :N_CONCEPTS]
    candidates = []

    for i in range(N_CONCEPTS):
        for j in range(i + 1, N_CONCEPTS):
            interaction = base[:, i] * base[:, j]

            if interaction.std() < 1e-8:
                continue

            scores = []
            for class_index in range(target_logits.shape[1]):
                target = target_logits[:, class_index]
                if target.std() < 1e-8:
                    continue
                corr = np.corrcoef(interaction, target)[0, 1]
                if np.isfinite(corr):
                    scores.append(abs(float(corr)))

            score = max(scores) if scores else 0.0
            if score >= MIN_PAIR_CORRELATION:
                candidates.append((score, i, j))

    candidates.sort(reverse=True)

    selected = [
        (i, j)
        for _, i, j in candidates[:MAX_INTERACTION_PAIRS]
    ]

    return np.asarray(selected, dtype=np.int64)


def build_surrogate_features(
    standardized_concepts: np.ndarray,
    interaction_pairs: np.ndarray,
) -> np.ndarray:
    base = standardized_concepts[:, :N_CONCEPTS]

    interaction_columns = [
        base[:, i] * base[:, j]
        for i, j in interaction_pairs
    ]

    if interaction_columns:
        interactions = np.column_stack(interaction_columns)
        return np.column_stack(
            [standardized_concepts, interactions]
        ).astype(np.float32)

    return standardized_concepts.astype(np.float32)


def train_surrogate(
    concept_activations: np.ndarray,
    target_logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[
    SurrogateModel,
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=SURROGATE_TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    fit_indices, validation_indices = train_test_split(
        train_indices,
        test_size=INTERACTION_VALIDATION_SIZE,
        random_state=SEED + 1,
        stratify=labels[train_indices],
    )

    scaler = StandardScaler()
    scaler.fit(concept_activations[fit_indices])

    concepts_all = scaler.transform(concept_activations)
    concepts_fit = concepts_all[fit_indices]
    concepts_validation = concepts_all[validation_indices]
    concepts_test = concepts_all[test_indices]

    inner_node_model = Ridge(alpha=RIDGE_ALPHA)
    inner_node_model.fit(
        concepts_fit,
        target_logits[fit_indices],
    )

    inner_fit_node_logits = inner_node_model.predict(concepts_fit)
    validation_node_logits = inner_node_model.predict(
        concepts_validation
    )
    inner_fit_residuals = (
        target_logits[fit_indices] - inner_fit_node_logits
    )

    ranked_pairs = select_interaction_pairs(
        concepts_fit,
        inner_fit_residuals,
    )

    base_fit = concepts_fit[:, :N_CONCEPTS]
    base_validation = concepts_validation[:, :N_CONCEPTS]

    validation_original_classes = target_logits[
        validation_indices
    ].argmax(axis=1)

    selection_rows = []

    for requested_count in INTERACTION_PAIR_CANDIDATES:
        count = min(int(requested_count), len(ranked_pairs))
        pairs = ranked_pairs[:count]

        if count == 0:
            validation_logits = validation_node_logits
        else:
            interaction_fit = np.column_stack(
                [
                    base_fit[:, i] * base_fit[:, j]
                    for i, j in pairs
                ]
            ).astype(np.float32)
            interaction_validation = np.column_stack(
                [
                    base_validation[:, i] * base_validation[:, j]
                    for i, j in pairs
                ]
            ).astype(np.float32)

            residual_model = Ridge(alpha=RIDGE_ALPHA)
            residual_model.fit(
                interaction_fit,
                inner_fit_residuals,
            )
            validation_logits = (
                validation_node_logits
                + residual_model.predict(interaction_validation)
            )

        fidelity = accuracy_score(
            validation_original_classes,
            validation_logits.argmax(axis=1),
        )
        score_r2 = r2_score(
            target_logits[validation_indices],
            validation_logits,
            multioutput="variance_weighted",
        )

        selection_rows.append(
            {
                "n_pairs": count,
                "validation_fidelity": float(fidelity),
                "validation_r2": float(score_r2),
            }
        )

    best_row = max(
        selection_rows,
        key=lambda row: (
            row["validation_fidelity"],
            row["validation_r2"],
            -row["n_pairs"],
        ),
    )
    selected_count = int(best_row["n_pairs"])

    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_DIR / "interaction_budget_selection.csv",
        index=False,
    )

    scaler = StandardScaler()
    scaler.fit(concept_activations[train_indices])
    concepts_all = scaler.transform(concept_activations)
    concepts_train = concepts_all[train_indices]
    concepts_test = concepts_all[test_indices]

    node_model = Ridge(alpha=RIDGE_ALPHA)
    node_model.fit(
        concepts_train,
        target_logits[train_indices],
    )

    node_train_logits = node_model.predict(concepts_train)
    node_test_logits = node_model.predict(concepts_test)
    node_all_logits = node_model.predict(concepts_all)
    train_residuals = (
        target_logits[train_indices] - node_train_logits
    )

    ranked_pairs = select_interaction_pairs(
        concepts_train,
        train_residuals,
    )
    interaction_pairs = ranked_pairs[:selected_count]

    base_train = concepts_train[:, :N_CONCEPTS]
    base_all = concepts_all[:, :N_CONCEPTS]
    base_test = concepts_test[:, :N_CONCEPTS]

    if len(interaction_pairs) > 0:
        interaction_train = np.column_stack(
            [
                base_train[:, i] * base_train[:, j]
                for i, j in interaction_pairs
            ]
        ).astype(np.float32)
        interaction_all = np.column_stack(
            [
                base_all[:, i] * base_all[:, j]
                for i, j in interaction_pairs
            ]
        ).astype(np.float32)
        interaction_test = np.column_stack(
            [
                base_test[:, i] * base_test[:, j]
                for i, j in interaction_pairs
            ]
        ).astype(np.float32)

        residual_model = Ridge(alpha=RIDGE_ALPHA)
        residual_model.fit(interaction_train, train_residuals)

        residual_all = residual_model.predict(interaction_all)
        residual_test = residual_model.predict(interaction_test)
        interaction_coefficients = residual_model.coef_.astype(
            np.float32
        )
        interaction_intercept = np.asarray(
            residual_model.intercept_,
            dtype=np.float32,
        )
    else:
        residual_all = np.zeros_like(node_all_logits)
        residual_test = np.zeros_like(node_test_logits)
        interaction_coefficients = np.zeros(
            (target_logits.shape[1], 0),
            dtype=np.float32,
        )
        interaction_intercept = np.zeros(
            target_logits.shape[1],
            dtype=np.float32,
        )

    final_all_logits = node_all_logits + residual_all
    final_test_logits = node_test_logits + residual_test

    original_test_classes = target_logits[test_indices].argmax(axis=1)
    node_test_classes = node_test_logits.argmax(axis=1)
    final_test_classes = final_test_logits.argmax(axis=1)

    node_r2 = r2_score(
        target_logits[test_indices],
        node_test_logits,
        multioutput="variance_weighted",
    )
    final_r2 = r2_score(
        target_logits[test_indices],
        final_test_logits,
        multioutput="variance_weighted",
    )
    node_fidelity = accuracy_score(
        original_test_classes,
        node_test_classes,
    )
    final_fidelity = accuracy_score(
        original_test_classes,
        final_test_classes,
    )

    metrics = {
        "node_only_logit_r2": float(node_r2),
        "node_only_prediction_fidelity": float(node_fidelity),
        "residual_interaction_logit_r2": float(final_r2),
        "residual_interaction_logit_rmse": float(
            math.sqrt(
                mean_squared_error(
                    target_logits[test_indices],
                    final_test_logits,
                )
            )
        ),
        "residual_interaction_prediction_fidelity": float(
            final_fidelity
        ),
        "interaction_fidelity_gain": float(
            final_fidelity - node_fidelity
        ),
        "interaction_r2_gain": float(final_r2 - node_r2),
        "selected_interaction_pairs": selected_count,
        "interaction_selection_validation_fidelity": float(
            best_row["validation_fidelity"]
        ),
        "interaction_selection_validation_r2": float(
            best_row["validation_r2"]
        ),
        "mean_absolute_train_residual_before_interactions": float(
            np.mean(np.abs(train_residuals))
        ),
        "n_base_concepts": int(N_CONCEPTS),
        "n_concept_features": int(concept_activations.shape[1]),
        "n_interaction_pairs": int(len(interaction_pairs)),
        "n_surrogate_train": int(len(train_indices)),
        "n_surrogate_test": int(len(test_indices)),
    }

    combined_coefficients = np.concatenate(
        [
            node_model.coef_.astype(np.float32),
            interaction_coefficients,
        ],
        axis=1,
    )
    combined_intercept = (
        np.asarray(node_model.intercept_, dtype=np.float32)
        + interaction_intercept
    )

    model = SurrogateModel(
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        interaction_pairs=interaction_pairs,
        coefficients=combined_coefficients,
        intercept=combined_intercept,
        concept_feature_count=concept_activations.shape[1],
        base_concept_count=N_CONCEPTS,
    )

    np.savez_compressed(
        SURROGATE_CACHE_PATH,
        scaler_mean=model.scaler_mean,
        scaler_scale=model.scaler_scale,
        interaction_pairs=model.interaction_pairs,
        coefficients=model.coefficients,
        intercept=model.intercept,
        predicted_logits=final_all_logits,
        node_only_predicted_logits=node_all_logits,
        residual_predicted_logits=residual_all,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    return (
        model,
        metrics,
        final_all_logits.astype(np.float32),
        train_indices.astype(np.int64),
        test_indices.astype(np.int64),
    )


def standardized_concept_vector(
    model: SurrogateModel,
    concept_vector: np.ndarray,
) -> np.ndarray:
    return (
        concept_vector - model.scaler_mean
    ) / (
        model.scaler_scale + 1e-8
    )


def surrogate_feature_vector(
    model: SurrogateModel,
    concept_vector: np.ndarray,
) -> np.ndarray:
    z = standardized_concept_vector(model, concept_vector)
    base = z[: model.base_concept_count]

    interactions = np.asarray(
        [base[i] * base[j] for i, j in model.interaction_pairs],
        dtype=np.float32,
    )
    return np.concatenate([z, interactions])


def surrogate_logits(
    model: SurrogateModel,
    concept_vector: np.ndarray,
) -> np.ndarray:
    features = surrogate_feature_vector(model, concept_vector)
    return model.coefficients @ features + model.intercept


def decompose_surrogate_contributions(
    model: SurrogateModel,
    concept_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    features = surrogate_feature_vector(model, concept_vector)
    node_features = features[: model.concept_feature_count]
    interaction_features = features[model.concept_feature_count :]

    node_coefficients = model.coefficients[
        :, : model.concept_feature_count
    ]
    interaction_coefficients = model.coefficients[
        :, model.concept_feature_count :
    ]

    raw_node_contributions = (
        node_coefficients * node_features[None, :]
    )

    node_contributions = np.zeros(
        (model.coefficients.shape[0], model.base_concept_count),
        dtype=np.float32,
    )

    for block_index in range(4):
        start = block_index * model.base_concept_count
        stop = start + model.base_concept_count
        node_contributions += raw_node_contributions[:, start:stop]

    edge_contributions = (
        interaction_coefficients
        * interaction_features[None, :]
    )

    return node_contributions, edge_contributions

def concept_counterfactuals(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
) -> List[Dict[str, float]]:
    original = surrogate_logits(model, concept_vector)
    baseline = model.scaler_mean

    results = []

    for concept_index in range(model.base_concept_count):
        modified = concept_vector.copy()

        for block_index in range(4):
            feature_index = (
                block_index * model.base_concept_count
                + concept_index
            )
            modified[feature_index] = baseline[feature_index]

        new_logits = surrogate_logits(model, modified)

        new_prediction = int(np.argmax(new_logits))

        results.append(
            {
                "concept_index": int(concept_index),
                "predicted_logit_drop": float(
                    original[predicted_class]
                    - new_logits[predicted_class]
                ),
                "true_logit_gain": float(
                    new_logits[true_class]
                    - original[true_class]
                ),
                "margin_improvement": float(
                    (
                        new_logits[true_class]
                        - new_logits[predicted_class]
                    )
                    - (
                        original[true_class]
                        - original[predicted_class]
                    )
                ),
                "new_predicted_class_index": new_prediction,
                "new_predicted_class": CIFAR10_CLASSES[
                    new_prediction
                ],
                "corrects_to_true_class": bool(
                    new_prediction == true_class
                ),
            }
        )

    results.sort(
        key=lambda item: item["margin_improvement"],
        reverse=True,
    )
    return results[:TOP_COUNTERFACTUAL_CONCEPTS]


def interaction_counterfactuals(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
) -> List[Dict[str, float]]:
    original = surrogate_logits(model, concept_vector)
    baseline = model.scaler_mean

    def neutralize(
        vector: np.ndarray,
        concept_indices: Sequence[int],
    ) -> np.ndarray:
        modified = vector.copy()
        for concept_index in concept_indices:
            for block_index in range(4):
                feature_index = (
                    block_index * model.base_concept_count
                    + concept_index
                )
                modified[feature_index] = baseline[feature_index]
        return modified

    results = []

    for pair_index, (i, j) in enumerate(model.interaction_pairs):
        logits_i = surrogate_logits(
            model,
            neutralize(concept_vector, [int(i)]),
        )
        logits_j = surrogate_logits(
            model,
            neutralize(concept_vector, [int(j)]),
        )
        logits_ij = surrogate_logits(
            model,
            neutralize(concept_vector, [int(i), int(j)]),
        )

        interaction_vector = (
            original - logits_i - logits_j + logits_ij
        )

        predicted_interaction = float(
            interaction_vector[predicted_class]
        )
        true_interaction = float(
            interaction_vector[true_class]
        )

        margin_damage = float(
            predicted_interaction - true_interaction
        )

        results.append(
            {
                "pair_index": int(pair_index),
                "concept_i": int(i),
                "concept_j": int(j),
                "predicted_class_interaction": (
                    predicted_interaction
                ),
                "true_class_interaction": true_interaction,
                "wrong_vs_true_margin_damage": margin_damage,
                "absolute_interaction_strength": float(
                    max(
                        abs(predicted_interaction),
                        abs(true_interaction),
                    )
                ),
            }
        )

    results.sort(
        key=lambda item: item["wrong_vs_true_margin_damage"],
        reverse=True,
    )
    return results[:TOP_COUNTERFACTUAL_INTERACTIONS]


def neutralize_base_concept(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    concept_index: int,
) -> np.ndarray:
    modified = concept_vector.copy()
    for block_index in range(4):
        feature_index = (
            block_index * model.base_concept_count
            + concept_index
        )
        modified[feature_index] = model.scaler_mean[feature_index]
    return modified


def intervention_outcome(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    concept_index: int,
    predicted_class: int,
    true_class: int,
) -> Dict[str, object]:
    original_logits = surrogate_logits(model, concept_vector)
    modified_logits = surrogate_logits(
        model,
        neutralize_base_concept(
            model,
            concept_vector,
            concept_index,
        ),
    )
    original_margin = (
        original_logits[true_class] - original_logits[predicted_class]
    )
    modified_margin = (
        modified_logits[true_class] - modified_logits[predicted_class]
    )
    new_prediction = int(np.argmax(modified_logits))

    return {
        "concept_index": int(concept_index),
        "margin_improvement": float(
            modified_margin - original_margin
        ),
        "corrects_to_true_class": bool(
            new_prediction == true_class
        ),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP_SAMPLES,
) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")

    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.choice(array, size=len(array), replace=True)
        estimates[index] = sample.mean()

    return (
        float(array.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def evaluate_counterfactual_baselines(
    model: SurrogateModel,
    concept_activations: np.ndarray,
    classifier_logits: np.ndarray,
    labels: np.ndarray,
    error_indices: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(SEED + 100)
    rows = []

    raw_node_coefficients = model.coefficients[
        :, :model.concept_feature_count
    ]
    aggregated_coefficients = np.zeros(
        (len(CIFAR10_CLASSES), model.base_concept_count),
        dtype=np.float32,
    )
    for block_index in range(4):
        start = block_index * model.base_concept_count
        stop = start + model.base_concept_count
        aggregated_coefficients += raw_node_coefficients[:, start:stop]

    faithful_error_indices = []

    for image_index in tqdm(
        error_indices,
        desc="Evaluating counterfactual baselines",
    ):
        true_class = int(labels[image_index])
        predicted_class = int(
            np.argmax(classifier_logits[image_index])
        )
        concept_vector = concept_activations[image_index]

        surrogate_prediction = int(
            np.argmax(surrogate_logits(model, concept_vector))
        )
        if surrogate_prediction != predicted_class:
            continue

        faithful_error_indices.append(int(image_index))

        all_counterfactuals = concept_counterfactuals(
            model,
            concept_vector,
            predicted_class,
            true_class,
        )
        proposed_index = int(
            all_counterfactuals[0]["concept_index"]
        )

        node_contributions, _ = decompose_surrogate_contributions(
            model,
            concept_vector,
        )
        margin_damage = (
            node_contributions[predicted_class]
            - node_contributions[true_class]
        )
        local_index = int(np.argmax(margin_damage))

        activation_index = int(
            np.argmax(concept_vector[:model.base_concept_count])
        )

        global_index = int(
            np.argmax(
                aggregated_coefficients[predicted_class]
                - aggregated_coefficients[true_class]
            )
        )

        selected = {
            "proposed_best_cf": proposed_index,
            "local_node_contribution": local_index,
            "highest_activation": activation_index,
            "global_wrong_class_coefficient": global_index,
        }

        for method, concept_index in selected.items():
            outcome = intervention_outcome(
                model,
                concept_vector,
                concept_index,
                predicted_class,
                true_class,
            )
            rows.append(
                {
                    "image_index": int(image_index),
                    "method": method,
                    **outcome,
                }
            )

        random_outcomes = []
        for _ in range(N_RANDOM_BASELINE_DRAWS):
            random_index = int(
                rng.integers(0, model.base_concept_count)
            )
            random_outcomes.append(
                intervention_outcome(
                    model,
                    concept_vector,
                    random_index,
                    predicted_class,
                    true_class,
                )
            )

        rows.append(
            {
                "image_index": int(image_index),
                "method": "random_concept",
                "concept_index": -1,
                "margin_improvement": float(
                    np.mean(
                        [
                            item["margin_improvement"]
                            for item in random_outcomes
                        ]
                    )
                ),
                "corrects_to_true_class": float(
                    np.mean(
                        [
                            item["corrects_to_true_class"]
                            for item in random_outcomes
                        ]
                    )
                ),
            }
        )

    details = pd.DataFrame(rows)
    summary_rows = []
    bootstrap_rng = np.random.default_rng(SEED + 200)

    if not details.empty:
        for method, group in details.groupby("method"):
            margin_mean, margin_low, margin_high = bootstrap_mean_ci(
                group["margin_improvement"].to_numpy(),
                bootstrap_rng,
            )
            correction_mean, correction_low, correction_high = (
                bootstrap_mean_ci(
                    group["corrects_to_true_class"].astype(
                        float
                    ).to_numpy(),
                    bootstrap_rng,
                )
            )
            positive_rate = float(
                (group["margin_improvement"] > 0).mean()
            )

            summary_rows.append(
                {
                    "method": method,
                    "n_faithful_errors": int(len(group)),
                    "mean_margin_improvement": margin_mean,
                    "margin_ci95_low": margin_low,
                    "margin_ci95_high": margin_high,
                    "positive_margin_rate": positive_rate,
                    "correction_rate": correction_mean,
                    "correction_ci95_low": correction_low,
                    "correction_ci95_high": correction_high,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    coverage = {
        "n_all_test_errors": int(len(error_indices)),
        "n_faithful_test_errors": int(len(faithful_error_indices)),
        "faithful_error_coverage": float(
            len(faithful_error_indices) / max(1, len(error_indices))
        ),
    }

    return details, {
        "coverage": coverage,
        "summary": summary_df.to_dict(orient="records"),
    }

def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def concept_name(
    concept_labels: Mapping[int, Mapping[str, object]],
    concept_index: int,
) -> str:
    value = concept_labels.get(concept_index, {})
    return str(value.get("label", f"concept_{concept_index:03d}"))


def build_local_explanation_graph(
    image_index: int,
    true_class: int,
    classifier_logits: np.ndarray,
    concept_vector: np.ndarray,
    model: SurrogateModel,
    concept_labels: Mapping[int, Mapping[str, object]],
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    probabilities = softmax(classifier_logits)
    predicted_class = int(np.argmax(classifier_logits))

    class_indices = list(
        np.argsort(classifier_logits)[-TOP_CLASS_NODES:][::-1]
    )
    if true_class not in class_indices:
        class_indices.append(true_class)

    node_contributions, interaction_contributions = (
        decompose_surrogate_contributions(
            model,
            concept_vector,
        )
    )

    graph = nx.DiGraph(
        image_index=int(image_index),
        predicted_class=CIFAR10_CLASSES[predicted_class],
        true_class=CIFAR10_CLASSES[true_class],
    )

    for class_index in class_indices:
        graph.add_node(
            f"class:{CIFAR10_CLASSES[class_index]}",
            node_type="class",
            class_index=int(class_index),
            label=CIFAR10_CLASSES[class_index],
            probability=float(probabilities[class_index]),
            logit=float(classifier_logits[class_index]),
            is_predicted=bool(class_index == predicted_class),
            is_ground_truth=bool(class_index == true_class),
        )

    concept_importance = np.max(
        np.abs(node_contributions[class_indices]),
        axis=0,
    )
    selected_concepts = list(
        np.argsort(concept_importance)[-TOP_CONCEPT_NODES:][::-1]
    )

    for concept_index in selected_concepts:
        graph.add_node(
            f"concept:{concept_index}",
            node_type="concept",
            concept_index=int(concept_index),
            label=concept_name(concept_labels, concept_index),
            activation=float(concept_vector[concept_index]),
        )

    candidate_class_edges = []
    for class_index in class_indices:
        for concept_index in selected_concepts:
            contribution = float(
                node_contributions[class_index, concept_index]
            )
            if abs(contribution) < MIN_ABS_CONTRIBUTION:
                continue
            candidate_class_edges.append(
                (
                    abs(contribution),
                    concept_index,
                    class_index,
                    contribution,
                )
            )

    candidate_class_edges.sort(reverse=True)

    for _, concept_index, class_index, contribution in (
        candidate_class_edges[:TOP_CONCEPT_CLASS_EDGES]
    ):
        graph.add_edge(
            f"concept:{concept_index}",
            f"class:{CIFAR10_CLASSES[class_index]}",
            edge_type="concept_to_class",
            weight=float(abs(contribution)),
            signed_contribution=float(contribution),
            supports=bool(contribution > 0),
        )

    candidate_interactions = []

    for pair_index, (i, j) in enumerate(model.interaction_pairs):
        if i not in selected_concepts or j not in selected_concepts:
            continue

        contributions = {
            int(class_index): float(
                interaction_contributions[class_index, pair_index]
            )
            for class_index in class_indices
        }

        score = max(abs(value) for value in contributions.values())
        if score < MIN_ABS_INTERACTION:
            continue

        candidate_interactions.append(
            (score, pair_index, int(i), int(j), contributions)
        )

    candidate_interactions.sort(reverse=True)

    for _, pair_index, i, j, contributions in (
        candidate_interactions[:TOP_INTERACTION_EDGES]
    ):
        graph.add_edge(
            f"concept:{i}",
            f"concept:{j}",
            edge_type="concept_interaction",
            pair_index=int(pair_index),
            weight=float(
                max(abs(v) for v in contributions.values())
            ),
            class_contributions=json.dumps(contributions),
        )

    concept_cfs = concept_counterfactuals(
        model,
        concept_vector,
        predicted_class,
        true_class,
    )
    interaction_cfs = interaction_counterfactuals(
        model,
        concept_vector,
        predicted_class,
        true_class,
    )

    explanation = {
        "image_index": int(image_index),
        "true_class": CIFAR10_CLASSES[true_class],
        "predicted_class": CIFAR10_CLASSES[predicted_class],
        "probabilities": {
            CIFAR10_CLASSES[index]: float(probabilities[index])
            for index in class_indices
        },
        "top_concept_counterfactuals": [
            {
                **item,
                "concept_label": concept_name(
                    concept_labels,
                    item["concept_index"],
                ),
            }
            for item in concept_cfs
        ],
        "top_interaction_counterfactuals": [
            {
                **item,
                "concept_i_label": concept_name(
                    concept_labels,
                    item["concept_i"],
                ),
                "concept_j_label": concept_name(
                    concept_labels,
                    item["concept_j"],
                ),
            }
            for item in interaction_cfs
        ],
    }

    return graph, explanation


def plot_local_graph(
    graph: nx.DiGraph,
    path: Path,
) -> None:
    plt.figure(figsize=(14, 9))

    class_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["node_type"] == "class"
    ]
    concept_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["node_type"] == "concept"
    ]

    position = {}

    class_y = np.linspace(0.85, 0.15, max(1, len(class_nodes)))
    for y, node in zip(class_y, class_nodes):
        position[node] = (1.0, y)

    concept_y = np.linspace(0.95, 0.05, max(1, len(concept_nodes)))
    for y, node in zip(concept_y, concept_nodes):
        position[node] = (0.0, y)

    nx.draw_networkx_nodes(
        graph,
        position,
        nodelist=concept_nodes,
        node_shape="o",
        node_size=1500,
    )
    nx.draw_networkx_nodes(
        graph,
        position,
        nodelist=class_nodes,
        node_shape="s",
        node_size=1800,
    )

    concept_class_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data["edge_type"] == "concept_to_class"
    ]
    interaction_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data["edge_type"] == "concept_interaction"
    ]

    if concept_class_edges:
        widths = [
            1.0 + 4.0 * graph[u][v]["weight"]
            for u, v in concept_class_edges
        ]
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=concept_class_edges,
            width=widths,
            arrows=True,
            arrowsize=14,
            alpha=0.75,
        )

    if interaction_edges:
        widths = [
            1.0 + 4.0 * graph[u][v]["weight"]
            for u, v in interaction_edges
        ]
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=interaction_edges,
            width=widths,
            style="dashed",
            arrows=False,
            alpha=0.6,
        )

    labels = {
        node: data["label"]
        for node, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(
        graph,
        position,
        labels=labels,
        font_size=8,
    )

    plt.title(
        f"Local Class–Concept Explanation\n"
        f"true={graph.graph['true_class']} | "
        f"predicted={graph.graph['predicted_class']}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def build_global_graph(
    model: SurrogateModel,
    concept_labels: Mapping[int, Mapping[str, object]],
) -> nx.DiGraph:
    graph = nx.DiGraph(graph_type="global_class_concept")

    for class_index, class_name in enumerate(CIFAR10_CLASSES):
        graph.add_node(
            f"class:{class_name}",
            node_type="class",
            class_index=class_index,
            label=class_name,
        )

    for concept_index in range(N_CONCEPTS):
        graph.add_node(
            f"concept:{concept_index}",
            node_type="concept",
            concept_index=concept_index,
            label=concept_name(
                concept_labels,
                concept_index,
            ),
        )

    raw_node_coefficients = model.coefficients[
        :, : model.concept_feature_count
    ]

    node_coefficients = np.zeros(
        (len(CIFAR10_CLASSES), N_CONCEPTS),
        dtype=np.float32,
    )
    for block_index in range(4):
        start = block_index * N_CONCEPTS
        stop = start + N_CONCEPTS
        node_coefficients += raw_node_coefficients[:, start:stop]

    for concept_index in range(N_CONCEPTS):
        class_order = np.argsort(
            np.abs(node_coefficients[:, concept_index])
        )[-3:][::-1]

        for class_index in class_order:
            coefficient = float(
                node_coefficients[class_index, concept_index]
            )
            graph.add_edge(
                f"concept:{concept_index}",
                f"class:{CIFAR10_CLASSES[class_index]}",
                edge_type="concept_to_class",
                weight=abs(coefficient),
                signed_coefficient=coefficient,
            )

    interaction_coefficients = model.coefficients[
        :, model.concept_feature_count :
    ]

    for pair_index, (i, j) in enumerate(model.interaction_pairs):
        class_effects = interaction_coefficients[:, pair_index]
        max_class = int(np.argmax(np.abs(class_effects)))
        coefficient = float(class_effects[max_class])

        graph.add_edge(
            f"concept:{int(i)}",
            f"concept:{int(j)}",
            edge_type="concept_interaction",
            weight=abs(coefficient),
            strongest_class=CIFAR10_CLASSES[max_class],
            signed_coefficient=coefficient,
        )

    return graph


def concept_statistics(
    concept_activations: np.ndarray,
    labels: np.ndarray,
    concept_labels: Mapping[int, Mapping[str, object]],
) -> pd.DataFrame:
    rows = []

    base_scores = base_concept_scores(concept_activations)

    for concept_index in range(N_CONCEPTS):
        activations = base_scores[:, concept_index]
        class_means = [
            float(activations[labels == class_index].mean())
            for class_index in range(len(CIFAR10_CLASSES))
        ]
        top_class_index = int(np.argmax(class_means))

        rows.append(
            {
                "concept_index": concept_index,
                "concept_label": concept_name(
                    concept_labels,
                    concept_index,
                ),
                "mean_activation": float(activations.mean()),
                "std_activation": float(activations.std()),
                "top_class": CIFAR10_CLASSES[top_class_index],
                "top_class_mean_activation": class_means[
                    top_class_index
                ],
                "class_selectivity": float(
                    np.max(class_means)
                    - np.partition(class_means, -2)[-2]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_run_config() -> None:
    keys = [
        "SEED",
        "DEVICE",
        "BATCH_SIZE",
        "N_IMAGES",
        "CONCEPT_LAYER",
        "N_CONCEPTS",
        "PATCHES_PER_IMAGE_FOR_CLUSTERING",
        "MAX_INTERACTION_PAIRS",
        "N_REPRESENTATIVE_CROPS",
        "AUTO_LABEL_MODE",
        "BLIP_MODEL_PATH",
        "RIDGE_ALPHA",
        "MAX_INTERACTION_PAIRS",
        "INTERACTION_PAIR_CANDIDATES",
        "INTERACTION_VALIDATION_SIZE",
        "N_EXPLANATIONS",
        "N_RANDOM_BASELINE_DRAWS",
        "N_BOOTSTRAP_SAMPLES",
    ]

    config = {key: globals()[key] for key in keys}
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def run_experiment() -> None:
    print("Device:", DEVICE)
    print("Output directory:", OUTPUT_DIR.resolve())

    model, transform = load_model()
    loader, subset_indices = load_data(transform)
    raw_dataset = load_raw_cifar10()

    (
        spatial_features,
        _imagenet_logits,
        labels,
        subset_positions,
    ) = extract_spatial_features(model, loader)

    classifier_logits, probe_metrics = train_cifar_probe(
        spatial_features,
        labels,
    )
    classifier_predictions = classifier_logits.argmax(axis=1)

    normalized_spatial = flatten_and_normalize_spatial_features(
        spatial_features
    )
    print(
        "Normalized spatial feature shape:",
        normalized_spatial.shape,
    )
    centroids = discover_concepts(normalized_spatial)

    (
        concept_activations,
        best_locations,
    ) = compute_concept_activations(
        normalized_spatial,
        centroids,
    )

    feature_h = spatial_features.shape[2]
    feature_w = spatial_features.shape[3]

    representatives = save_representative_crops(
        raw_dataset=raw_dataset,
        subset_indices=subset_indices,
        concept_activations=base_concept_scores(concept_activations),
        best_locations=best_locations,
        feature_h=feature_h,
        feature_w=feature_w,
    )

    concept_labels = load_or_create_concept_labels(
        representatives
    )

    (
        surrogate,
        surrogate_metrics,
        surrogate_logits_all,
        surrogate_train_indices,
        surrogate_test_indices,
    ) = train_surrogate(
        concept_activations,
        classifier_logits,
        labels,
    )

    test_predictions = classifier_predictions[
        surrogate_test_indices
    ]
    test_labels = labels[surrogate_test_indices]

    metrics = {
        **probe_metrics,
        **surrogate_metrics,
        "classifier_accuracy_on_all_samples": float(
            accuracy_score(labels, classifier_predictions)
        ),
        "classifier_accuracy_on_explanation_test_split": float(
            accuracy_score(test_labels, test_predictions)
        ),
        "n_misclassifications_all_samples": int(
            np.sum(classifier_predictions != labels)
        ),
        "n_misclassifications_explanation_test_split": int(
            np.sum(test_predictions != test_labels)
        ),
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    concept_statistics(
        concept_activations,
        labels,
        concept_labels,
    ).to_csv(
        OUTPUT_DIR / "concept_statistics.csv",
        index=False,
    )

    global_graph = build_global_graph(
        surrogate,
        concept_labels,
    )
    nx.write_graphml(
        global_graph,
        OUTPUT_DIR / "global_class_concept_graph.graphml",
    )

    test_error_mask = (
        classifier_predictions[surrogate_test_indices]
        != labels[surrogate_test_indices]
    )
    error_indices = surrogate_test_indices[test_error_mask]

    baseline_details, baseline_summary = (
        evaluate_counterfactual_baselines(
            model=surrogate,
            concept_activations=concept_activations,
            classifier_logits=classifier_logits,
            labels=labels,
            error_indices=error_indices,
        )
    )
    baseline_details.to_csv(
        OUTPUT_DIR / "counterfactual_baseline_details.csv",
        index=False,
    )
    pd.DataFrame(
        baseline_summary["summary"]
    ).to_csv(
        OUTPUT_DIR / "counterfactual_baseline_summary.csv",
        index=False,
    )
    (OUTPUT_DIR / "counterfactual_baseline_summary.json").write_text(
        json.dumps(baseline_summary, indent=2),
        encoding="utf-8",
    )
    metrics.update(baseline_summary["coverage"])

    error_confidences = np.max(
        np.apply_along_axis(
            softmax,
            1,
            classifier_logits[error_indices],
        ),
        axis=1,
    )
    ranked_errors = error_indices[
        np.argsort(error_confidences)[::-1]
    ][:N_EXPLANATIONS]

    explanation_rows = []

    for rank, image_position in enumerate(
        tqdm(ranked_errors, desc="Building local explanations")
    ):
        graph, explanation = build_local_explanation_graph(
            image_index=int(image_position),
            true_class=int(labels[image_position]),
            classifier_logits=classifier_logits[image_position],
            concept_vector=concept_activations[image_position],
            model=surrogate,
            concept_labels=concept_labels,
        )

        prefix = (
            f"error_{rank:03d}_"
            f"true_{CIFAR10_CLASSES[labels[image_position]]}_"
            f"pred_{CIFAR10_CLASSES[classifier_predictions[image_position]]}"
        )

        nx.write_graphml(
            graph,
            EXPLANATION_DIR / f"{prefix}.graphml",
        )
        plot_local_graph(
            graph,
            EXPLANATION_DIR / f"{prefix}.png",
        )
        (EXPLANATION_DIR / f"{prefix}.json").write_text(
            json.dumps(explanation, indent=2),
            encoding="utf-8",
        )

        dataset_index = int(subset_indices[image_position])
        image, _ = raw_dataset[dataset_index]
        image.resize((256, 256)).save(
            EXPLANATION_DIR / f"{prefix}_image.png"
        )

        best_cf = (
            explanation["top_concept_counterfactuals"][0]
            if explanation["top_concept_counterfactuals"]
            else {}
        )

        explanation_rows.append(
            {
                "rank": rank,
                "image_position": int(image_position),
                "dataset_index": dataset_index,
                "true_class": CIFAR10_CLASSES[
                    labels[image_position]
                ],
                "predicted_class": CIFAR10_CLASSES[
                    classifier_predictions[image_position]
                ],
                "confidence": float(
                    softmax(
                        classifier_logits[image_position]
                    ).max()
                ),
                "best_counterfactual_concept": best_cf.get(
                    "concept_label"
                ),
                "best_margin_improvement": best_cf.get(
                    "margin_improvement"
                ),
                "best_counterfactual_corrects": bool(
                    best_cf.get("corrects_to_true_class", False)
                ),
            }
        )

    explanations_df = pd.DataFrame(explanation_rows)
    explanations_df.to_csv(
        OUTPUT_DIR / "misclassification_explanations.csv",
        index=False,
    )

    if len(explanations_df) > 0:
        valid_margin = explanations_df[
            "best_margin_improvement"
        ].dropna()

        counterfactual_summary = {
            "n_explanations": int(len(explanations_df)),
            "mean_best_margin_improvement": float(
                valid_margin.mean()
            ) if len(valid_margin) else float("nan"),
            "median_best_margin_improvement": float(
                valid_margin.median()
            ) if len(valid_margin) else float("nan"),
            "positive_margin_improvement_rate": float(
                (valid_margin > 0).mean()
            ) if len(valid_margin) else float("nan"),
            "top_concept_correction_rate": float(
                explanations_df[
                    "best_counterfactual_corrects"
                ].mean()
            ),
        }
    else:
        counterfactual_summary = {
            "n_explanations": 0,
            "mean_best_margin_improvement": float("nan"),
            "median_best_margin_improvement": float("nan"),
            "positive_margin_improvement_rate": float("nan"),
            "top_concept_correction_rate": float("nan"),
        }

    (OUTPUT_DIR / "counterfactual_summary.json").write_text(
        json.dumps(counterfactual_summary, indent=2),
        encoding="utf-8",
    )

    metrics.update(counterfactual_summary)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    save_run_config()

    print("\nExperiment completed.")
    print(json.dumps(metrics, indent=2))
    print("\nMain outputs:")
    print(" -", OUTPUT_DIR / "metrics.json")
    print(" -", OUTPUT_DIR / "concept_statistics.csv")
    print(" -", OUTPUT_DIR / "global_class_concept_graph.graphml")
    print(" -", OUTPUT_DIR / "misclassification_explanations.csv")
    print(" -", EXPLANATION_DIR)


if __name__ == "__main__":
    run_experiment()