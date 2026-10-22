from __future__ import annotations

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
CNN_INTERVENTION_TOP_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
CNN_INTERVENTION_STRENGTHS = [0.25, 0.50, 0.75, 1.00]
CNN_INTERVENTION_VALIDATION_FRACTION = 0.30
CNN_INTERVENTION_MAX_ERRORS = None
CNN_HYBRID_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]
CNN_RANDOM_ENERGY_MATCHED_DRAWS = 20
TOP_COUNTERFACTUAL_CONCEPTS = 8
TOP_COUNTERFACTUAL_INTERACTIONS = 8
TOP_N_BUDGETS = [1, 2, 3, 5, 8, 10]
GRAPH_VS_LIST_BOOTSTRAP_SAMPLES = 1000
GRAPH_VS_LIST_TIE_TOLERANCE = 1e-8
GRAPH_BEAM_WIDTH = 48
GRAPH_CANDIDATE_POOL = 36
GRAPH_EDGE_CANDIDATE_PAIRS = 80
GRAPH_EVAL_VALIDATION_FRACTION = 0.30
GRAPH_INTERACTION_RICH_QUANTILE = 0.75
GRAPH_MIN_MEAN_MARGIN_ADVANTAGE = 0.01
GRAPH_MIN_WIN_RATE = 0.10
GRAPH_MIN_CORRECTION_RATE_GAIN = 0.01
DIRECT_CNN_GRAPH_BUDGETS = [1, 2, 3, 5, 8, 10]
DIRECT_CNN_PAIR_POOL = 24
DIRECT_CNN_GRAPH_CANDIDATE_POOL = 28
DIRECT_CNN_GRAPH_EDGE_POOL = 120
DIRECT_CNN_GRAPH_BEAM_WIDTH = 64
DIRECT_CNN_GRAPH_EDGE_WEIGHTS = [0.25, 0.50, 1.0, 2.0, 4.0]
DIRECT_CNN_GRAPH_INTERACTION_RICH_QUANTILE = 0.75
DIRECT_CNN_GRAPH_VALIDATION_FRACTION = 0.30
DIRECT_CNN_GRAPH_BOOTSTRAP_SAMPLES = 2000
DIRECT_CNN_GRAPH_TIE_TOLERANCE = 1e-7
DIRECT_CNN_GRAPH_MIN_MEAN_ADVANTAGE = 0.01
DIRECT_CNN_GRAPH_MIN_WIN_RATE = 0.10
DIRECT_CNN_GRAPH_MIN_CORRECTION_GAIN = 0.01
SEMANTIC_ANALYSIS_ENABLED = True
SEMANTIC_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_DESCRIPTION_MAX_CAPTIONS = 6
SEMANTIC_VISUAL_WEIGHT = 0.30
SEMANTIC_ANOMALY_THRESHOLD = 0.50
SEMANTIC_RANDOM_PAIRS = 2000
SEMANTIC_EMBEDDING_BATCH_SIZE = 32
MIN_ABS_CONTRIBUTION = 0.01
MIN_ABS_INTERACTION = 0.01

KAGGLE_INPUT_ROOT = Path("/kaggle/input")

CIFAR10_BATCHES_DIR = Path(
    "/kaggle/input/datasets/pankrzysiu/cifar10-python/"
    "cifar-10-batches-py"
)
CIFAR10_DATASET_ROOT = CIFAR10_BATCHES_DIR.parent

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
SEMANTIC_EMBEDDING_CACHE_PATH = (
    CACHE_DIR
    / f"semantic_embeddings_{N_CONCEPTS}_"
    f"{SEMANTIC_TEXT_MODEL.split('/')[-1]}.npz"
)

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
    expected = Path(CIFAR10_BATCHES_DIR)
    required_files = [
        expected / "batches.meta",
        expected / "data_batch_1",
        expected / "data_batch_2",
        expected / "data_batch_3",
        expected / "data_batch_4",
        expected / "data_batch_5",
        expected / "test_batch",
    ]

    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The requested CIFAR-10 dataset is incomplete or unavailable. "
            f"Expected batches directory: {expected}. "
            f"Missing files: {missing}"
        )

    root = expected.parent
    print("Using exact CIFAR-10 batches directory:", expected)
    return root


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


def continue_resnet_from_layer3(
    model: nn.Module,
    layer3_features: torch.Tensor,
) -> torch.Tensor:
    x = model.layer4(layer3_features)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def apply_concept_projection_removal(
    layer3_features: torch.Tensor,
    centroid: np.ndarray,
    top_fraction: float,
    strength: float,
    target_delta_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if layer3_features.ndim != 4:
        raise ValueError(
            f"Expected B x C x H x W, received {layer3_features.shape}"
        )

    centroid_tensor = torch.as_tensor(
        centroid,
        dtype=layer3_features.dtype,
        device=layer3_features.device,
    )
    centroid_tensor = centroid_tensor / (
        centroid_tensor.norm() + 1e-8
    )

    batch, channels, height, width = layer3_features.shape
    spatial = layer3_features.permute(0, 2, 3, 1).reshape(
        batch, height * width, channels
    )

    norms = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = spatial / norms
    similarity = normalized @ centroid_tensor

    n_locations = similarity.shape[1]
    top_k = max(
        1,
        min(
            n_locations,
            int(math.ceil(top_fraction * n_locations)),
        ),
    )

    top_indices = torch.topk(
        similarity,
        k=top_k,
        dim=1,
        largest=True,
    ).indices

    mask = torch.zeros_like(similarity, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)

    positive_similarity = torch.clamp(similarity, min=0.0)
    removal_scale = (
        positive_similarity * mask.to(similarity.dtype)
    ).unsqueeze(-1)

    removal = (
        strength
        * removal_scale
        * norms
        * centroid_tensor.view(1, 1, -1)
    )

    original_delta_norm = float(
        removal.reshape(batch, -1).norm(dim=1).mean().item()
    )

    if target_delta_norm is not None and original_delta_norm > 1e-12:
        scale = float(target_delta_norm) / original_delta_norm
        removal = removal * scale

    final_delta_norm = float(
        removal.reshape(batch, -1).norm(dim=1).mean().item()
    )

    modified = spatial - removal
    modified = modified.reshape(
        batch, height, width, channels
    ).permute(0, 3, 1, 2).contiguous()

    stats = {
        "top_fraction": float(top_fraction),
        "strength": float(strength),
        "selected_locations": int(top_k),
        "mean_positive_similarity": float(
            positive_similarity[mask].mean().item()
            if mask.any()
            else 0.0
        ),
        "delta_norm_before_matching": original_delta_norm,
        "delta_norm_after_matching": final_delta_norm,
    }

    return modified, stats


class CIFARLinearProbe(nn.Module):

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_cifar_probe(
    spatial_features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[
    np.ndarray,
    Dict[str, float],
    Dict[str, np.ndarray],
]:
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

    probe_bundle = {
        "coef": classifier.coef_.astype(np.float32),
        "intercept": np.asarray(
            classifier.intercept_,
            dtype=np.float32,
        ),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }

    return (
        all_logits.astype(np.float32),
        metrics,
        probe_bundle,
    )


def cifar_probe_logits_from_pooled_features(
    pooled_features: np.ndarray,
    probe_bundle: Mapping[str, np.ndarray],
) -> np.ndarray:
    scaled = (
        pooled_features - probe_bundle["scaler_mean"]
    ) / (
        probe_bundle["scaler_scale"] + 1e-8
    )
    return (
        scaled @ probe_bundle["coef"].T
        + probe_bundle["intercept"]
    ).astype(np.float32)


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
class ConceptRelationData:
    descriptions: List[str]
    text_embeddings: np.ndarray
    text_similarity: np.ndarray
    visual_similarity: np.ndarray
    combined_similarity: np.ndarray


def build_concept_description(
    concept_labels: Mapping[int, Mapping[str, object]],
    concept_index: int,
) -> str:
    information = concept_labels.get(concept_index, {})
    label = str(
        information.get(
            "label",
            f"concept_{concept_index:03d}",
        )
    ).replace("_", " ")

    keywords = [
        str(value).replace("_", " ")
        for value in information.get("keywords", [])
        if str(value).strip()
    ]
    captions = [
        str(value).strip()
        for value in information.get("captions", [])
        if str(value).strip()
    ][:SEMANTIC_DESCRIPTION_MAX_CAPTIONS]

    parts = [f"Visual concept: {label}."]
    if keywords:
        parts.append(
            "Keywords: " + ", ".join(keywords) + "."
        )
    if captions:
        parts.append(
            "Representative image descriptions: "
            + " ".join(captions)
        )

    return " ".join(parts)


def cosine_similarity_matrix(
    vectors: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )
    normalized = vectors / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    return np.clip(similarity, -1.0, 1.0).astype(np.float32)


def load_or_compute_semantic_embeddings(
    descriptions: Sequence[str],
) -> np.ndarray:
    description_array = np.asarray(
        list(descriptions),
        dtype=np.str_,
    )

    if SEMANTIC_EMBEDDING_CACHE_PATH.exists():
        try:
            with np.load(
                SEMANTIC_EMBEDDING_CACHE_PATH,
                allow_pickle=False,
            ) as cached:
                required = {
                    "embeddings",
                    "descriptions",
                    "model_name",
                }
                if required.issubset(cached.files):
                    cached_descriptions = cached[
                        "descriptions"
                    ].astype(str)
                    cached_model = str(
                        cached["model_name"].item()
                    )
                    embeddings = cached["embeddings"]
                    valid = (
                        cached_model == SEMANTIC_TEXT_MODEL
                        and np.array_equal(
                            cached_descriptions,
                            description_array,
                        )
                        and embeddings.shape[0]
                        == len(descriptions)
                    )
                    if valid:
                        print(
                            "Loaded semantic embedding cache:",
                            SEMANTIC_EMBEDDING_CACHE_PATH,
                        )
                        return embeddings.astype(np.float32)
        except Exception as error:
            warnings.warn(
                "Could not read semantic embedding cache: "
                f"{error}"
            )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic analysis requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from error

    print(
        "Loading semantic text encoder:",
        SEMANTIC_TEXT_MODEL,
    )
    encoder = SentenceTransformer(
        SEMANTIC_TEXT_MODEL,
        device=DEVICE,
    )
    embeddings = encoder.encode(
        list(descriptions),
        batch_size=SEMANTIC_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.savez_compressed(
        SEMANTIC_EMBEDDING_CACHE_PATH,
        embeddings=embeddings,
        descriptions=description_array,
        model_name=np.asarray(SEMANTIC_TEXT_MODEL),
    )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def build_concept_relation_data(
    concept_labels: Mapping[int, Mapping[str, object]],
    centroids: np.ndarray,
) -> Optional[ConceptRelationData]:
    if not SEMANTIC_ANALYSIS_ENABLED:
        return None

    descriptions = [
        build_concept_description(
            concept_labels,
            concept_index,
        )
        for concept_index in range(N_CONCEPTS)
    ]

    text_embeddings = (
        load_or_compute_semantic_embeddings(descriptions)
    )
    text_similarity = cosine_similarity_matrix(
        text_embeddings
    )
    visual_similarity = cosine_similarity_matrix(
        centroids
    )

    text_similarity_01 = (
        text_similarity + 1.0
    ) / 2.0
    visual_similarity_01 = (
        visual_similarity + 1.0
    ) / 2.0

    combined_similarity = (
        (1.0 - SEMANTIC_VISUAL_WEIGHT)
        * text_similarity_01
        + SEMANTIC_VISUAL_WEIGHT
        * visual_similarity_01
    ).astype(np.float32)

    np.fill_diagonal(text_similarity_01, 1.0)
    np.fill_diagonal(visual_similarity_01, 1.0)
    np.fill_diagonal(combined_similarity, 1.0)

    return ConceptRelationData(
        descriptions=descriptions,
        text_embeddings=text_embeddings,
        text_similarity=text_similarity_01.astype(
            np.float32
        ),
        visual_similarity=visual_similarity_01.astype(
            np.float32
        ),
        combined_similarity=combined_similarity,
    )


def concept_pair_relation_attributes(
    relation_data: Optional[ConceptRelationData],
    concept_i: int,
    concept_j: int,
    interaction_strength: float,
) -> Dict[str, object]:

    if relation_data is None:
        return {}

    text_similarity = float(
        relation_data.text_similarity[
            concept_i,
            concept_j,
        ]
    )
    visual_similarity = float(
        relation_data.visual_similarity[
            concept_i,
            concept_j,
        ]
    )
    combined_similarity = float(
        relation_data.combined_similarity[
            concept_i,
            concept_j,
        ]
    )

    bounded_strength = float(
        abs(interaction_strength)
        / (1.0 + abs(interaction_strength))
    )
    anomaly_score = float(
        bounded_strength * (1.0 - combined_similarity)
    )
    coherence_score = float(
        bounded_strength * combined_similarity
    )

    if combined_similarity >= 0.70:
        relation_type = "semantically_coherent"
    elif anomaly_score >= SEMANTIC_ANOMALY_THRESHOLD:
        relation_type = "semantic_anomaly"
    else:
        relation_type = "contextual_or_uncertain"

    return {
        "text_similarity": text_similarity,
        "visual_similarity": visual_similarity,
        "combined_similarity": combined_similarity,
        "semantic_coherence_score": coherence_score,
        "semantic_anomaly_score": anomaly_score,
        "semantic_relation_type": relation_type,
    }


def analyze_interaction_semantics(
    model: SurrogateModel,
    concept_labels: Mapping[int, Mapping[str, object]],
    relation_data: Optional[ConceptRelationData],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if relation_data is None:
        empty = pd.DataFrame()
        return empty, {
            "semantic_analysis_enabled": False,
        }

    interaction_coefficients = model.coefficients[
        :, model.concept_feature_count:
    ]

    rows = []
    for pair_index, (raw_i, raw_j) in enumerate(
        model.interaction_pairs
    ):
        i = int(raw_i)
        j = int(raw_j)
        class_effects = interaction_coefficients[
            :,
            pair_index,
        ]
        strongest_class_index = int(
            np.argmax(np.abs(class_effects))
        )
        signed_coefficient = float(
            class_effects[strongest_class_index]
        )
        strength = abs(signed_coefficient)

        attributes = concept_pair_relation_attributes(
            relation_data,
            i,
            j,
            strength,
        )

        rows.append(
            {
                "pair_index": int(pair_index),
                "concept_i": i,
                "concept_i_label": concept_name(
                    concept_labels,
                    i,
                ),
                "concept_j": j,
                "concept_j_label": concept_name(
                    concept_labels,
                    j,
                ),
                "strongest_class": CIFAR10_CLASSES[
                    strongest_class_index
                ],
                "signed_interaction_coefficient": (
                    signed_coefficient
                ),
                "absolute_interaction_strength": strength,
                **attributes,
            }
        )

    edge_df = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED + 700)
    all_pairs = np.asarray(
        [
            (i, j)
            for i in range(N_CONCEPTS)
            for j in range(i + 1, N_CONCEPTS)
        ],
        dtype=np.int64,
    )
    sample_size = min(
        SEMANTIC_RANDOM_PAIRS,
        len(all_pairs),
    )
    sampled_indices = rng.choice(
        len(all_pairs),
        size=sample_size,
        replace=False,
    )
    random_pairs = all_pairs[sampled_indices]
    random_text = np.asarray(
        [
            relation_data.text_similarity[i, j]
            for i, j in random_pairs
        ],
        dtype=np.float64,
    )
    random_visual = np.asarray(
        [
            relation_data.visual_similarity[i, j]
            for i, j in random_pairs
        ],
        dtype=np.float64,
    )
    random_combined = np.asarray(
        [
            relation_data.combined_similarity[i, j]
            for i, j in random_pairs
        ],
        dtype=np.float64,
    )

    if len(edge_df) > 1:
        spearman_text = float(
            edge_df[
                "absolute_interaction_strength"
            ].rank().corr(
                edge_df["text_similarity"].rank()
            )
        )
        spearman_visual = float(
            edge_df[
                "absolute_interaction_strength"
            ].rank().corr(
                edge_df["visual_similarity"].rank()
            )
        )
        spearman_combined = float(
            edge_df[
                "absolute_interaction_strength"
            ].rank().corr(
                edge_df["combined_similarity"].rank()
            )
        )
    else:
        spearman_text = float("nan")
        spearman_visual = float("nan")
        spearman_combined = float("nan")

    summary = {
        "semantic_analysis_enabled": True,
        "semantic_text_model": SEMANTIC_TEXT_MODEL,
        "n_interaction_edges": int(len(edge_df)),
        "n_random_pairs": int(sample_size),
        "mean_edge_text_similarity": float(
            edge_df["text_similarity"].mean()
        ) if len(edge_df) else float("nan"),
        "mean_random_text_similarity": float(
            random_text.mean()
        ),
        "mean_edge_visual_similarity": float(
            edge_df["visual_similarity"].mean()
        ) if len(edge_df) else float("nan"),
        "mean_random_visual_similarity": float(
            random_visual.mean()
        ),
        "mean_edge_combined_similarity": float(
            edge_df["combined_similarity"].mean()
        ) if len(edge_df) else float("nan"),
        "mean_random_combined_similarity": float(
            random_combined.mean()
        ),
        "semantic_similarity_gain_over_random": float(
            edge_df["combined_similarity"].mean()
            - random_combined.mean()
        ) if len(edge_df) else float("nan"),
        "spearman_interaction_vs_text_similarity": (
            spearman_text
        ),
        "spearman_interaction_vs_visual_similarity": (
            spearman_visual
        ),
        "spearman_interaction_vs_combined_similarity": (
            spearman_combined
        ),
        "n_semantically_coherent_edges": int(
            (
                edge_df["semantic_relation_type"]
                == "semantically_coherent"
            ).sum()
        ) if len(edge_df) else 0,
        "n_semantic_anomaly_edges": int(
            (
                edge_df["semantic_relation_type"]
                == "semantic_anomaly"
            ).sum()
        ) if len(edge_df) else 0,
        "n_contextual_or_uncertain_edges": int(
            (
                edge_df["semantic_relation_type"]
                == "contextual_or_uncertain"
            ).sum()
        ) if len(edge_df) else 0,
    }

    return edge_df, summary


def plot_semantic_interaction_analysis(
    edge_df: pd.DataFrame,
    path: Path,
) -> None:
    if len(edge_df) == 0:
        return

    plt.figure(figsize=(9, 6))
    plt.scatter(
        edge_df["combined_similarity"],
        edge_df["absolute_interaction_strength"],
        s=50,
        alpha=0.75,
    )

    most_anomalous = edge_df.nlargest(
        min(8, len(edge_df)),
        "semantic_anomaly_score",
    )
    for _, row in most_anomalous.iterrows():
        plt.annotate(
            (
                f"{row['concept_i_label']}--"
                f"{row['concept_j_label']}"
            ),
            (
                row["combined_similarity"],
                row["absolute_interaction_strength"],
            ),
            fontsize=7,
        )

    plt.xlabel("Combined semantic/visual similarity")
    plt.ylabel("Absolute interaction strength")
    plt.title(
        "Concept interaction strength vs. relation similarity"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()



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



def neutralize_concept_set(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    concept_indices: Sequence[int],
) -> np.ndarray:
    modified = concept_vector.copy()
    for raw_index in concept_indices:
        concept_index = int(raw_index)
        for block_index in range(4):
            feature_index = (
                block_index * model.base_concept_count
                + concept_index
            )
            modified[feature_index] = model.scaler_mean[feature_index]
    return modified


def concept_set_intervention_outcome(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    concept_indices: Sequence[int],
    predicted_class: int,
    true_class: int,
) -> Dict[str, object]:
    original_logits = surrogate_logits(model, concept_vector)
    modified_vector = neutralize_concept_set(
        model,
        concept_vector,
        concept_indices,
    )
    modified_logits = surrogate_logits(model, modified_vector)

    original_margin = float(
        original_logits[true_class] - original_logits[predicted_class]
    )
    modified_margin = float(
        modified_logits[true_class] - modified_logits[predicted_class]
    )
    new_prediction = int(np.argmax(modified_logits))

    return {
        "selected_concepts": [int(value) for value in concept_indices],
        "n_selected": int(len(concept_indices)),
        "original_margin": original_margin,
        "modified_margin": modified_margin,
        "margin_improvement": float(modified_margin - original_margin),
        "corrects_to_true_class": bool(new_prediction == true_class),
        "new_predicted_class_index": new_prediction,
        "new_predicted_class": CIFAR10_CLASSES[new_prediction],
    }


def rank_concepts_as_independent_list(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
) -> List[int]:
    scored = []
    for concept_index in range(model.base_concept_count):
        outcome = concept_set_intervention_outcome(
            model=model,
            concept_vector=concept_vector,
            concept_indices=[concept_index],
            predicted_class=predicted_class,
            true_class=true_class,
        )
        scored.append((float(outcome["margin_improvement"]), concept_index))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [int(concept_index) for _, concept_index in scored]


def local_graph_edge_scores(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
) -> List[Tuple[float, int, int]]:
    _, edge_contributions = decompose_surrogate_contributions(
        model,
        concept_vector,
    )
    if edge_contributions.shape[1] == 0:
        return []

    damage = (
        edge_contributions[predicted_class]
        - edge_contributions[true_class]
    )
    rows = []
    for pair_index, (raw_i, raw_j) in enumerate(model.interaction_pairs):
        rows.append(
            (
                float(damage[pair_index]),
                int(raw_i),
                int(raw_j),
            )
        )
    rows.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)
    return rows


def local_interaction_richness_score(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
) -> float:
    node_contributions, edge_contributions = decompose_surrogate_contributions(
        model,
        concept_vector,
    )
    node_damage = (
        node_contributions[predicted_class]
        - node_contributions[true_class]
    )
    edge_damage = (
        edge_contributions[predicted_class]
        - edge_contributions[true_class]
    )
    positive_node = float(np.maximum(node_damage, 0.0).sum())
    positive_edge = float(np.maximum(edge_damage, 0.0).sum())
    return positive_edge / (positive_node + positive_edge + 1e-8)


def graph_candidate_concepts(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
    independent_ranking: Sequence[int],
) -> List[int]:
    candidates = list(independent_ranking[:GRAPH_CANDIDATE_POOL])
    seen = set(candidates)

    for score, i, j in local_graph_edge_scores(
        model,
        concept_vector,
        predicted_class,
        true_class,
    )[:GRAPH_EDGE_CANDIDATE_PAIRS]:
        if score <= 0:
            continue
        for value in (i, j):
            if value not in seen:
                candidates.append(value)
                seen.add(value)

    return candidates


def select_concept_sets_with_graph_beam_search(
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
    budgets: Sequence[int],
    independent_ranking: Sequence[int],
) -> Dict[int, List[int]]:
    requested = sorted(set(int(v) for v in budgets if int(v) > 0))
    max_budget = min(max(requested), model.base_concept_count)
    pool = graph_candidate_concepts(
        model,
        concept_vector,
        predicted_class,
        true_class,
        independent_ranking,
    )

    for concept_index in independent_ranking:
        if concept_index not in pool:
            pool.append(int(concept_index))
        if len(pool) >= max(GRAPH_CANDIDATE_POOL, max_budget):
            break

    cache: Dict[Tuple[int, ...], float] = {}

    def score(concept_set: Tuple[int, ...]) -> float:
        key = tuple(sorted(concept_set))
        if key not in cache:
            cache[key] = float(
                concept_set_intervention_outcome(
                    model=model,
                    concept_vector=concept_vector,
                    concept_indices=key,
                    predicted_class=predicted_class,
                    true_class=true_class,
                )["margin_improvement"]
            )
        return cache[key]

    beam: List[Tuple[int, ...]] = [tuple()]
    selected_by_budget: Dict[int, List[int]] = {}

    for depth in range(1, max_budget + 1):
        expanded = set()
        for current in beam:
            current_set = set(current)
            for concept_index in pool:
                if concept_index in current_set:
                    continue
                expanded.add(tuple(sorted((*current, int(concept_index)))))

        expanded.add(tuple(sorted(independent_ranking[:depth])))

        ranked = sorted(
            expanded,
            key=lambda values: (score(values), tuple(-v for v in values)),
            reverse=True,
        )
        beam = ranked[:GRAPH_BEAM_WIDTH]

        if depth in requested:
            best = max(beam, key=score)
            selected_by_budget[depth] = list(best)

    return selected_by_budget


def paired_bootstrap_difference_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> Tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[bootstrap_index] = sample.mean()

    return (
        float(values.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def summarize_graph_vs_list_scope(
    details: pd.DataFrame,
    budgets: Sequence[int],
    scope_name: str,
    bootstrap_rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    paired_rows = []

    if details.empty:
        return pd.DataFrame(), pd.DataFrame()

    for (budget, method), group in details.groupby(
        ["budget_n", "method"], sort=True
    ):
        margin_mean, margin_low, margin_high = bootstrap_mean_ci(
            group["margin_improvement"].to_numpy(), bootstrap_rng
        )
        correction_mean, correction_low, correction_high = bootstrap_mean_ci(
            group["corrects_to_true_class"].astype(float).to_numpy(),
            bootstrap_rng,
        )
        summary_rows.append(
            {
                "scope": scope_name,
                "budget_n": int(budget),
                "method": method,
                "n_faithful_errors": int(len(group)),
                "mean_margin_improvement": margin_mean,
                "margin_ci95_low": margin_low,
                "margin_ci95_high": margin_high,
                "positive_margin_rate": float(
                    (group["margin_improvement"] > 0).mean()
                ),
                "correction_rate": correction_mean,
                "correction_ci95_low": correction_low,
                "correction_ci95_high": correction_high,
            }
        )

    pivot = details.pivot_table(
        index=["image_index", "budget_n"],
        columns="method",
        values=["margin_improvement", "corrects_to_true_class"],
        aggfunc="first",
    )

    for budget in budgets:
        try:
            budget_pivot = pivot.xs(budget, level="budget_n")
            graph_margin = budget_pivot[
                ("margin_improvement", "graph_beam_search")
            ].to_numpy(dtype=np.float64)
            list_margin = budget_pivot[
                ("margin_improvement", "topn_independent_list")
            ].to_numpy(dtype=np.float64)
            graph_correct = budget_pivot[
                ("corrects_to_true_class", "graph_beam_search")
            ].to_numpy(dtype=np.float64)
            list_correct = budget_pivot[
                ("corrects_to_true_class", "topn_independent_list")
            ].to_numpy(dtype=np.float64)
        except KeyError:
            continue

        difference = graph_margin - list_margin
        mean_diff, low_diff, high_diff = paired_bootstrap_difference_ci(
            difference,
            bootstrap_rng,
            GRAPH_VS_LIST_BOOTSTRAP_SAMPLES,
        )
        tolerance = GRAPH_VS_LIST_TIE_TOLERANCE
        win_rate = float((difference > tolerance).mean())
        correction_gain = float(graph_correct.mean() - list_correct.mean())
        practical = bool(
            low_diff > 0
            and mean_diff >= GRAPH_MIN_MEAN_MARGIN_ADVANTAGE
            and win_rate >= GRAPH_MIN_WIN_RATE
            and correction_gain >= GRAPH_MIN_CORRECTION_RATE_GAIN
        )

        paired_rows.append(
            {
                "scope": scope_name,
                "budget_n": int(budget),
                "n_paired_errors": int(len(difference)),
                "mean_graph_minus_list_margin": mean_diff,
                "median_graph_minus_list_margin": float(np.median(difference)),
                "paired_ci95_low": low_diff,
                "paired_ci95_high": high_diff,
                "graph_win_rate": win_rate,
                "tie_rate": float((np.abs(difference) <= tolerance).mean()),
                "list_win_rate": float((difference < -tolerance).mean()),
                "graph_correction_rate": float(graph_correct.mean()),
                "list_correction_rate": float(list_correct.mean()),
                "correction_rate_difference": correction_gain,
                "graph_strictly_better_ci": bool(low_diff > 0),
                "graph_practically_better": practical,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def evaluate_graph_vs_topn_list(
    model: SurrogateModel,
    concept_activations: np.ndarray,
    classifier_logits: np.ndarray,
    labels: np.ndarray,
    error_indices: np.ndarray,
    concept_labels: Mapping[int, Mapping[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    budgets = sorted(
        {
            int(value)
            for value in TOP_N_BUDGETS
            if 1 <= int(value) <= model.base_concept_count
        }
    )
    if not budgets:
        raise ValueError("TOP_N_BUDGETS contains no valid budget.")

    faithful = []
    for image_index in error_indices:
        concept_vector = concept_activations[image_index]
        predicted_class = int(np.argmax(classifier_logits[image_index]))
        if int(np.argmax(surrogate_logits(model, concept_vector))) == predicted_class:
            faithful.append(int(image_index))

    faithful = np.asarray(faithful, dtype=np.int64)
    if len(faithful) < 4:
        raise RuntimeError("Too few faithful errors for graph evaluation.")

    stratify_labels = labels[faithful]
    try:
        validation_indices, test_indices = train_test_split(
            faithful,
            test_size=1.0 - GRAPH_EVAL_VALIDATION_FRACTION,
            random_state=SEED + 811,
            stratify=stratify_labels,
        )
    except ValueError:
        validation_indices, test_indices = train_test_split(
            faithful,
            test_size=1.0 - GRAPH_EVAL_VALIDATION_FRACTION,
            random_state=SEED + 811,
            stratify=None,
        )

    richness_validation = []
    for image_index in validation_indices:
        predicted_class = int(np.argmax(classifier_logits[image_index]))
        richness_validation.append(
            local_interaction_richness_score(
                model,
                concept_activations[image_index],
                predicted_class,
                int(labels[image_index]),
            )
        )
    richness_threshold = float(
        np.quantile(richness_validation, GRAPH_INTERACTION_RICH_QUANTILE)
    )

    detail_rows = []
    for image_index in tqdm(
        test_indices,
        desc="Graph beam search vs top-N list on held-out errors",
    ):
        true_class = int(labels[image_index])
        predicted_class = int(np.argmax(classifier_logits[image_index]))
        concept_vector = concept_activations[image_index]
        richness = local_interaction_richness_score(
            model,
            concept_vector,
            predicted_class,
            true_class,
        )

        independent_ranking = rank_concepts_as_independent_list(
            model,
            concept_vector,
            predicted_class,
            true_class,
        )
        graph_sets = select_concept_sets_with_graph_beam_search(
            model,
            concept_vector,
            predicted_class,
            true_class,
            budgets,
            independent_ranking,
        )

        for budget in budgets:
            selections = {
                "topn_independent_list": independent_ranking[:budget],
                "graph_beam_search": graph_sets[budget],
            }
            for method, selected in selections.items():
                outcome = concept_set_intervention_outcome(
                    model,
                    concept_vector,
                    selected,
                    predicted_class,
                    true_class,
                )
                detail_rows.append(
                    {
                        "image_index": int(image_index),
                        "true_class": CIFAR10_CLASSES[true_class],
                        "predicted_class": CIFAR10_CLASSES[predicted_class],
                        "interaction_richness": richness,
                        "is_interaction_rich": bool(
                            richness >= richness_threshold
                        ),
                        "budget_n": int(budget),
                        "method": method,
                        "selected_concept_indices": json.dumps(selected),
                        "selected_concept_labels": json.dumps(
                            [concept_name(concept_labels, idx) for idx in selected]
                        ),
                        **{
                            key: value
                            for key, value in outcome.items()
                            if key not in {"selected_concepts", "n_selected"}
                        },
                    }
                )

    details = pd.DataFrame(detail_rows)
    bootstrap_rng = np.random.default_rng(SEED + 812)

    all_summary, all_paired = summarize_graph_vs_list_scope(
        details,
        budgets,
        "all_heldout_faithful_errors",
        bootstrap_rng,
    )
    rich_details = details[details["is_interaction_rich"]].copy()
    rich_summary, rich_paired = summarize_graph_vs_list_scope(
        rich_details,
        budgets,
        "interaction_rich_heldout_errors",
        bootstrap_rng,
    )

    summary = pd.concat([all_summary, rich_summary], ignore_index=True)
    paired = pd.concat([all_paired, rich_paired], ignore_index=True)

    informative = paired[paired["budget_n"] > 1]
    practical_rows = informative[informative["graph_practically_better"]]
    validated = practical_rows["budget_n"].astype(int).tolist()

    if not informative.empty:
        best_row = informative.loc[
            informative["mean_graph_minus_list_margin"].idxmax()
        ]
        best_budget = int(best_row["budget_n"])
        best_scope = str(best_row["scope"])
        best_advantage = float(best_row["mean_graph_minus_list_margin"])
    else:
        best_budget = None
        best_scope = None
        best_advantage = float("nan")

    metadata = {
        "comparison_question": (
            "Does interaction-aware set search outperform an equally sized "
            "list of independently ranked concepts?"
        ),
        "list_method": "topn_independent_list",
        "graph_method": "graph_beam_search",
        "budgets": budgets,
        "beam_width": GRAPH_BEAM_WIDTH,
        "candidate_pool": GRAPH_CANDIDATE_POOL,
        "n_test_errors": int(len(error_indices)),
        "n_faithful_errors_before_graph_split": int(len(faithful)),
        "n_graph_validation_errors": int(len(validation_indices)),
        "n_graph_test_errors": int(len(test_indices)),
        "interaction_richness_quantile": GRAPH_INTERACTION_RICH_QUANTILE,
        "interaction_richness_threshold_from_validation": richness_threshold,
        "practical_thresholds": {
            "minimum_mean_margin_advantage": GRAPH_MIN_MEAN_MARGIN_ADVANTAGE,
            "minimum_graph_win_rate": GRAPH_MIN_WIN_RATE,
            "minimum_correction_rate_gain": GRAPH_MIN_CORRECTION_RATE_GAIN,
            "paired_ci95_lower_must_be_positive": True,
        },
        "validated_budgets_practical": validated,
        "graph_motivation_validated": bool(validated),
        "best_budget_by_mean_margin_advantage": best_budget,
        "best_scope_by_mean_margin_advantage": best_scope,
        "best_mean_graph_minus_list_margin": best_advantage,
        "interpretation": (
            "The graph is declared useful only when it is statistically and "
            "practically better on untouched graph-test errors. The "
            "interaction-rich scope is defined by a threshold learned only "
            "from graph-validation errors."
        ),
        "paired_comparison": paired.to_dict(orient="records"),
    }
    return details, summary, metadata

def plot_graph_vs_topn_summary(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    if summary.empty:
        return

    plt.figure(figsize=(9, 6))
    for (scope, method), group in summary.groupby(["scope", "method"]):
        ordered = group.sort_values("budget_n")
        label = f"{method} | {scope}"
        linestyle = "--" if scope.startswith("interaction_rich") else "-"
        plt.plot(
            ordered["budget_n"],
            ordered["mean_margin_improvement"],
            marker="o",
            linestyle=linestyle,
            label=label,
        )

    plt.xlabel("Number of neutralized concepts (same budget)")
    plt.ylabel("Mean true-vs-wrong margin improvement")
    plt.title("Graph beam search vs. independent top-N concept list")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

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


def select_concept_for_method(
    method: str,
    model: SurrogateModel,
    concept_vector: np.ndarray,
    predicted_class: int,
    true_class: int,
    aggregated_coefficients: np.ndarray,
    rng: np.random.Generator,
    hybrid_alpha: float = 0.5,
) -> int:
    node_contributions, _ = decompose_surrogate_contributions(
        model,
        concept_vector,
    )

    local_score = (
        node_contributions[predicted_class]
        - node_contributions[true_class]
    )

    global_score = (
        aggregated_coefficients[predicted_class]
        - aggregated_coefficients[true_class]
    )

    if method == "proposed_best_cf":
        return int(
            concept_counterfactuals(
                model,
                concept_vector,
                predicted_class,
                true_class,
            )[0]["concept_index"]
        )

    if method == "local_node_contribution":
        return int(np.argmax(local_score))

    if method == "highest_activation":
        return int(
            np.argmax(
                concept_vector[: model.base_concept_count]
            )
        )

    if method == "global_wrong_class_coefficient":
        return int(np.argmax(global_score))

    if method == "hybrid_global_local":
        def normalize(values: np.ndarray) -> np.ndarray:
            values = values.astype(np.float64)
            std = values.std()
            if std < 1e-8:
                return values - values.mean()
            return (values - values.mean()) / std

        score = (
            hybrid_alpha * normalize(local_score)
            + (1.0 - hybrid_alpha) * normalize(global_score)
        )
        return int(np.argmax(score))

    if method == "random_concept":
        return int(
            rng.integers(0, model.base_concept_count)
        )

    raise ValueError(f"Unknown method: {method}")



def apply_concept_set_projection_removal(
    layer3_features: torch.Tensor,
    concept_indices: Sequence[int],
    centroids: np.ndarray,
    top_fraction: float,
    strength: float,
) -> torch.Tensor:
    if layer3_features.ndim != 4:
        raise ValueError("Expected B x C x H x W layer3 features.")

    unique_indices = sorted(set(int(i) for i in concept_indices))
    if not unique_indices:
        return layer3_features.clone()

    batch, channels, height, width = layer3_features.shape
    spatial = layer3_features.permute(0, 2, 3, 1).reshape(
        batch, height * width, channels
    )
    norms = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = spatial / norms
    total_removal = torch.zeros_like(spatial)

    n_locations = height * width
    top_k = max(1, min(n_locations, int(math.ceil(top_fraction * n_locations))))

    for concept_index in unique_indices:
        centroid = torch.as_tensor(
            centroids[concept_index],
            dtype=layer3_features.dtype,
            device=layer3_features.device,
        )
        centroid = centroid / (centroid.norm() + 1e-8)
        similarity = normalized @ centroid
        top_indices = torch.topk(
            similarity, k=top_k, dim=1, largest=True
        ).indices
        mask = torch.zeros_like(similarity, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)
        positive = torch.clamp(similarity, min=0.0)
        scale = (positive * mask.to(similarity.dtype)).unsqueeze(-1)
        total_removal += (
            strength * scale * norms * centroid.view(1, 1, -1)
        )

    removal_norm = total_removal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    max_removal_norm = 0.95 * norms
    total_removal = total_removal * torch.minimum(
        torch.ones_like(removal_norm), max_removal_norm / removal_norm
    )

    modified = spatial - total_removal
    return modified.reshape(batch, height, width, channels).permute(
        0, 3, 1, 2
    ).contiguous()


def evaluate_direct_cnn_set(
    layer3_tensor,
    concept_indices,
    centroids,
    probe_bundle,
    original_logits,
    predicted_class,
    true_class,
    top_fraction,
    strength,
):
    modified = apply_concept_set_projection_removal(
        layer3_tensor,
        concept_indices,
        centroids,
        top_fraction,
        strength,
    )
    with torch.no_grad():
        pooled = modified.mean(dim=(2, 3)).cpu().numpy()
    new_logits = cifar_probe_logits_from_pooled_features(
        pooled, probe_bundle
    )[0]

    old_margin = float(
        original_logits[true_class] - original_logits[predicted_class]
    )
    new_margin = float(new_logits[true_class] - new_logits[predicted_class])
    margin_improvement = new_margin - old_margin
    headroom = max(1e-8, -old_margin)

    return {
        "margin_improvement": float(margin_improvement),
        "relative_margin_recovery": float(margin_improvement / headroom),
        "old_margin": old_margin,
        "new_margin": new_margin,
        "corrects_to_true_class": bool(np.argmax(new_logits) == true_class),
        "new_prediction": int(np.argmax(new_logits)),
    }


def direct_cnn_nodes_and_edges(
    layer3_tensor,
    centroids,
    probe_bundle,
    original_logits,
    predicted_class,
    true_class,
    top_fraction,
    strength,
):
    cache: Dict[Tuple[int, ...], Dict[str, object]] = {}

    def ev(indices):
        key = tuple(sorted(set(int(i) for i in indices)))
        if key not in cache:
            cache[key] = evaluate_direct_cnn_set(
                layer3_tensor,
                key,
                centroids,
                probe_bundle,
                original_logits,
                predicted_class,
                true_class,
                top_fraction,
                strength,
            )
        return cache[key]

    node_scores = np.asarray(
        [ev([i])["margin_improvement"] for i in range(len(centroids))],
        dtype=np.float32,
    )
    independent = np.argsort(node_scores)[::-1]
    pool = independent[: min(DIRECT_CNN_PAIR_POOL, len(centroids))]

    edge_scores: Dict[Tuple[int, int], float] = {}
    for pos, raw_i in enumerate(pool):
        i = int(raw_i)
        for raw_j in pool[pos + 1 :]:
            j = int(raw_j)
            pair_gain = float(ev([i, j])["margin_improvement"])
            edge_scores[(min(i, j), max(i, j))] = float(
                pair_gain - node_scores[i] - node_scores[j]
            )

    positive_edges = sorted(
        (max(0.0, float(v)) for v in edge_scores.values()), reverse=True
    )
    top_positive = positive_edges[: min(10, len(positive_edges))]
    richness = float(sum(top_positive))
    node_scale = float(
        np.sum(np.maximum(node_scores[pool[: min(10, len(pool))]], 0.0))
    )
    normalized_richness = richness / max(node_scale, 1e-8)

    return node_scores, edge_scores, cache, normalized_richness


def graph_objective(
    selected: Sequence[int],
    node_scores: np.ndarray,
    edge_scores: Mapping[Tuple[int, int], float],
    edge_weight: float,
) -> float:
    chosen = tuple(sorted(set(int(i) for i in selected)))
    score = float(sum(float(node_scores[i]) for i in chosen))
    for pos, i in enumerate(chosen):
        for j in chosen[pos + 1 :]:
            score += edge_weight * float(
                edge_scores.get((min(i, j), max(i, j)), 0.0)
            )
    return score


def select_direct_graph_sets_beam(
    node_scores: np.ndarray,
    edge_scores: Mapping[Tuple[int, int], float],
    budgets: Sequence[int],
    edge_weight: float,
) -> Tuple[List[int], Dict[int, List[int]]]:
    independent = [int(v) for v in np.argsort(node_scores)[::-1]]
    max_budget = max(int(v) for v in budgets)

    candidate_set = set(independent[:DIRECT_CNN_GRAPH_CANDIDATE_POOL])
    strongest_edges = sorted(
        edge_scores.items(), key=lambda item: abs(float(item[1])), reverse=True
    )[:DIRECT_CNN_GRAPH_EDGE_POOL]
    for (i, j), _ in strongest_edges:
        candidate_set.add(int(i))
        candidate_set.add(int(j))

    candidates = sorted(
        candidate_set, key=lambda i: float(node_scores[i]), reverse=True
    )[: max(DIRECT_CNN_GRAPH_CANDIDATE_POOL, max_budget)]

    requested = sorted(set(int(v) for v in budgets))
    beam: List[Tuple[int, ...]] = [tuple()]
    by_budget: Dict[int, List[int]] = {}

    for depth in range(1, max_budget + 1):
        expanded: Dict[Tuple[int, ...], float] = {}
        for current in beam:
            used = set(current)
            for candidate in candidates:
                if candidate in used:
                    continue
                proposal = tuple(sorted(current + (int(candidate),)))
                expanded[proposal] = graph_objective(
                    proposal, node_scores, edge_scores, edge_weight
                )

        ranked = sorted(
            expanded.items(), key=lambda item: item[1], reverse=True
        )
        beam = [item[0] for item in ranked[:DIRECT_CNN_GRAPH_BEAM_WIDTH]]
        if not beam:
            fallback = tuple(independent[:depth])
            beam = [fallback]

        if depth in requested:
            by_budget[depth] = list(beam[0])

    return independent, by_budget


def _prepare_direct_image_record(
    image_index,
    probe_bundle,
    centroids,
    spatial_features,
    classifier_logits,
    labels,
    top_fraction,
    strength,
):
    true_class = int(labels[image_index])
    predicted_class = int(np.argmax(classifier_logits[image_index]))
    original_logits = classifier_logits[image_index]
    layer3_tensor = torch.as_tensor(
        spatial_features[image_index : image_index + 1],
        dtype=torch.float32,
        device=DEVICE,
    )
    node_scores, edge_scores, cache, richness = direct_cnn_nodes_and_edges(
        layer3_tensor,
        centroids,
        probe_bundle,
        original_logits,
        predicted_class,
        true_class,
        top_fraction,
        strength,
    )
    return {
        "image_index": int(image_index),
        "true_class": true_class,
        "predicted_class": predicted_class,
        "original_logits": original_logits,
        "layer3_tensor": layer3_tensor,
        "node_scores": node_scores,
        "edge_scores": edge_scores,
        "cache": cache,
        "interaction_richness": float(richness),
    }


def _evaluate_record_selection(
    record,
    selected,
    centroids,
    probe_bundle,
    top_fraction,
    strength,
):
    key = tuple(sorted(set(int(i) for i in selected)))
    if key not in record["cache"]:
        record["cache"][key] = evaluate_direct_cnn_set(
            record["layer3_tensor"],
            key,
            centroids,
            probe_bundle,
            record["original_logits"],
            record["predicted_class"],
            record["true_class"],
            top_fraction,
            strength,
        )
    return record["cache"][key]


def _paired_scope_summary(details, scope_name, budgets, rng):
    scoped = details[details["evaluation_scope"] == scope_name]
    rows = []
    validated = []
    if scoped.empty:
        return rows, validated

    pivot = scoped.pivot_table(
        index=["image_index", "budget_n"],
        columns="method",
        values=[
            "margin_improvement",
            "relative_margin_recovery",
            "corrects_to_true_class",
        ],
        aggfunc="first",
    )

    for budget in budgets:
        try:
            block = pivot.xs(budget, level="budget_n")
        except KeyError:
            continue
        graph_margin = block[
            ("margin_improvement", "direct_cnn_interaction_graph")
        ].to_numpy(float)
        list_margin = block[
            ("margin_improvement", "direct_topn_independent_list")
        ].to_numpy(float)
        graph_relative = block[
            ("relative_margin_recovery", "direct_cnn_interaction_graph")
        ].to_numpy(float)
        list_relative = block[
            ("relative_margin_recovery", "direct_topn_independent_list")
        ].to_numpy(float)
        graph_correct = block[
            ("corrects_to_true_class", "direct_cnn_interaction_graph")
        ].to_numpy(float)
        list_correct = block[
            ("corrects_to_true_class", "direct_topn_independent_list")
        ].to_numpy(float)

        diff = graph_margin - list_margin
        relative_diff = graph_relative - list_relative
        mean, low, high = paired_bootstrap_difference_ci(
            diff, rng, DIRECT_CNN_GRAPH_BOOTSTRAP_SAMPLES
        )
        rel_mean, rel_low, rel_high = paired_bootstrap_difference_ci(
            relative_diff, rng, DIRECT_CNN_GRAPH_BOOTSTRAP_SAMPLES
        )
        correction_gain = float(graph_correct.mean() - list_correct.mean())
        win_rate = float(
            (diff > DIRECT_CNN_GRAPH_TIE_TOLERANCE).mean()
        )
        list_win_rate = float(
            (diff < -DIRECT_CNN_GRAPH_TIE_TOLERANCE).mean()
        )
        tie_rate = float(
            (np.abs(diff) <= DIRECT_CNN_GRAPH_TIE_TOLERANCE).mean()
        )
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        paired_effect_size = float(mean / sd) if sd > 1e-12 else 0.0

        practical = bool(
            budget > 1
            and low > 0.0
            and mean >= DIRECT_CNN_GRAPH_MIN_MEAN_ADVANTAGE
            and win_rate >= DIRECT_CNN_GRAPH_MIN_WIN_RATE
            and correction_gain >= DIRECT_CNN_GRAPH_MIN_CORRECTION_GAIN
        )
        if practical:
            validated.append(int(budget))

        rows.append(
            {
                "evaluation_scope": scope_name,
                "budget_n": int(budget),
                "n_paired_errors": int(len(diff)),
                "mean_graph_minus_list_margin": float(mean),
                "median_graph_minus_list_margin": float(np.median(diff)),
                "paired_ci95_low": float(low),
                "paired_ci95_high": float(high),
                "mean_relative_margin_recovery_gain": float(rel_mean),
                "relative_gain_ci95_low": float(rel_low),
                "relative_gain_ci95_high": float(rel_high),
                "paired_effect_size_dz": paired_effect_size,
                "graph_win_rate": win_rate,
                "tie_rate": tie_rate,
                "list_win_rate": list_win_rate,
                "graph_correction_rate": float(graph_correct.mean()),
                "list_correction_rate": float(list_correct.mean()),
                "correction_rate_difference": correction_gain,
                "graph_practically_better": practical,
            }
        )
    return rows, validated


def evaluate_direct_cnn_graph_vs_list(
    probe_bundle,
    centroids,
    spatial_features,
    classifier_logits,
    labels,
    faithful_error_indices,
    concept_labels,
    selected_config,
):
    indices = np.asarray(faithful_error_indices, dtype=np.int64)
    try:
        validation_indices, test_indices = train_test_split(
            indices,
            test_size=1.0 - DIRECT_CNN_GRAPH_VALIDATION_FRACTION,
            random_state=SEED + 1201,
            stratify=labels[indices],
        )
    except ValueError:
        validation_indices, test_indices = train_test_split(
            indices,
            test_size=1.0 - DIRECT_CNN_GRAPH_VALIDATION_FRACTION,
            random_state=SEED + 1201,
        )

    top_fraction = float(selected_config["top_fraction"])
    strength = float(selected_config["strength"])
    budgets = [
        int(v)
        for v in DIRECT_CNN_GRAPH_BUDGETS
        if v <= min(DIRECT_CNN_PAIR_POOL, len(centroids))
    ]

    validation_records = [
        _prepare_direct_image_record(
            int(image_index),
            probe_bundle,
            centroids,
            spatial_features,
            classifier_logits,
            labels,
            top_fraction,
            strength,
        )
        for image_index in tqdm(
            validation_indices,
            desc="Preparing direct-CNN graph validation errors",
        )
    ]

    selected_edge_weights: Dict[int, float] = {}
    validation_selection_rows = []
    for budget in budgets:
        best_weight = float(DIRECT_CNN_GRAPH_EDGE_WEIGHTS[0])
        best_key = (-float("inf"), -float("inf"), -float("inf"))
        for edge_weight in DIRECT_CNN_GRAPH_EDGE_WEIGHTS:
            differences = []
            correction_differences = []
            for record in validation_records:
                independent, graph_sets = select_direct_graph_sets_beam(
                    record["node_scores"],
                    record["edge_scores"],
                    [budget],
                    float(edge_weight),
                )
                list_outcome = _evaluate_record_selection(
                    record,
                    independent[:budget],
                    centroids,
                    probe_bundle,
                    top_fraction,
                    strength,
                )
                graph_outcome = _evaluate_record_selection(
                    record,
                    graph_sets[budget],
                    centroids,
                    probe_bundle,
                    top_fraction,
                    strength,
                )
                differences.append(
                    float(graph_outcome["margin_improvement"])
                    - float(list_outcome["margin_improvement"])
                )
                correction_differences.append(
                    float(graph_outcome["corrects_to_true_class"])
                    - float(list_outcome["corrects_to_true_class"])
                )

            mean_diff = float(np.mean(differences))
            correction_diff = float(np.mean(correction_differences))
            win_rate = float(
                (
                    np.asarray(differences)
                    > DIRECT_CNN_GRAPH_TIE_TOLERANCE
                ).mean()
            )
            key = (correction_diff, mean_diff, win_rate)
            validation_selection_rows.append(
                {
                    "budget_n": int(budget),
                    "edge_weight": float(edge_weight),
                    "validation_mean_margin_advantage": mean_diff,
                    "validation_correction_rate_gain": correction_diff,
                    "validation_graph_win_rate": win_rate,
                }
            )
            if key > best_key:
                best_key = key
                best_weight = float(edge_weight)
        selected_edge_weights[int(budget)] = best_weight

    validation_richness = np.asarray(
        [record["interaction_richness"] for record in validation_records],
        dtype=np.float64,
    )
    richness_threshold = float(
        np.quantile(
            validation_richness,
            DIRECT_CNN_GRAPH_INTERACTION_RICH_QUANTILE,
        )
    )

    test_records = [
        _prepare_direct_image_record(
            int(image_index),
            probe_bundle,
            centroids,
            spatial_features,
            classifier_logits,
            labels,
            top_fraction,
            strength,
        )
        for image_index in tqdm(
            test_indices,
            desc="Preparing untouched direct-CNN graph test errors",
        )
    ]

    rows = []
    edge_rows = []
    for record in tqdm(
        test_records,
        desc="Beam graph versus independent top-N on direct CNN",
    ):
        image_index = int(record["image_index"])
        rich = bool(record["interaction_richness"] >= richness_threshold)

        for (i, j), synergy in record["edge_scores"].items():
            edge_rows.append(
                {
                    "image_index": image_index,
                    "concept_i": int(i),
                    "concept_j": int(j),
                    "concept_i_label": concept_name(concept_labels, int(i)),
                    "concept_j_label": concept_name(concept_labels, int(j)),
                    "direct_double_ablation_synergy": float(synergy),
                    "node_score_i": float(record["node_scores"][i]),
                    "node_score_j": float(record["node_scores"][j]),
                    "interaction_richness": float(
                        record["interaction_richness"]
                    ),
                    "is_interaction_rich": rich,
                }
            )

        independent = [
            int(v) for v in np.argsort(record["node_scores"])[::-1]
        ]
        for budget in budgets:
            edge_weight = selected_edge_weights[budget]
            _, graph_sets = select_direct_graph_sets_beam(
                record["node_scores"],
                record["edge_scores"],
                [budget],
                edge_weight,
            )
            selections = {
                "direct_topn_independent_list": independent[:budget],
                "direct_cnn_interaction_graph": graph_sets[budget],
            }
            for method, selected in selections.items():
                outcome = _evaluate_record_selection(
                    record,
                    selected,
                    centroids,
                    probe_bundle,
                    top_fraction,
                    strength,
                )
                common = {
                    "image_index": image_index,
                    "true_class": CIFAR10_CLASSES[record["true_class"]],
                    "predicted_class": CIFAR10_CLASSES[
                        record["predicted_class"]
                    ],
                    "budget_n": int(budget),
                    "method": method,
                    "selected_edge_weight": float(edge_weight),
                    "interaction_richness": float(
                        record["interaction_richness"]
                    ),
                    "is_interaction_rich": rich,
                    "selected_concept_indices": json.dumps(selected),
                    "selected_concept_labels": json.dumps(
                        [concept_name(concept_labels, i) for i in selected]
                    ),
                    **outcome,
                }
                rows.append({**common, "evaluation_scope": "all_test_errors"})
                if rich:
                    rows.append(
                        {**common, "evaluation_scope": "interaction_rich_test_errors"}
                    )

    details = pd.DataFrame(rows)
    pd.DataFrame(edge_rows).to_csv(
        OUTPUT_DIR / "direct_cnn_interaction_edges.csv", index=False
    )
    pd.DataFrame(validation_selection_rows).to_csv(
        OUTPUT_DIR / "direct_cnn_graph_validation_selection.csv", index=False
    )

    rng = np.random.default_rng(SEED + 1202)
    summary_rows = []
    for (scope, budget, method), group in details.groupby(
        ["evaluation_scope", "budget_n", "method"]
    ):
        margin_mean, margin_low, margin_high = bootstrap_mean_ci(
            group["margin_improvement"].to_numpy(float),
            rng,
            DIRECT_CNN_GRAPH_BOOTSTRAP_SAMPLES,
        )
        relative_mean, relative_low, relative_high = bootstrap_mean_ci(
            group["relative_margin_recovery"].to_numpy(float),
            rng,
            DIRECT_CNN_GRAPH_BOOTSTRAP_SAMPLES,
        )
        correction_mean, correction_low, correction_high = bootstrap_mean_ci(
            group["corrects_to_true_class"].astype(float).to_numpy(),
            rng,
            DIRECT_CNN_GRAPH_BOOTSTRAP_SAMPLES,
        )
        summary_rows.append(
            {
                "evaluation_scope": scope,
                "budget_n": int(budget),
                "method": method,
                "n_test_errors": int(len(group)),
                "mean_margin_improvement": margin_mean,
                "margin_ci95_low": margin_low,
                "margin_ci95_high": margin_high,
                "mean_relative_margin_recovery": relative_mean,
                "relative_ci95_low": relative_low,
                "relative_ci95_high": relative_high,
                "correction_rate": correction_mean,
                "correction_ci95_low": correction_low,
                "correction_ci95_high": correction_high,
            }
        )

    paired_all, validated_all = _paired_scope_summary(
        details, "all_test_errors", budgets, rng
    )
    paired_rich, validated_rich = _paired_scope_summary(
        details, "interaction_rich_test_errors", budgets, rng
    )
    paired_rows = paired_all + paired_rich
    paired_df = pd.DataFrame(paired_rows)

    summary = pd.concat(
        [
            pd.DataFrame(summary_rows),
            paired_df.assign(method="paired_graph_minus_list"),
        ],
        ignore_index=True,
        sort=False,
    )

    if len(paired_df):
        rich_candidates = paired_df[
            paired_df["evaluation_scope"]
            == "interaction_rich_test_errors"
        ]
        best_source = rich_candidates if len(rich_candidates) else paired_df
        best = best_source.loc[
            best_source["mean_graph_minus_list_margin"].idxmax()
        ]
        best_budget = int(best["budget_n"])
        best_advantage = float(best["mean_graph_minus_list_margin"])
    else:
        best_budget = int(budgets[0])
        best_advantage = float("nan")

    metadata = {
        "methodological_change": (
            "Order-invariant joint layer3 interventions, validation-selected "
            "edge weights, graph beam search, and a validation-defined "
            "interaction-rich held-out test scope."
        ),
        "n_validation_errors_reserved": int(len(validation_indices)),
        "n_untouched_test_errors": int(len(test_indices)),
        "n_interaction_rich_test_errors": int(
            sum(record["interaction_richness"] >= richness_threshold for record in test_records)
        ),
        "budgets": budgets,
        "pair_pool": int(DIRECT_CNN_PAIR_POOL),
        "candidate_pool": int(DIRECT_CNN_GRAPH_CANDIDATE_POOL),
        "beam_width": int(DIRECT_CNN_GRAPH_BEAM_WIDTH),
        "edge_weight_candidates": [
            float(v) for v in DIRECT_CNN_GRAPH_EDGE_WEIGHTS
        ],
        "selected_edge_weights_by_budget": {
            str(k): float(v) for k, v in selected_edge_weights.items()
        },
        "interaction_rich_quantile": float(
            DIRECT_CNN_GRAPH_INTERACTION_RICH_QUANTILE
        ),
        "interaction_richness_threshold_from_validation": richness_threshold,
        "top_fraction": top_fraction,
        "strength": strength,
        "direct_cnn_graph_motivation_validated": bool(
            validated_all or validated_rich
        ),
        "overall_graph_motivation_validated": bool(validated_all),
        "interaction_rich_graph_motivation_validated": bool(validated_rich),
        "validated_budgets": sorted(
            set(validated_all + validated_rich)
        ),
        "validated_budgets_overall": validated_all,
        "validated_budgets_interaction_rich": validated_rich,
        "best_budget": best_budget,
        "best_mean_graph_minus_list_margin": best_advantage,
        "paired_comparison": paired_rows,
        "interpretation_note": (
            "A conditional gain on interaction-rich errors must be reported as "
            "conditional; it is not an overall-dataset gain. No result is forced."
        ),
    }
    return details, summary, metadata

def evaluate_single_cnn_intervention(
    layer3_tensor: torch.Tensor,
    concept_index: int,
    centroids: np.ndarray,
    probe_bundle: Mapping[str, np.ndarray],
    original_logits: np.ndarray,
    predicted_class: int,
    true_class: int,
    top_fraction: float,
    strength: float,
    target_delta_norm: Optional[float] = None,
) -> Dict[str, object]:
    modified_layer3, perturbation_stats = (
        apply_concept_projection_removal(
            layer3_features=layer3_tensor,
            centroid=centroids[concept_index],
            top_fraction=top_fraction,
            strength=strength,
            target_delta_norm=target_delta_norm,
        )
    )

    with torch.no_grad():
        modified_pooled = (
            modified_layer3.mean(dim=(2, 3))
            .cpu()
            .numpy()
        )

    modified_logits = cifar_probe_logits_from_pooled_features(
        modified_pooled,
        probe_bundle,
    )[0]

    original_margin = float(
        original_logits[true_class]
        - original_logits[predicted_class]
    )
    modified_margin = float(
        modified_logits[true_class]
        - modified_logits[predicted_class]
    )

    logit_change = original_logits - modified_logits
    non_predicted_indices = [
        index
        for index in range(len(CIFAR10_CLASSES))
        if index != predicted_class
    ]
    selectivity = float(
        logit_change[predicted_class]
        - np.mean(
            np.abs(logit_change[non_predicted_indices])
        )
    )

    return {
        "concept_index": int(concept_index),
        "cnn_margin_improvement": float(
            modified_margin - original_margin
        ),
        "cnn_corrects_to_true_class": bool(
            int(np.argmax(modified_logits)) == true_class
        ),
        "cnn_new_prediction": int(np.argmax(modified_logits)),
        "predicted_logit_drop": float(
            original_logits[predicted_class]
            - modified_logits[predicted_class]
        ),
        "true_logit_change": float(
            modified_logits[true_class]
            - original_logits[true_class]
        ),
        "intervention_selectivity": selectivity,
        **perturbation_stats,
    }


def select_cnn_intervention_configuration(
    resnet_model: nn.Module,
    probe_bundle: Mapping[str, np.ndarray],
    centroids: np.ndarray,
    spatial_features: np.ndarray,
    concept_activations: np.ndarray,
    classifier_logits: np.ndarray,
    labels: np.ndarray,
    surrogate_model: SurrogateModel,
    validation_indices: Sequence[int],
    aggregated_coefficients: np.ndarray,
) -> Dict[str, object]:
    rng = np.random.default_rng(SEED + 350)
    rows = []

    for top_fraction in CNN_INTERVENTION_TOP_FRACTIONS:
        for strength in CNN_INTERVENTION_STRENGTHS:
            for alpha in CNN_HYBRID_ALPHAS:
                proposed_margins = []
                proposed_corrections = []
                proposed_selectivity = []
                random_margins = []
                random_corrections = []

                for image_index in validation_indices:
                    true_class = int(labels[image_index])
                    predicted_class = int(
                        np.argmax(classifier_logits[image_index])
                    )
                    concept_vector = concept_activations[image_index]
                    layer3_tensor = torch.as_tensor(
                        spatial_features[
                            image_index:image_index + 1
                        ],
                        dtype=torch.float32,
                        device=DEVICE,
                    )
                    original_logits = classifier_logits[image_index]

                    selected_concept = select_concept_for_method(
                        method="hybrid_global_local",
                        model=surrogate_model,
                        concept_vector=concept_vector,
                        predicted_class=predicted_class,
                        true_class=true_class,
                        aggregated_coefficients=(
                            aggregated_coefficients
                        ),
                        rng=rng,
                        hybrid_alpha=alpha,
                    )

                    proposed = evaluate_single_cnn_intervention(
                        layer3_tensor=layer3_tensor,
                        concept_index=selected_concept,
                        centroids=centroids,
                        probe_bundle=probe_bundle,
                        original_logits=original_logits,
                        predicted_class=predicted_class,
                        true_class=true_class,
                        top_fraction=top_fraction,
                        strength=strength,
                    )

                    proposed_margins.append(
                        proposed["cnn_margin_improvement"]
                    )
                    proposed_corrections.append(
                        float(
                            proposed[
                                "cnn_corrects_to_true_class"
                            ]
                        )
                    )
                    proposed_selectivity.append(
                        proposed["intervention_selectivity"]
                    )

                    target_norm = proposed[
                        "delta_norm_after_matching"
                    ]

                    random_draw_margins = []
                    random_draw_corrections = []

                    for _ in range(
                        CNN_RANDOM_ENERGY_MATCHED_DRAWS
                    ):
                        random_index = int(
                            rng.integers(
                                0,
                                surrogate_model.base_concept_count,
                            )
                        )
                        random_result = (
                            evaluate_single_cnn_intervention(
                                layer3_tensor=layer3_tensor,
                                concept_index=random_index,
                                centroids=centroids,
                                probe_bundle=probe_bundle,
                                original_logits=original_logits,
                                predicted_class=predicted_class,
                                true_class=true_class,
                                top_fraction=top_fraction,
                                strength=strength,
                                target_delta_norm=target_norm,
                            )
                        )
                        random_draw_margins.append(
                            random_result[
                                "cnn_margin_improvement"
                            ]
                        )
                        random_draw_corrections.append(
                            float(
                                random_result[
                                    "cnn_corrects_to_true_class"
                                ]
                            )
                        )

                    random_margins.append(
                        float(np.mean(random_draw_margins))
                    )
                    random_corrections.append(
                        float(
                            np.mean(random_draw_corrections)
                        )
                    )

                proposed_margin = float(
                    np.mean(proposed_margins)
                )
                random_margin = float(
                    np.mean(random_margins)
                )
                proposed_correction = float(
                    np.mean(proposed_corrections)
                )
                random_correction = float(
                    np.mean(random_corrections)
                )

                rows.append(
                    {
                        "top_fraction": float(top_fraction),
                        "strength": float(strength),
                        "hybrid_alpha": float(alpha),
                        "proposed_mean_margin_gain": (
                            proposed_margin
                        ),
                        "random_mean_margin_gain": random_margin,
                        "margin_gain_over_random": float(
                            proposed_margin - random_margin
                        ),
                        "proposed_correction_rate": (
                            proposed_correction
                        ),
                        "random_correction_rate": (
                            random_correction
                        ),
                        "correction_gain_over_random": float(
                            proposed_correction
                            - random_correction
                        ),
                        "mean_selectivity": float(
                            np.mean(proposed_selectivity)
                        ),
                    }
                )

    selection_df = pd.DataFrame(rows)
    selection_df.to_csv(
        OUTPUT_DIR / "cnn_intervention_sweep.csv",
        index=False,
    )

    best = max(
        rows,
        key=lambda row: (
            row["margin_gain_over_random"],
            row["correction_gain_over_random"],
            row["mean_selectivity"],
            -row["strength"],
            -row["top_fraction"],
        ),
    )

    return best


def evaluate_cnn_intervention_transfer(
    resnet_model: nn.Module,
    probe_bundle: Mapping[str, np.ndarray],
    centroids: np.ndarray,
    spatial_features: np.ndarray,
    concept_activations: np.ndarray,
    classifier_logits: np.ndarray,
    labels: np.ndarray,
    surrogate_model: SurrogateModel,
    faithful_error_indices: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(SEED + 300)

    all_indices = np.asarray(
        faithful_error_indices,
        dtype=np.int64,
    )

    if CNN_INTERVENTION_MAX_ERRORS is not None:
        all_indices = all_indices[
            : int(CNN_INTERVENTION_MAX_ERRORS)
        ]

    validation_indices, evaluation_indices = train_test_split(
        all_indices,
        test_size=(
            1.0 - CNN_INTERVENTION_VALIDATION_FRACTION
        ),
        random_state=SEED + 301,
        stratify=labels[all_indices],
    )

    raw_node_coefficients = surrogate_model.coefficients[
        :, : surrogate_model.concept_feature_count
    ]
    aggregated_coefficients = np.zeros(
        (
            len(CIFAR10_CLASSES),
            surrogate_model.base_concept_count,
        ),
        dtype=np.float32,
    )
    for block_index in range(4):
        start = (
            block_index * surrogate_model.base_concept_count
        )
        stop = start + surrogate_model.base_concept_count
        aggregated_coefficients += raw_node_coefficients[
            :, start:stop
        ]

    selected_config = select_cnn_intervention_configuration(
        resnet_model=resnet_model,
        probe_bundle=probe_bundle,
        centroids=centroids,
        spatial_features=spatial_features,
        concept_activations=concept_activations,
        classifier_logits=classifier_logits,
        labels=labels,
        surrogate_model=surrogate_model,
        validation_indices=validation_indices,
        aggregated_coefficients=aggregated_coefficients,
    )

    methods = [
        "hybrid_global_local",
        "local_node_contribution",
        "global_wrong_class_coefficient",
        "highest_activation",
        "random_concept",
    ]

    rows = []

    for image_index in tqdm(
        evaluation_indices,
        desc="Evaluating selected CNN intervention",
    ):
        true_class = int(labels[image_index])
        predicted_class = int(
            np.argmax(classifier_logits[image_index])
        )
        concept_vector = concept_activations[image_index]
        original_logits = classifier_logits[image_index]

        layer3_tensor = torch.as_tensor(
            spatial_features[
                image_index:image_index + 1
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        hybrid_index = select_concept_for_method(
            method="hybrid_global_local",
            model=surrogate_model,
            concept_vector=concept_vector,
            predicted_class=predicted_class,
            true_class=true_class,
            aggregated_coefficients=aggregated_coefficients,
            rng=rng,
            hybrid_alpha=selected_config["hybrid_alpha"],
        )
        hybrid_result = evaluate_single_cnn_intervention(
            layer3_tensor=layer3_tensor,
            concept_index=hybrid_index,
            centroids=centroids,
            probe_bundle=probe_bundle,
            original_logits=original_logits,
            predicted_class=predicted_class,
            true_class=true_class,
            top_fraction=selected_config["top_fraction"],
            strength=selected_config["strength"],
        )
        target_norm = hybrid_result[
            "delta_norm_after_matching"
        ]

        for method in methods:
            draws = (
                CNN_RANDOM_ENERGY_MATCHED_DRAWS
                if method == "random_concept"
                else 1
            )
            draw_rows = []

            for draw_index in range(draws):
                if method == "hybrid_global_local":
                    concept_index = hybrid_index
                    result = hybrid_result
                else:
                    concept_index = select_concept_for_method(
                        method=method,
                        model=surrogate_model,
                        concept_vector=concept_vector,
                        predicted_class=predicted_class,
                        true_class=true_class,
                        aggregated_coefficients=(
                            aggregated_coefficients
                        ),
                        rng=rng,
                        hybrid_alpha=selected_config[
                            "hybrid_alpha"
                        ],
                    )
                    result = evaluate_single_cnn_intervention(
                        layer3_tensor=layer3_tensor,
                        concept_index=concept_index,
                        centroids=centroids,
                        probe_bundle=probe_bundle,
                        original_logits=original_logits,
                        predicted_class=predicted_class,
                        true_class=true_class,
                        top_fraction=selected_config[
                            "top_fraction"
                        ],
                        strength=selected_config["strength"],
                        target_delta_norm=(
                            target_norm
                            if method == "random_concept"
                            else None
                        ),
                    )

                surrogate_result = intervention_outcome(
                    surrogate_model,
                    concept_vector,
                    concept_index,
                    predicted_class,
                    true_class,
                )

                draw_rows.append(
                    {
                        **result,
                        "surrogate_margin_improvement": float(
                            surrogate_result[
                                "margin_improvement"
                            ]
                        ),
                        "surrogate_corrects_to_true_class": float(
                            surrogate_result[
                                "corrects_to_true_class"
                            ]
                        ),
                    }
                )

            rows.append(
                {
                    "image_index": int(image_index),
                    "method": method,
                    "concept_index": (
                        int(draw_rows[0]["concept_index"])
                        if draws == 1
                        else -1
                    ),
                    "cnn_margin_improvement": float(
                        np.mean(
                            [
                                item[
                                    "cnn_margin_improvement"
                                ]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "cnn_corrects_to_true_class": float(
                        np.mean(
                            [
                                item[
                                    "cnn_corrects_to_true_class"
                                ]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "surrogate_margin_improvement": float(
                        np.mean(
                            [
                                item[
                                    "surrogate_margin_improvement"
                                ]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "surrogate_corrects_to_true_class": float(
                        np.mean(
                            [
                                item[
                                    "surrogate_corrects_to_true_class"
                                ]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "predicted_logit_drop": float(
                        np.mean(
                            [
                                item["predicted_logit_drop"]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "true_logit_change": float(
                        np.mean(
                            [
                                item["true_logit_change"]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "intervention_selectivity": float(
                        np.mean(
                            [
                                item[
                                    "intervention_selectivity"
                                ]
                                for item in draw_rows
                            ]
                        )
                    ),
                    "delta_norm": float(
                        np.mean(
                            [
                                item[
                                    "delta_norm_after_matching"
                                ]
                                for item in draw_rows
                            ]
                        )
                    ),
                }
            )

    details = pd.DataFrame(rows)
    summary_rows = []
    bootstrap_rng = np.random.default_rng(SEED + 400)

    for method, group in details.groupby("method"):
        margin_mean, margin_low, margin_high = bootstrap_mean_ci(
            group["cnn_margin_improvement"].to_numpy(),
            bootstrap_rng,
        )
        correction_mean, correction_low, correction_high = (
            bootstrap_mean_ci(
                group["cnn_corrects_to_true_class"].to_numpy(),
                bootstrap_rng,
            )
        )

        correlation = float("nan")
        if (
            group["cnn_margin_improvement"].std() > 1e-8
            and group["surrogate_margin_improvement"].std() > 1e-8
        ):
            correlation = float(
                np.corrcoef(
                    group["cnn_margin_improvement"],
                    group["surrogate_margin_improvement"],
                )[0, 1]
            )

        direction_agreement = float(
            (
                (
                    group["cnn_margin_improvement"] > 0
                )
                == (
                    group[
                        "surrogate_margin_improvement"
                    ] > 0
                )
            ).mean()
        )

        summary_rows.append(
            {
                "method": method,
                "n_intervention_test_errors": int(len(group)),
                "cnn_mean_margin_improvement": margin_mean,
                "cnn_margin_ci95_low": margin_low,
                "cnn_margin_ci95_high": margin_high,
                "cnn_positive_margin_rate": float(
                    (
                        group["cnn_margin_improvement"] > 0
                    ).mean()
                ),
                "cnn_correction_rate": correction_mean,
                "cnn_correction_ci95_low": correction_low,
                "cnn_correction_ci95_high": correction_high,
                "mean_predicted_logit_drop": float(
                    group["predicted_logit_drop"].mean()
                ),
                "mean_true_logit_change": float(
                    group["true_logit_change"].mean()
                ),
                "mean_intervention_selectivity": float(
                    group["intervention_selectivity"].mean()
                ),
                "mean_delta_norm": float(
                    group["delta_norm"].mean()
                ),
                "surrogate_cnn_margin_correlation": correlation,
                "surrogate_cnn_direction_agreement": (
                    direction_agreement
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    metadata = {
        "n_faithful_errors_total": int(len(all_indices)),
        "n_intervention_validation_errors": int(
            len(validation_indices)
        ),
        "n_intervention_test_errors": int(
            len(evaluation_indices)
        ),
        "selected_configuration": selected_config,
        "summary": summary_df.to_dict(orient="records"),
    }

    return details, summary_df, metadata



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
    relation_data: Optional[ConceptRelationData] = None,
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
        interaction_strength = float(
            max(abs(v) for v in contributions.values())
        )
        relation_attributes = (
            concept_pair_relation_attributes(
                relation_data,
                i,
                j,
                interaction_strength,
            )
        )

        graph.add_edge(
            f"concept:{i}",
            f"concept:{j}",
            edge_type="concept_interaction",
            pair_index=int(pair_index),
            weight=interaction_strength,
            class_contributions=json.dumps(contributions),
            **relation_attributes,
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
                **concept_pair_relation_attributes(
                    relation_data,
                    int(item["concept_i"]),
                    int(item["concept_j"]),
                    float(
                        item["absolute_interaction_strength"]
                    ),
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
        coherent_edges = []
        anomaly_edges = []
        uncertain_edges = []

        for edge in interaction_edges:
            relation_type = graph[
                edge[0]
            ][edge[1]].get(
                "semantic_relation_type",
                "contextual_or_uncertain",
            )
            if relation_type == "semantically_coherent":
                coherent_edges.append(edge)
            elif relation_type == "semantic_anomaly":
                anomaly_edges.append(edge)
            else:
                uncertain_edges.append(edge)

        for edge_group, line_style, edge_alpha in [
            (coherent_edges, "solid", 0.75),
            (uncertain_edges, "dashed", 0.60),
            (anomaly_edges, "dotted", 0.90),
        ]:
            if not edge_group:
                continue
            widths = [
                1.0 + 4.0 * graph[u][v]["weight"]
                for u, v in edge_group
            ]
            nx.draw_networkx_edges(
                graph,
                position,
                edgelist=edge_group,
                width=widths,
                style=line_style,
                arrows=False,
                alpha=edge_alpha,
            )

        semantic_edge_labels = {}
        for u, v in interaction_edges:
            data = graph[u][v]
            if "combined_similarity" in data:
                semantic_edge_labels[(u, v)] = (
                    f"I={data['weight']:.2f}\n"
                    f"S={data['combined_similarity']:.2f}\n"
                    f"A={data['semantic_anomaly_score']:.2f}"
                )

        if semantic_edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                position,
                edge_labels=semantic_edge_labels,
                font_size=6,
                rotate=False,
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
    relation_data: Optional[ConceptRelationData] = None,
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

        relation_attributes = (
            concept_pair_relation_attributes(
                relation_data,
                int(i),
                int(j),
                abs(coefficient),
            )
        )

        graph.add_edge(
            f"concept:{int(i)}",
            f"concept:{int(j)}",
            edge_type="concept_interaction",
            weight=abs(coefficient),
            strongest_class=CIFAR10_CLASSES[max_class],
            signed_coefficient=coefficient,
            **relation_attributes,
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
        "CNN_INTERVENTION_TOP_FRACTIONS",
        "CNN_INTERVENTION_STRENGTHS",
        "CNN_INTERVENTION_VALIDATION_FRACTION",
        "CNN_INTERVENTION_MAX_ERRORS",
        "CNN_HYBRID_ALPHAS",
        "CNN_RANDOM_ENERGY_MATCHED_DRAWS",
        "TOP_N_BUDGETS",
        "GRAPH_VS_LIST_BOOTSTRAP_SAMPLES",
        "GRAPH_VS_LIST_TIE_TOLERANCE",
        "CIFAR10_BATCHES_DIR",
        "SEMANTIC_ANALYSIS_ENABLED",
        "SEMANTIC_TEXT_MODEL",
        "SEMANTIC_DESCRIPTION_MAX_CAPTIONS",
        "SEMANTIC_VISUAL_WEIGHT",
        "SEMANTIC_ANOMALY_THRESHOLD",
        "SEMANTIC_RANDOM_PAIRS",
        "SEMANTIC_EMBEDDING_BATCH_SIZE",
    ]

    config = {
        key: (str(globals()[key]) if isinstance(globals()[key], Path) else globals()[key])
        for key in keys
    }
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

    (
        classifier_logits,
        probe_metrics,
        probe_bundle,
    ) = train_cifar_probe(
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

    relation_data = build_concept_relation_data(
        concept_labels=concept_labels,
        centroids=centroids,
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

    (
        semantic_interaction_edges,
        semantic_interaction_summary,
    ) = analyze_interaction_semantics(
        model=surrogate,
        concept_labels=concept_labels,
        relation_data=relation_data,
    )

    semantic_interaction_edges.to_csv(
        OUTPUT_DIR / "semantic_interaction_edges.csv",
        index=False,
    )
    (
        OUTPUT_DIR / "semantic_interaction_summary.json"
    ).write_text(
        json.dumps(
            semantic_interaction_summary,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_semantic_interaction_analysis(
        semantic_interaction_edges,
        OUTPUT_DIR
        / "semantic_interaction_strength_vs_similarity.png",
    )

    metrics.update(
        {
            key: value
            for key, value in semantic_interaction_summary.items()
            if isinstance(value, (int, float, bool))
        }
    )

    global_graph = build_global_graph(
        surrogate,
        concept_labels,
        relation_data=relation_data,
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

    faithful_error_indices = (
        baseline_details.loc[
            baseline_details["method"] == "proposed_best_cf",
            "image_index",
        ]
        .drop_duplicates()
        .astype(int)
        .to_numpy()
    )

    (
        graph_vs_list_details,
        graph_vs_list_summary,
        graph_vs_list_metadata,
    ) = evaluate_graph_vs_topn_list(
        model=surrogate,
        concept_activations=concept_activations,
        classifier_logits=classifier_logits,
        labels=labels,
        error_indices=error_indices,
        concept_labels=concept_labels,
    )

    graph_vs_list_details.to_csv(
        OUTPUT_DIR / "graph_vs_topn_details.csv",
        index=False,
    )
    graph_vs_list_summary.to_csv(
        OUTPUT_DIR / "graph_vs_topn_summary.csv",
        index=False,
    )
    pd.DataFrame(
        graph_vs_list_metadata["paired_comparison"]
    ).to_csv(
        OUTPUT_DIR / "graph_vs_topn_paired.csv",
        index=False,
    )
    (OUTPUT_DIR / "graph_vs_topn_summary.json").write_text(
        json.dumps(graph_vs_list_metadata, indent=2),
        encoding="utf-8",
    )
    plot_graph_vs_topn_summary(
        graph_vs_list_summary,
        OUTPUT_DIR / "graph_vs_topn_margin.png",
    )

    metrics["graph_motivation_validated"] = bool(
        graph_vs_list_metadata["graph_motivation_validated"]
    )
    metrics["graph_validated_budgets_ci95"] = (
        graph_vs_list_metadata["validated_budgets_practical"]
    )
    metrics["graph_best_budget"] = (
        graph_vs_list_metadata[
            "best_budget_by_mean_margin_advantage"
        ]
    )
    metrics["graph_best_mean_margin_advantage"] = float(
        graph_vs_list_metadata[
            "best_mean_graph_minus_list_margin"
        ]
    )

    (
        cnn_intervention_details,
        cnn_intervention_summary,
        cnn_intervention_metadata,
    ) = evaluate_cnn_intervention_transfer(
        resnet_model=model,
        probe_bundle=probe_bundle,
        centroids=centroids,
        spatial_features=spatial_features,
        concept_activations=concept_activations,
        classifier_logits=classifier_logits,
        labels=labels,
        surrogate_model=surrogate,
        faithful_error_indices=faithful_error_indices,
    )

    cnn_intervention_details.to_csv(
        OUTPUT_DIR / "cnn_intervention_details.csv",
        index=False,
    )
    cnn_intervention_summary.to_csv(
        OUTPUT_DIR / "cnn_intervention_summary.csv",
        index=False,
    )
    (
        OUTPUT_DIR / "cnn_intervention_summary.json"
    ).write_text(
        json.dumps(
            cnn_intervention_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics["n_cnn_intervention_errors"] = int(
        cnn_intervention_metadata[
            "n_intervention_test_errors"
        ]
    )
    metrics[
        "n_cnn_intervention_validation_errors"
    ] = int(
        cnn_intervention_metadata[
            "n_intervention_validation_errors"
        ]
    )
    metrics[
        "selected_cnn_intervention_top_fraction"
    ] = float(
        cnn_intervention_metadata[
            "selected_configuration"
        ]["top_fraction"]
    )
    metrics[
        "selected_cnn_intervention_strength"
    ] = float(
        cnn_intervention_metadata[
            "selected_configuration"
        ]["strength"]
    )
    metrics[
        "selected_cnn_hybrid_alpha"
    ] = float(
        cnn_intervention_metadata[
            "selected_configuration"
        ]["hybrid_alpha"]
    )

    proposed_rows = cnn_intervention_summary[
        cnn_intervention_summary["method"]
        == "hybrid_global_local"
    ]
    if len(proposed_rows) == 1:
        proposed_row = proposed_rows.iloc[0]
        metrics["cnn_positive_margin_rate"] = float(
            proposed_row["cnn_positive_margin_rate"]
        )
        metrics["cnn_correction_rate"] = float(
            proposed_row["cnn_correction_rate"]
        )
        metrics[
            "surrogate_cnn_margin_correlation"
        ] = float(
            proposed_row[
                "surrogate_cnn_margin_correlation"
            ]
        )
        metrics[
            "surrogate_cnn_direction_agreement"
        ] = float(
            proposed_row[
                "surrogate_cnn_direction_agreement"
            ]
        )


    direct_graph_details, direct_graph_summary, direct_graph_metadata = (
        evaluate_direct_cnn_graph_vs_list(
            probe_bundle=probe_bundle, centroids=centroids,
            spatial_features=spatial_features,
            classifier_logits=classifier_logits, labels=labels,
            faithful_error_indices=faithful_error_indices,
            concept_labels=concept_labels,
            selected_config=cnn_intervention_metadata["selected_configuration"],
        )
    )
    direct_graph_details.to_csv(
        OUTPUT_DIR / "direct_cnn_graph_vs_topn_details.csv", index=False
    )
    direct_graph_summary.to_csv(
        OUTPUT_DIR / "direct_cnn_graph_vs_topn_summary.csv", index=False
    )
    (OUTPUT_DIR / "direct_cnn_graph_vs_topn_summary.json").write_text(
        json.dumps(direct_graph_metadata, indent=2), encoding="utf-8"
    )
    metrics["direct_cnn_graph_motivation_validated"] = bool(
        direct_graph_metadata["direct_cnn_graph_motivation_validated"]
    )
    metrics["direct_cnn_graph_validated_budgets"] = direct_graph_metadata["validated_budgets"]
    metrics["direct_cnn_graph_best_budget"] = int(direct_graph_metadata["best_budget"])
    metrics["direct_cnn_graph_best_mean_margin_advantage"] = float(
        direct_graph_metadata["best_mean_graph_minus_list_margin"]
    )
    metrics["direct_cnn_graph_validated_overall"] = bool(
        direct_graph_metadata.get("overall_graph_motivation_validated", False)
    )
    metrics["direct_cnn_graph_validated_interaction_rich"] = bool(
        direct_graph_metadata.get(
            "interaction_rich_graph_motivation_validated", False
        )
    )
    metrics["direct_cnn_graph_validated_budgets_overall"] = (
        direct_graph_metadata.get("validated_budgets_overall", [])
    )
    metrics["direct_cnn_graph_validated_budgets_interaction_rich"] = (
        direct_graph_metadata.get(
            "validated_budgets_interaction_rich", []
        )
    )
    metrics["n_direct_cnn_interaction_rich_test_errors"] = int(
        direct_graph_metadata.get("n_interaction_rich_test_errors", 0)
    )

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
            relation_data=relation_data,
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
    print(" -", OUTPUT_DIR / "semantic_interaction_edges.csv")
    print(" -", OUTPUT_DIR / "semantic_interaction_summary.json")
    print(
        " -",
        OUTPUT_DIR
        / "semantic_interaction_strength_vs_similarity.png",
    )
    print(" -", OUTPUT_DIR / "misclassification_explanations.csv")
    print(" -", OUTPUT_DIR / "graph_vs_topn_details.csv")
    print(" -", OUTPUT_DIR / "graph_vs_topn_summary.csv")
    print(" -", OUTPUT_DIR / "graph_vs_topn_paired.csv")
    print(" -", OUTPUT_DIR / "graph_vs_topn_summary.json")
    print(" -", OUTPUT_DIR / "graph_vs_topn_margin.png")
    print(" -", OUTPUT_DIR / "direct_cnn_graph_vs_topn_details.csv")
    print(" -", OUTPUT_DIR / "direct_cnn_graph_vs_topn_summary.csv")
    print(" -", OUTPUT_DIR / "direct_cnn_graph_vs_topn_summary.json")
    print(" -", OUTPUT_DIR / "direct_cnn_interaction_edges.csv")
    print(" -", OUTPUT_DIR / "direct_cnn_graph_validation_selection.csv")
    print(" -", EXPLANATION_DIR)


if __name__ == "__main__":
    run_experiment()