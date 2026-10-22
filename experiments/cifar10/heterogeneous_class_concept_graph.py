from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

SEED = 42
BATCH_SIZE = 64
N_IMAGES = 5000
N_PERMUTATIONS = 100
CLASS_SELECTIVITY_THRESHOLD = 0.35
CONCEPT_CORRELATION_THRESHOLD = 0.08
CONCEPT_CORRELATION_MARGIN = 0.01
CLASS_EDGE_TOP_QUANTILE = 0.70
CONCEPT_EDGE_TOP_QUANTILE = 0.85
MIN_CONCEPT_CLASS_FREQUENCY = 2
MAX_CONCEPT_CLASS_FREQUENCY = 6
TOP_CONCEPTS_PER_CLASS = 10
MAX_CONCEPTS_TOTAL = 60
MAX_CONCEPTS_IN_GRAPH = 40
USE_BIGRAMS = True
TOP_CLASS_CONCEPT_EDGES = 7

OUTPUT_DIR = "outputs_wordnet_heterogeneous_graphs"
DATA_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

CLASS_SYNSETS = {
    "airplane": "airplane.n.01",
    "automobile": "car.n.01",
    "bird": "bird.n.01",
    "cat": "cat.n.01",
    "deer": "deer.n.01",
    "dog": "dog.n.01",
    "frog": "frog.n.01",
    "horse": "horse.n.01",
    "ship": "ship.n.01",
    "truck": "truck.n.01",
}

RELATION_WEIGHTS = {
    "self": 1.00,
    "hypernym": 0.85,
    "instance_hypernym": 0.85,
    "hyponym": 0.45,
    "instance_hyponym": 0.45,
    "part_meronym": 0.65,
    "member_meronym": 0.55,
    "substance_meronym": 0.55,
    "part_holonym": 0.45,
    "member_holonym": 0.45,
    "substance_holonym": 0.45,
    "similar_to": 0.35,
    "also_see": 0.30,
}

GENERIC_TERMS = {
    "entity",
    "object",
    "thing",
    "something",
    "anything",
    "someone",
    "one",
    "kind",
    "type",
    "class",
    "group",
    "member",
    "part",
    "form",
    "make",
    "use",
    "used",
    "usually",
    "especially",
    "often",
    "characterized",
    "consisting",
    "including",
    "having",
    "relating",
    "large",
    "small",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z_-]*")

FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "which", "with", "without", "who", "whose", "can", "may",
}

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def ensure_nltk_resources() -> None:

    try:
        _ = wn.synset("dog.n.01")
    except LookupError:
        print("NLTK WordNet is missing; attempting a one-time download...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        try:
            _ = wn.synset("dog.n.01")
        except LookupError as exc:
            raise RuntimeError(
                "WordNet could not be downloaded. Connect to the internet and run "
                "`python -m nltk.downloader wordnet omw-1.4`, then rerun the script."
            ) from exc

    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model.to(DEVICE)
    transform = weights.transforms()
    return model, transform


def load_cifar10(transform) -> DataLoader:
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    indices = np.random.choice(
        len(dataset),
        size=min(N_IMAGES, len(dataset)),
        replace=False,
    )

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def extract_layer_activations(model, loader):
    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    activations = {name: [] for name in layer_names}
    labels = []
    hooks = []

    def make_hook(layer_name):
        def hook(_module, _inp, out):
            pooled = out.detach().mean(dim=(2, 3)).cpu().numpy()
            activations[layer_name].append(pooled)

        return hook

    for name in layer_names:
        layer = getattr(model, name)
        hooks.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for images, y in tqdm(loader, desc="Extracting activations"):
            images = images.to(DEVICE, non_blocking=True)
            _ = model(images)
            labels.append(y.numpy())

    for hook in hooks:
        hook.remove()

    labels = np.concatenate(labels, axis=0)

    for name in layer_names:
        activations[name] = np.concatenate(activations[name], axis=0)

    return activations, labels


@dataclass(frozen=True)
class GlossPart:
    relation: str
    synset_name: str
    weight: float
    text: str


def synset_text(synset) -> str:

    lemmas = " ".join(lemma.name().replace("_", " ") for lemma in synset.lemmas())
    examples = " ".join(synset.examples())
    return f"{lemmas}. {synset.definition()}. {examples}".strip()


def related_synsets(synset) -> List[Tuple[str, object]]:
    relations = [
        ("hypernym", synset.hypernyms()),
        ("instance_hypernym", synset.instance_hypernyms()),
        ("hyponym", synset.hyponyms()),
        ("instance_hyponym", synset.instance_hyponyms()),
        ("part_meronym", synset.part_meronyms()),
        ("member_meronym", synset.member_meronyms()),
        ("substance_meronym", synset.substance_meronyms()),
        ("part_holonym", synset.part_holonyms()),
        ("member_holonym", synset.member_holonyms()),
        ("substance_holonym", synset.substance_holonyms()),
        ("similar_to", synset.similar_tos()),
        ("also_see", synset.also_sees()),
    ]

    output: List[Tuple[str, object]] = []
    seen = set()

    for relation_name, synsets in relations:
        for related in synsets:
            key = (relation_name, related.name())
            if key not in seen:
                seen.add(key)
                output.append((relation_name, related))

    return output


def build_extended_gloss(synset) -> List[GlossPart]:

    parts = [
        GlossPart(
            relation="self",
            synset_name=synset.name(),
            weight=RELATION_WEIGHTS["self"],
            text=synset_text(synset),
        )
    ]

    for relation_name, related in related_synsets(synset):
        parts.append(
            GlossPart(
                relation=relation_name,
                synset_name=related.name(),
                weight=RELATION_WEIGHTS[relation_name],
                text=synset_text(related),
            )
        )

    return parts


def normalize_tokens(
    text: str,
    lemmatizer: WordNetLemmatizer,
    stop_words: set,
    excluded_terms: set,
    use_bigrams: bool = True,
) -> List[str]:
    raw_tokens = [match.group(0).lower().replace("_", " ") for match in TOKEN_PATTERN.finditer(text)]

    tokens: List[str] = []
    for token in raw_tokens:
        token = token.strip("-_")

        if len(token) < 3:
            continue
        if token in stop_words or token in GENERIC_TERMS or token in excluded_terms:
            continue

        lemma = lemmatizer.lemmatize(token, pos="n")
        lemma = lemmatizer.lemmatize(lemma, pos="v")

        if len(lemma) < 3:
            continue
        if lemma in stop_words or lemma in GENERIC_TERMS or lemma in excluded_terms:
            continue
        if not lemma.replace("-", "").isalpha():
            continue

        tokens.append(lemma)

    if not use_bigrams:
        return tokens

    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
        if tokens[i] != tokens[i + 1]
    ]

    return tokens + bigrams


def build_wordnet_concept_matrix(
    class_names: Sequence[str],
    class_synsets: Mapping[str, str],
):
    ensure_nltk_resources()

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set(FALLBACK_STOP_WORDS)

    lemmatizer = WordNetLemmatizer()

    excluded_terms = set(class_names)
    excluded_terms.update({"car", "auto", "plane", "aeroplane"})

    class_term_scores: Dict[str, Counter] = {}
    class_term_relations: Dict[str, Dict[str, Counter]] = {}
    gloss_rows = []

    for class_name in class_names:
        synset = wn.synset(class_synsets[class_name])
        parts = build_extended_gloss(synset)

        term_scores: Counter = Counter()
        term_relations: Dict[str, Counter] = defaultdict(Counter)

        for part in parts:
            terms = normalize_tokens(
                part.text,
                lemmatizer=lemmatizer,
                stop_words=stop_words,
                excluded_terms=excluded_terms,
                use_bigrams=USE_BIGRAMS,
            )

            counts = Counter(terms)
            for term, count in counts.items():
                weighted_count = part.weight * count
                term_scores[term] += weighted_count
                term_relations[term][part.relation] += weighted_count

            gloss_rows.append(
                {
                    "class": class_name,
                    "base_synset": synset.name(),
                    "relation": part.relation,
                    "related_synset": part.synset_name,
                    "relation_weight": part.weight,
                    "text": part.text,
                }
            )

        class_term_scores[class_name] = term_scores
        class_term_relations[class_name] = term_relations

    n_classes = len(class_names)

    document_frequency = Counter()
    for class_name in class_names:
        for term in class_term_scores[class_name]:
            if class_term_scores[class_name][term] > 0:
                document_frequency[term] += 1

    candidate_terms = [
        term
        for term, df in document_frequency.items()
        if MIN_CONCEPT_CLASS_FREQUENCY <= df <= MAX_CONCEPT_CLASS_FREQUENCY
    ]

    if not candidate_terms:
        raise RuntimeError(
            "No shared WordNet concepts survived filtering. "
            "Try increasing MAX_CONCEPT_CLASS_FREQUENCY or disabling bigrams."
        )

    idf = {
        term: math.log((1.0 + n_classes) / (1.0 + document_frequency[term])) + 1.0
        for term in candidate_terms
    }

    preliminary = np.zeros((n_classes, len(candidate_terms)), dtype=np.float32)

    for i, class_name in enumerate(class_names):
        for j, term in enumerate(candidate_terms):
            tf = class_term_scores[class_name].get(term, 0.0)
            if tf > 0:
                preliminary[i, j] = math.log1p(tf) * idf[term]

    selected_indices = set()
    for i in range(n_classes):
        positive = np.where(preliminary[i] > 0)[0]
        if len(positive) == 0:
            continue
        ranked = positive[np.argsort(preliminary[i, positive])[::-1]]
        selected_indices.update(ranked[:TOP_CONCEPTS_PER_CLASS].tolist())

    if not selected_indices:
        raise RuntimeError("No class received any candidate WordNet concept.")

    selected_indices = list(selected_indices)

    if len(selected_indices) > MAX_CONCEPTS_TOTAL:
        global_scores = preliminary[:, selected_indices].sum(axis=0)
        keep_local = np.argsort(global_scores)[::-1][:MAX_CONCEPTS_TOTAL]
        selected_indices = [selected_indices[k] for k in keep_local]

    concept_names = [candidate_terms[j] for j in selected_indices]

    global_score_by_term = {
        term: float(preliminary[:, candidate_terms.index(term)].sum())
        for term in concept_names
    }
    concept_names = sorted(
        concept_names,
        key=lambda term: (-global_score_by_term[term], term),
    )

    B = np.zeros((n_classes, len(concept_names)), dtype=np.float32)

    for i, class_name in enumerate(class_names):
        for j, term in enumerate(concept_names):
            tf = class_term_scores[class_name].get(term, 0.0)
            if tf > 0:
                B[i, j] = math.log1p(tf) * idf[term]

    row_norms = np.linalg.norm(B, axis=1, keepdims=True)
    B = B / np.maximum(row_norms, 1e-8)

    concept_rows = []
    for j, term in enumerate(concept_names):
        relation_totals = Counter()
        present_classes = []

        for class_name in class_names:
            if class_term_scores[class_name].get(term, 0.0) > 0:
                present_classes.append(class_name)
                relation_totals.update(class_term_relations[class_name][term])

        dominant_relation = ""
        if relation_totals:
            dominant_relation = relation_totals.most_common(1)[0][0]

        concept_rows.append(
            {
                "concept": term,
                "document_frequency": document_frequency[term],
                "idf": idf[term],
                "global_weight": float(B[:, j].sum()),
                "dominant_relation": dominant_relation,
                "classes": ", ".join(present_classes),
            }
        )

    matrix_df = pd.DataFrame(B, index=class_names, columns=concept_names)
    concept_df = pd.DataFrame(concept_rows).sort_values(
        ["global_weight", "concept"], ascending=[False, True]
    )
    gloss_df = pd.DataFrame(gloss_rows)

    return B, concept_names, matrix_df, concept_df, gloss_df



def standardize_columns(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / np.maximum(std, 1e-8)


def assign_channels_to_classes(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    threshold: float = CLASS_SELECTIVITY_THRESHOLD,
):
    n_classes = len(class_names)
    n_channels = features.shape[1]
    class_means = np.zeros((n_classes, n_channels), dtype=np.float32)

    for class_idx in range(n_classes):
        class_means[class_idx] = features[labels == class_idx].mean(axis=0)

    sorted_classes = np.argsort(class_means, axis=0)
    top_class = sorted_classes[-1]
    second_class = sorted_classes[-2]
    channel_indices = np.arange(n_channels)

    top_mean = class_means[top_class, channel_indices]
    second_mean = class_means[second_class, channel_indices]
    selectivity = (top_mean - second_mean) / (
        class_means.std(axis=0) + 1e-8
    )

    assignments = np.full(n_channels, fill_value=-1, dtype=int)
    selected = selectivity >= threshold
    assignments[selected] = top_class[selected]

    return assignments, selectivity, class_means


def assign_channels_to_semantic_concepts(
    features: np.ndarray,
    labels: np.ndarray,
    class_concept_matrix: np.ndarray,
    min_correlation: float = CONCEPT_CORRELATION_THRESHOLD,
    min_margin: float = CONCEPT_CORRELATION_MARGIN,
):
    concept_targets = class_concept_matrix[labels]
    X = standardize_columns(features)
    Y = standardize_columns(concept_targets)

    corr = (X.T @ Y) / max(1, X.shape[0])
    corr = np.clip(corr, -1.0, 1.0)

    n_channels, n_concepts = corr.shape
    assignments = np.full(n_channels, fill_value=-1, dtype=int)

    if n_concepts == 0:
        return assignments, np.zeros(n_channels), np.zeros(n_channels), corr

    sorted_concepts = np.argsort(corr, axis=1)
    top_concept = sorted_concepts[:, -1]
    top_corr = corr[np.arange(n_channels), top_concept]

    if n_concepts >= 2:
        second_concept = sorted_concepts[:, -2]
        second_corr = corr[np.arange(n_channels), second_concept]
    else:
        second_corr = np.zeros(n_channels, dtype=np.float64)

    margin = top_corr - second_corr
    selected = (top_corr >= min_correlation) & (margin >= min_margin)
    assignments[selected] = top_concept[selected]

    return assignments, top_corr, margin, corr


def compute_group_weight_matrix(
    features: np.ndarray,
    assignments: np.ndarray,
    n_groups: int,
) -> np.ndarray:

    selected_channels = np.where(assignments >= 0)[0]
    W = np.zeros((n_groups, n_groups), dtype=np.float32)

    if len(selected_channels) < 2:
        return W

    X = standardize_columns(features[:, selected_channels])
    group_per_channel = assignments[selected_channels]

    corr = (X.T @ X) / max(1, X.shape[0])
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.maximum(corr, 0.0)

    for p in range(n_groups):
        idx_p = np.where(group_per_channel == p)[0]
        if len(idx_p) == 0:
            continue

        for q in range(p + 1, n_groups):
            idx_q = np.where(group_per_channel == q)[0]
            if len(idx_q) == 0:
                continue

            values = corr[np.ix_(idx_p, idx_q)].reshape(-1)
            if len(values) > 0:
                W[p, q] = float(values.mean())
                W[q, p] = W[p, q]

    np.fill_diagonal(W, 0.0)
    return W



def cosine_similarity_matrix(B: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    normalized = B / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices_from(matrix, k=1)
    return matrix[indices]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def neural_semantic_alignment(
    neural_matrix: np.ndarray,
    semantic_matrix: np.ndarray,
) -> float:
    return safe_spearman(
        upper_triangle_values(neural_matrix),
        upper_triangle_values(semantic_matrix),
    )


def permutation_alignment_baseline(
    neural_matrix: np.ndarray,
    semantic_matrix: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
):
    values = []
    n_classes = neural_matrix.shape[0]

    for _ in range(n_permutations):
        permutation = np.random.permutation(n_classes)
        permuted_semantic = semantic_matrix[np.ix_(permutation, permutation)]
        values.append(neural_semantic_alignment(neural_matrix, permuted_semantic))

    values = np.asarray(values, dtype=float)

    return {
        "alignment_random_mean": float(np.nanmean(values)),
        "alignment_random_std": float(np.nanstd(values)),
    }


def build_thresholded_graph_from_matrix(
    W: np.ndarray,
    node_names: Sequence[str],
    present_indices: Iterable[int] | None = None,
    edge_top_quantile: float = 0.70,
) -> nx.Graph:
    if present_indices is None:
        present_indices = range(len(node_names))

    present_indices = sorted(set(int(i) for i in present_indices))
    G = nx.Graph()

    for idx in present_indices:
        G.add_node(node_names[idx])

    weights = [
        float(W[i, j])
        for pos, i in enumerate(present_indices)
        for j in present_indices[pos + 1 :]
        if W[i, j] > 0
    ]

    if not weights:
        return G

    threshold = float(np.quantile(weights, edge_top_quantile))

    for pos, i in enumerate(present_indices):
        for j in present_indices[pos + 1 :]:
            if W[i, j] >= threshold and W[i, j] > 0:
                G.add_edge(node_names[i], node_names[j], weight=float(W[i, j]))

    return G


def compute_graph_modularity(G: nx.Graph):
    if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
        return 0.0, []

    communities = list(greedy_modularity_communities(G, weight="weight"))
    score = modularity(G, communities, weight="weight")
    return float(score), [sorted(list(c)) for c in communities]



def top_concepts_for_class(
    class_idx: int,
    B: np.ndarray,
    top_k: int,
) -> List[int]:
    positive = np.where(B[class_idx] > 0)[0]
    if len(positive) == 0:
        return []

    ranked = positive[np.argsort(B[class_idx, positive])[::-1]]
    return ranked[:top_k].tolist()


def select_graph_concepts(
    B: np.ndarray,
    concept_assignments: np.ndarray,
) -> List[int]:
    selected = set()

    for class_idx in range(B.shape[0]):
        selected.update(
            top_concepts_for_class(class_idx, B, TOP_CLASS_CONCEPT_EDGES)
        )

    active_concepts = set(concept_assignments[concept_assignments >= 0].tolist())
    selected.update(active_concepts)

    selected = list(selected)

    if len(selected) > MAX_CONCEPTS_IN_GRAPH:
        neural_counts = Counter(concept_assignments[concept_assignments >= 0].tolist())
        scores = []

        for concept_idx in selected:
            lexical_score = float(B[:, concept_idx].sum())
            neural_score = float(neural_counts.get(concept_idx, 0))
            scores.append((lexical_score + 0.05 * neural_score, concept_idx))

        scores.sort(reverse=True)
        selected = [concept_idx for _, concept_idx in scores[:MAX_CONCEPTS_IN_GRAPH]]

    return sorted(selected)


def matrix_edge_threshold(
    W: np.ndarray,
    indices: Sequence[int],
    quantile: float,
) -> float:
    weights = []

    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            if W[i, j] > 0:
                weights.append(float(W[i, j]))

    if not weights:
        return float("inf")

    return float(np.quantile(weights, quantile))


def build_heterogeneous_graph(
    class_names: Sequence[str],
    concept_names: Sequence[str],
    B: np.ndarray,
    class_neural_W: np.ndarray,
    concept_neural_W: np.ndarray,
    concept_assignments: np.ndarray,
    semantic_similarity: np.ndarray,
) -> nx.Graph:

    G = nx.Graph()

    graph_concepts = select_graph_concepts(B, concept_assignments)

    for class_idx, class_name in enumerate(class_names):
        G.add_node(
            f"class::{class_name}",
            label=class_name,
            node_type="class",
            class_index=int(class_idx),
        )

    for concept_idx in graph_concepts:
        concept_name = concept_names[concept_idx]
        G.add_node(
            f"concept::{concept_name}",
            label=concept_name,
            node_type="concept",
            concept_index=int(concept_idx),
        )

    graph_concept_set = set(graph_concepts)

    for class_idx, class_name in enumerate(class_names):
        for concept_idx in top_concepts_for_class(
            class_idx, B, TOP_CLASS_CONCEPT_EDGES
        ):
            if concept_idx not in graph_concept_set:
                continue

            weight = float(B[class_idx, concept_idx])
            if weight <= 0:
                continue

            G.add_edge(
                f"class::{class_name}",
                f"concept::{concept_names[concept_idx]}",
                weight=weight,
                edge_type="lexical",
                lexical_weight=weight,
            )

    class_indices = list(range(len(class_names)))
    class_threshold = matrix_edge_threshold(
        class_neural_W,
        class_indices,
        CLASS_EDGE_TOP_QUANTILE,
    )

    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            weight = float(class_neural_W[i, j])
            if weight <= 0 or weight < class_threshold:
                continue

            G.add_edge(
                f"class::{class_names[i]}",
                f"class::{class_names[j]}",
                weight=weight,
                edge_type="neural_class",
                neural_weight=weight,
                semantic_similarity=float(semantic_similarity[i, j]),
            )

    concept_threshold = matrix_edge_threshold(
        concept_neural_W,
        graph_concepts,
        CONCEPT_EDGE_TOP_QUANTILE,
    )

    for pos, i in enumerate(graph_concepts):
        for j in graph_concepts[pos + 1 :]:
            weight = float(concept_neural_W[i, j])
            if weight <= 0 or weight < concept_threshold:
                continue

            G.add_edge(
                f"concept::{concept_names[i]}",
                f"concept::{concept_names[j]}",
                weight=weight,
                edge_type="neural_concept",
                neural_weight=weight,
            )

    return G



def save_channel_assignments(
    layer_name: str,
    class_assignments: np.ndarray,
    class_selectivity: np.ndarray,
    concept_assignments: np.ndarray,
    concept_top_corr: np.ndarray,
    concept_margin: np.ndarray,
    concept_names: Sequence[str],
) -> None:
    rows = []

    for channel_idx in range(len(class_assignments)):
        class_idx = int(class_assignments[channel_idx])
        concept_idx = int(concept_assignments[channel_idx])

        rows.append(
            {
                "channel": channel_idx,
                "assigned_class": (
                    CIFAR10_CLASSES[class_idx] if class_idx >= 0 else ""
                ),
                "class_selectivity": float(class_selectivity[channel_idx]),
                "assigned_concept": (
                    concept_names[concept_idx] if concept_idx >= 0 else ""
                ),
                "concept_top_correlation": float(concept_top_corr[channel_idx]),
                "concept_correlation_margin": float(concept_margin[channel_idx]),
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, f"{layer_name}_channel_assignments.csv"),
        index=False,
    )


def plot_heterogeneous_graph(G: nx.Graph, title: str, path: str) -> None:
    plt.figure(figsize=(15, 11))

    if G.number_of_nodes() == 0:
        plt.title(f"{title} — empty graph")
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(
        G,
        seed=SEED,
        weight="weight",
        k=1.25 / math.sqrt(max(1, G.number_of_nodes())),
        iterations=350,
    )

    class_nodes = [
        node for node, data in G.nodes(data=True) if data["node_type"] == "class"
    ]
    concept_nodes = [
        node for node, data in G.nodes(data=True) if data["node_type"] == "concept"
    ]

    lexical_edges = [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data["edge_type"] == "lexical"
    ]
    class_edges = [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data["edge_type"] == "neural_class"
    ]
    concept_edges = [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data["edge_type"] == "neural_concept"
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=class_nodes,
        node_shape="o",
        node_size=1900,
        node_color="#f39c12",
        edgecolors="#6e4b00",
        linewidths=1.4,
        alpha=0.96,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=concept_nodes,
        node_shape="s",
        node_size=720,
        node_color="#85c1e9",
        edgecolors="#1f618d",
        linewidths=0.9,
        alpha=0.92,
    )

    if lexical_edges:
        lexical_widths = [
            0.7 + 4.0 * G[u][v]["weight"] for u, v in lexical_edges
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=lexical_edges,
            width=lexical_widths,
            edge_color="#9e9e9e",
            alpha=0.45,
        )

    if class_edges:
        class_widths = [
            1.5 + 12.0 * G[u][v]["weight"] for u, v in class_edges
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=class_edges,
            width=class_widths,
            edge_color="#c0392b",
            alpha=0.85,
        )

    if concept_edges:
        concept_widths = [
            0.8 + 9.0 * G[u][v]["weight"] for u, v in concept_edges
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=concept_edges,
            width=concept_widths,
            edge_color="#2471a3",
            style="dashed",
            alpha=0.72,
        )

    labels = {node: data["label"] for node, data in G.nodes(data=True)}

    class_labels = {node: labels[node] for node in class_nodes}
    concept_labels = {node: labels[node] for node in concept_nodes}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=class_labels,
        font_size=9,
        font_weight="bold",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=concept_labels,
        font_size=7,
    )

    legend_elements = [
        Patch(facecolor="#f39c12", edgecolor="#6e4b00", label="CIFAR-10 class"),
        Patch(facecolor="#85c1e9", edgecolor="#1f618d", label="WordNet concept"),
        Line2D([0], [0], color="#9e9e9e", lw=2, label="Lexical class–concept"),
        Line2D([0], [0], color="#c0392b", lw=3, label="Neural class–class"),
        Line2D(
            [0],
            [0],
            color="#2471a3",
            lw=2,
            linestyle="--",
            label="Neural concept–concept",
        ),
    ]

    plt.legend(handles=legend_elements, loc="upper left", frameon=True)
    plt.title(title, fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_alignment(results_df: pd.DataFrame, path: str) -> None:
    plt.figure(figsize=(8, 5))

    x = np.arange(len(results_df))
    width = 0.36

    plt.bar(
        x - width / 2,
        results_df["neural_semantic_alignment"],
        width,
        label="real alignment",
    )
    plt.bar(
        x + width / 2,
        results_df["alignment_random_mean"],
        width,
        label="permuted baseline",
    )

    plt.errorbar(
        x + width / 2,
        results_df["alignment_random_mean"],
        yerr=results_df["alignment_random_std"],
        fmt="none",
        capsize=4,
        color="black",
    )

    plt.axhline(0.0, linewidth=0.8, color="black")
    plt.xticks(x, results_df["layer"])
    plt.ylabel("Spearman correlation")
    plt.xlabel("ResNet-18 layer")
    plt.title("Neural class interactions vs WordNet semantic similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

ACTIVATIONS_CACHE = os.path.join(
    OUTPUT_DIR,
    f"activations_{N_IMAGES}_seed_{SEED}.npz",
)


def run_experiment() -> None:
    print("Device:", DEVICE)
    print("Building WordNet class-concept matrix...")

    (
        class_concept_matrix,
        concept_names,
        concept_matrix_df,
        concept_df,
        gloss_df,
    ) = build_wordnet_concept_matrix(CIFAR10_CLASSES, CLASS_SYNSETS)

    concept_matrix_path = os.path.join(
        OUTPUT_DIR, "wordnet_class_concept_matrix.csv"
    )
    concepts_path = os.path.join(OUTPUT_DIR, "wordnet_concepts.csv")
    glosses_path = os.path.join(OUTPUT_DIR, "wordnet_extended_glosses.csv")

    concept_matrix_df.to_csv(concept_matrix_path)
    concept_df.to_csv(concepts_path, index=False)
    gloss_df.to_csv(glosses_path, index=False)

    semantic_similarity = cosine_similarity_matrix(class_concept_matrix)
    semantic_similarity_df = pd.DataFrame(
        semantic_similarity,
        index=CIFAR10_CLASSES,
        columns=CIFAR10_CLASSES,
    )
    semantic_similarity_df.to_csv(
        os.path.join(OUTPUT_DIR, "wordnet_class_semantic_similarity.csv")
    )

    print(f"Retained WordNet concepts: {len(concept_names)}")
    print("Top concepts:", concept_df["concept"].head(20).tolist())

    model, transform = load_model()

    if os.path.exists(ACTIVATIONS_CACHE):
        print("Loading cached activations:", ACTIVATIONS_CACHE)
        cached = np.load(ACTIVATIONS_CACHE)
        activations = {
            "layer1": cached["layer1"],
            "layer2": cached["layer2"],
            "layer3": cached["layer3"],
            "layer4": cached["layer4"],
        }
        labels = cached["labels"]
    else:
        loader = load_cifar10(transform)
        activations, labels = extract_layer_activations(model, loader)

        np.savez_compressed(
            ACTIVATIONS_CACHE,
            labels=labels,
            **activations,
        )
        print("Saved activations cache:", ACTIVATIONS_CACHE)

    results = []

    for layer_name, features in activations.items():
        print(f"\n=== Processing {layer_name} ===")
        print("Feature matrix:", features.shape)

        (
            class_assignments,
            class_selectivity,
            _class_means,
        ) = assign_channels_to_classes(
            features,
            labels,
            CIFAR10_CLASSES,
            threshold=CLASS_SELECTIVITY_THRESHOLD,
        )

        class_neural_W = compute_group_weight_matrix(
            features,
            class_assignments,
            n_groups=len(CIFAR10_CLASSES),
        )

        (
            concept_assignments,
            concept_top_corr,
            concept_margin,
            _channel_concept_corr,
        ) = assign_channels_to_semantic_concepts(
            features,
            labels,
            class_concept_matrix,
            min_correlation=CONCEPT_CORRELATION_THRESHOLD,
            min_margin=CONCEPT_CORRELATION_MARGIN,
        )

        concept_neural_W = compute_group_weight_matrix(
            features,
            concept_assignments,
            n_groups=len(concept_names),
        )

        alignment_real = neural_semantic_alignment(
            class_neural_W,
            semantic_similarity,
        )
        baseline = permutation_alignment_baseline(
            class_neural_W,
            semantic_similarity,
            n_permutations=N_PERMUTATIONS,
        )

        present_class_indices = sorted(
            set(class_assignments[class_assignments >= 0].tolist())
        )
        class_graph = build_thresholded_graph_from_matrix(
            class_neural_W,
            CIFAR10_CLASSES,
            present_indices=present_class_indices,
            edge_top_quantile=CLASS_EDGE_TOP_QUANTILE,
        )
        class_modularity, class_communities = compute_graph_modularity(class_graph)

        heterogeneous_graph = build_heterogeneous_graph(
            CIFAR10_CLASSES,
            concept_names,
            class_concept_matrix,
            class_neural_W,
            concept_neural_W,
            concept_assignments,
            semantic_similarity,
        )

        graph_png_path = os.path.join(
            OUTPUT_DIR, f"{layer_name}_heterogeneous_graph.png"
        )
        graph_graphml_path = os.path.join(
            OUTPUT_DIR, f"{layer_name}_heterogeneous_graph.graphml"
        )

        plot_heterogeneous_graph(
            heterogeneous_graph,
            title=f"CIFAR-10 classes and WordNet concepts — {layer_name}",
            path=graph_png_path,
        )
        nx.write_graphml(heterogeneous_graph, graph_graphml_path)

        pd.DataFrame(
            class_neural_W,
            index=CIFAR10_CLASSES,
            columns=CIFAR10_CLASSES,
        ).to_csv(
            os.path.join(OUTPUT_DIR, f"{layer_name}_class_neural_matrix.csv")
        )

        pd.DataFrame(
            concept_neural_W,
            index=concept_names,
            columns=concept_names,
        ).to_csv(
            os.path.join(OUTPUT_DIR, f"{layer_name}_concept_neural_matrix.csv")
        )

        save_channel_assignments(
            layer_name,
            class_assignments,
            class_selectivity,
            concept_assignments,
            concept_top_corr,
            concept_margin,
            concept_names,
        )

        n_class_assigned = int(np.sum(class_assignments >= 0))
        n_concept_assigned = int(np.sum(concept_assignments >= 0))
        n_active_concepts = len(
            set(concept_assignments[concept_assignments >= 0].tolist())
        )

        print(f"Class-assigned channels: {n_class_assigned}/{features.shape[1]}")
        print(f"Concept-assigned channels: {n_concept_assigned}/{features.shape[1]}")
        print("Active neural concepts:", n_active_concepts)
        print("Neural-semantic alignment:", alignment_real)
        print(
            "Random alignment:",
            baseline["alignment_random_mean"],
            "+/-",
            baseline["alignment_random_std"],
        )
        print("Class modularity:", class_modularity)
        print("Class communities:", class_communities)

        results.append(
            {
                "layer": layer_name,
                "n_channels": features.shape[1],
                "n_class_assigned_channels": n_class_assigned,
                "n_concept_assigned_channels": n_concept_assigned,
                "n_active_neural_concepts": n_active_concepts,
                "n_heterogeneous_nodes": heterogeneous_graph.number_of_nodes(),
                "n_heterogeneous_edges": heterogeneous_graph.number_of_edges(),
                "class_graph_modularity": class_modularity,
                "neural_semantic_alignment": alignment_real,
                "alignment_random_mean": baseline["alignment_random_mean"],
                "alignment_random_std": baseline["alignment_random_std"],
                "class_communities": str(class_communities),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "heterogeneous_graph_results.csv"),
        index=False,
    )

    plot_alignment(
        results_df,
        os.path.join(OUTPUT_DIR, "neural_semantic_alignment_by_layer.png"),
    )

    print("\nSaved all results to:", OUTPUT_DIR)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
