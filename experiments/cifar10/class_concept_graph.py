import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

SEED = 42
BATCH_SIZE = 64
N_IMAGES = 5000
N_PERMUTATIONS = 100

SELECTIVITY_THRESHOLD = 0.35
EDGE_TOP_QUANTILE = 0.70

OUTPUT_DIR = "outputs_concept_graphs"
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

ANIMALS = {"bird", "cat", "deer", "dog", "frog", "horse"}
VEHICLES = {"airplane", "automobile", "ship", "truck"}
DATA_DIR = "./data"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model.to(DEVICE)
    transform = weights.transforms()
    return model, transform



def load_cifar10(transform):
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
        def hook(module, inp, out):]
            pooled = out.detach().mean(dim=(2, 3)).cpu().numpy()
            activations[layer_name].append(pooled)
        return hook

    for name in layer_names:
        layer = getattr(model, name)
        hooks.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for images, y in tqdm(loader, desc="Extracting activations"):
            images = images.to(DEVICE)
            _ = model(images)
            labels.append(y.numpy())

    for h in hooks:
        h.remove()

    labels = np.concatenate(labels, axis=0)

    for name in layer_names:
        activations[name] = np.concatenate(activations[name], axis=0)

    return activations, labels


def assign_channels_to_concepts(features, labels, class_names, threshold=0.35):
    n_classes = len(class_names)
    n_channels = features.shape[1]

    class_means = np.zeros((n_classes, n_channels), dtype=np.float32)

    for k in range(n_classes):
        class_means[k] = features[labels == k].mean(axis=0)

    sorted_classes = np.argsort(class_means, axis=0)

    top_class = sorted_classes[-1]
    second_class = sorted_classes[-2]

    channel_indices = np.arange(n_channels)

    top_mean = class_means[top_class, channel_indices]
    second_mean = class_means[second_class, channel_indices]

    selectivity = (top_mean - second_mean) / (class_means.std(axis=0) + 1e-8)

    assignments = np.full(n_channels, fill_value=-1, dtype=int)
    assignments[selectivity >= threshold] = top_class[selectivity >= threshold]

    return assignments, selectivity, class_means


def compute_concept_weight_matrix(features, assignments, n_classes):

    selected_channels = np.where(assignments >= 0)[0]

    W = np.zeros((n_classes, n_classes), dtype=np.float32)

    if len(selected_channels) < 2:
        return W

    X = features[:, selected_channels]
    concept_per_selected_channel = assignments[selected_channels]

    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    corr = (X.T @ X) / max(1, X.shape[0] - 1)

    corr = np.maximum(corr, 0.0)

    for p in range(n_classes):
        idx_p = np.where(concept_per_selected_channel == p)[0]
        if len(idx_p) == 0:
            continue

        for q in range(n_classes):
            if p == q:
                continue

            idx_q = np.where(concept_per_selected_channel == q)[0]
            if len(idx_q) == 0:
                continue

            vals = corr[np.ix_(idx_p, idx_q)].reshape(-1)
            W[p, q] = vals.mean()

    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)

    return W


def build_thresholded_graph(W, class_names, assignments, edge_top_quantile=0.70):
    present_concepts = sorted(set(assignments[assignments >= 0].tolist()))

    G = nx.Graph()

    for c in present_concepts:
        G.add_node(class_names[c])

    weights = []

    for i in present_concepts:
        for j in present_concepts:
            if i < j and W[i, j] > 0:
                weights.append(W[i, j])

    if len(weights) == 0:
        return G

    threshold = np.quantile(weights, edge_top_quantile)

    for i in present_concepts:
        for j in present_concepts:
            if i < j and W[i, j] >= threshold and W[i, j] > 0:
                G.add_edge(class_names[i], class_names[j], weight=float(W[i, j]))

    return G

def compute_graph_modularity(G):
    if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
        return 0.0, []

    communities = list(greedy_modularity_communities(G, weight="weight"))
    Q = modularity(G, communities, weight="weight")

    communities_as_lists = [sorted(list(c)) for c in communities]
    return float(Q), communities_as_lists


def semantic_group_ratio(W, class_names, assignments):
    present = sorted(set(assignments[assignments >= 0].tolist()))

    inside = []
    outside = []

    for i in present:
        for j in present:
            if i >= j:
                continue

            ci = class_names[i]
            cj = class_names[j]

            same_animal = ci in ANIMALS and cj in ANIMALS
            same_vehicle = ci in VEHICLES and cj in VEHICLES

            if same_animal or same_vehicle:
                inside.append(W[i, j])
            else:
                outside.append(W[i, j])

    if len(inside) == 0 or len(outside) == 0:
        return np.nan

    return float(np.mean(inside) / (np.mean(outside) + 1e-8))


def permutation_baseline(features, assignments, class_names, n_permutations=100):

    n_classes = len(class_names)

    Q_values = []
    ratio_values = []

    selected = assignments >= 0
    selected_values = assignments[selected].copy()

    for _ in range(n_permutations):
        permuted = assignments.copy()
        shuffled_values = selected_values.copy()
        np.random.shuffle(shuffled_values)
        permuted[selected] = shuffled_values

        W_perm = compute_concept_weight_matrix(features, permuted, n_classes)
        G_perm = build_thresholded_graph(W_perm, class_names, permuted, EDGE_TOP_QUANTILE)

        Q_perm, _ = compute_graph_modularity(G_perm)
        ratio_perm = semantic_group_ratio(W_perm, class_names, permuted)

        Q_values.append(Q_perm)
        ratio_values.append(ratio_perm)

    return {
        "Q_random_mean": float(np.nanmean(Q_values)),
        "Q_random_std": float(np.nanstd(Q_values)),
        "semantic_ratio_random_mean": float(np.nanmean(ratio_values)),
        "semantic_ratio_random_std": float(np.nanstd(ratio_values)),
    }

def plot_graph(G, title, path):
    plt.figure(figsize=(8, 6))

    if G.number_of_nodes() == 0:
        plt.title(title + " empty graph")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(G, seed=SEED, weight="weight")

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [1.0 + 8.0 * w for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=1200)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title(title)
    plt.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_bar(results_df, metric_real, metric_random, ylabel, path):
    plt.figure(figsize=(8, 5))

    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(x - width / 2, results_df[metric_real], width, label="real")
    plt.bar(x + width / 2, results_df[metric_random], width, label="random baseline")

    plt.xticks(x, results_df["layer"])
    plt.ylabel(ylabel)
    plt.xlabel("Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

ACTIVATIONS_CACHE = os.path.join(
    OUTPUT_DIR,
    f"activations_{N_IMAGES}_seed_{SEED}.npz",
)

def run_experiment():
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

        assignments, selectivity, class_means = assign_channels_to_concepts(
            features,
            labels,
            CIFAR10_CLASSES,
            threshold=SELECTIVITY_THRESHOLD,
        )

        n_assigned = int(np.sum(assignments >= 0))
        present_concepts = sorted(set(assignments[assignments >= 0].tolist()))

        print(f"Assigned channels: {n_assigned}/{features.shape[1]}")
        print("Present concepts:", [CIFAR10_CLASSES[c] for c in present_concepts])

        W = compute_concept_weight_matrix(
            features,
            assignments,
            n_classes=len(CIFAR10_CLASSES),
        )

        G = build_thresholded_graph(
            W,
            CIFAR10_CLASSES,
            assignments,
            edge_top_quantile=EDGE_TOP_QUANTILE,
        )

        Q_real, communities = compute_graph_modularity(G)
        ratio_real = semantic_group_ratio(W, CIFAR10_CLASSES, assignments)

        baseline = permutation_baseline(
            features,
            assignments,
            CIFAR10_CLASSES,
            n_permutations=N_PERMUTATIONS,
        )

        print("Modularity real:", Q_real)
        print("Modularity random:", baseline["Q_random_mean"], "+/-", baseline["Q_random_std"])
        print("Semantic ratio real:", ratio_real)
        print("Semantic ratio random:", baseline["semantic_ratio_random_mean"], "+/-", baseline["semantic_ratio_random_std"])
        print("Communities:", communities)

        graph_path = os.path.join(OUTPUT_DIR, f"{layer_name}_concept_graph.png")
        plot_graph(
            G,
            title=f"Concept Interaction Graph - {layer_name}",
            path=graph_path,
        )

        results.append({
            "layer": layer_name,
            "n_channels": features.shape[1],
            "n_assigned_channels": n_assigned,
            "n_present_concepts": len(present_concepts),
            "n_edges": G.number_of_edges(),
            "Q_real": Q_real,
            "Q_random_mean": baseline["Q_random_mean"],
            "Q_random_std": baseline["Q_random_std"],
            "semantic_ratio_real": ratio_real,
            "semantic_ratio_random_mean": baseline["semantic_ratio_random_mean"],
            "semantic_ratio_random_std": baseline["semantic_ratio_random_std"],
            "communities": str(communities),
        })

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(OUTPUT_DIR, "concept_graph_results.csv")
    results_df.to_csv(csv_path, index=False)

    plot_metric_bar(
        results_df,
        metric_real="Q_real",
        metric_random="Q_random_mean",
        ylabel="Graph modularity",
        path=os.path.join(OUTPUT_DIR, "modularity_by_layer.png"),
    )

    plot_metric_bar(
        results_df,
        metric_real="semantic_ratio_real",
        metric_random="semantic_ratio_random_mean",
        ylabel="Semantic in-group / out-group ratio",
        path=os.path.join(OUTPUT_DIR, "semantic_ratio_by_layer.png"),
    )

    print("\nSaved results to:", OUTPUT_DIR)
    print(results_df)


if __name__ == "__main__":
    run_experiment()