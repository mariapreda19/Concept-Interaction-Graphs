import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw

seed = 42
image_size = 32
n_per_combination = 100
output_dir = Path('/kaggle/working/synthetic_cig_exact')
output_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(seed)
concepts = ('A', 'B', 'C')
tasks = ('ADD', 'AND', 'XOR')
colors = {'A': '#E74C3C', 'B': '#3498DB', 'C': '#F1C40F'}
expected = {'ADD': 0.0, 'AND': 3.0, 'XOR': -6.0}
combinations = list(itertools.product((0, 1), repeat=3))
pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]


def score(task, a, b):
    if task == 'ADD':
        return -1.0 + 1.5*a + 1.5*b
    if task == 'AND':
        return -1.0 + 3.0*a*b
    if task == 'XOR':
        return -1.0 + 3.0*(a+b-2*a*b)
    raise ValueError(task)


def label(task, a, b):
    return int(score(task, a, b) > 0)


def sample_shape_parameters():
    sizes = {}
    for concept in concepts:
        sizes[concept] = int(rng.integers(5, 10))
    regions = rng.permutation(3)
    params = {}
    for concept, region in zip(concepts, regions):
        size = sizes[concept]
        x_min = int(region)*11
        x_max = min(x_min+10, image_size)
        x = int(rng.integers(x_min, x_max-size+1))
        y = int(rng.integers(0, image_size-size+1))
        params[concept] = (x, y, size)
    return params


def render_image(bits, params):
    image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for concept, present in zip(concepts, bits):
        if not present:
            continue
        x, y, size = params[concept]
        if concept == 'A':
            draw.ellipse((x, y, x+size-1, y+size-1), fill=(255, 0, 0))
        elif concept == 'B':
            draw.rectangle((x, y, x+size-1, y+size-1), fill=(0, 0, 255))
        elif concept == 'C':
            draw.polygon([(x+size//2, y), (x, y+size-1), (x+size-1, y+size-1)], fill=(255, 255, 0))
    return np.asarray(image, dtype=np.uint8).copy()


def find_example(bits):
    for item in dataset:
        if item['bits'] == bits:
            return item
    raise ValueError(bits)


def intervene(item, removed):
    bits = list(item['bits'])
    for concept in removed:
        bits[concepts.index(concept)] = 0
    bits = tuple(bits)
    image = render_image(bits, item['params'])
    return bits, image


def evaluate_score(task, bits):
    a, b, c = bits
    return score(task, a, b)


def delta(task, item, removed):
    bits, image = intervene(item, removed)
    return evaluate_score(task, bits) - evaluate_score(task, item['bits'])


def interaction(task, item, pair):
    i, j = pair
    return delta(task, item, (i, j)) - delta(task, item, (i,)) - delta(task, item, (j,))


def draw_graph(task, ax, summary):
    graph = nx.Graph()
    graph.add_nodes_from(concepts)
    task_summary = summary[summary['task'] == task]
    for _, row in task_summary.iterrows():
        weight = row['mean']
        if not np.isclose(weight, 0.0):
            pair = row['pair']
            graph.add_edge(pair[0], pair[1], weight=weight)
    positions = {'A': (-1, 0), 'B': (1, 0), 'C': (0, -1.2)}
    node_colors = []
    for node in graph.nodes():
        node_colors.append(colors[node])
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=1900, node_color=node_colors, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=15, font_weight='bold')
    positive_edges = []
    negative_edges = []
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        if data['weight'] > 0:
            positive_edges.append((u, v))
        if data['weight'] < 0:
            negative_edges.append((u, v))
        edge_labels[(u, v)] = f"{data['weight']:+.1f}"
    nx.draw_networkx_edges(graph, positions, edgelist=positive_edges, edge_color='green', width=3, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=negative_edges, edge_color='red', width=3, ax=ax)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=12, ax=ax)
    ax.set_title(task, fontsize=15)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 0.8)
    ax.axis('off')


dataset = []
for bits in combinations:
    for repetition in range(n_per_combination):
        params = sample_shape_parameters()
        image = render_image(bits, params)
        dataset.append({'bits': tuple(bits), 'params': params, 'image': image})

print('Number of images:', len(dataset))
print('Image shape:', dataset[0]['image'].shape)
print('Number of concept combinations:', len(combinations))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, bits in zip(axes.flat, combinations):
    example = find_example(bits)
    ax.imshow(example['image'])
    ax.set_title(f'A={bits[0]}, B={bits[1]}, C={bits[2]}')
    ax.axis('off')
plt.suptitle('Synthetic images — all 8 concept combinations')
plt.tight_layout()
plt.savefig(output_dir / 'synthetic_images.png', dpi=200)
plt.show()

reference_item = find_example((1, 1, 0))
theoretical_rows = []
for task in tasks:
    row = {
        'Task': task,
        'm(1,1)': evaluate_score(task, (1, 1, 0)),
        'm(0,1)': evaluate_score(task, (0, 1, 0)),
        'm(1,0)': evaluate_score(task, (1, 0, 0)),
        'm(0,0)': evaluate_score(task, (0, 0, 0)),
        'Delta(A)': delta(task, reference_item, ('A',)),
        'Delta(B)': delta(task, reference_item, ('B',)),
        'Delta(A,B)': delta(task, reference_item, ('A', 'B')),
        'I(A,B)': interaction(task, reference_item, ('A', 'B'))
    }
    theoretical_rows.append(row)
theoretical_df = pd.DataFrame(theoretical_rows)
print('\nTHEORETICAL TABLE FOR x = (1,1,0)')
print(theoretical_df.to_string(index=False))
theoretical_df.to_csv(output_dir / 'theoretical_table.csv', index=False)

rows = []
for task in tasks:
    for image_id, item in enumerate(dataset):
        bits = item['bits']
        row = {'task': task, 'image_id': image_id, 'A': bits[0], 'B': bits[1], 'C': bits[2], 'score': evaluate_score(task, bits), 'label': label(task, bits[0], bits[1])}
        for concept in concepts:
            row[f'delta_{concept}'] = delta(task, item, (concept,))
        for i, j in pairs:
            row[f'I_{i}{j}'] = interaction(task, item, (i, j))
        rows.append(row)
results_df = pd.DataFrame(rows)
results_df.to_csv(output_dir / 'all_interactions.csv', index=False)

summary_rows = []
for task in tasks:
    task_df = results_df[results_df['task'] == task]
    for i, j in pairs:
        subset = task_df[(task_df[i] == 1) & (task_df[j] == 1)]
        values = subset[f'I_{i}{j}'].to_numpy()
        summary_rows.append({'task': task, 'pair': i+j, 'mean': float(np.mean(values)), 'std': float(np.std(values)), 'min': float(np.min(values)), 'max': float(np.max(values)), 'n': len(values)})
summary_df = pd.DataFrame(summary_rows)
print('\nINTERACTION SUMMARY')
print(summary_df.to_string(index=False))
summary_df.to_csv(output_dir / 'interaction_summary.csv', index=False)

for task in tasks:
    task_summary = summary_df[summary_df['task'] == task]
    ab = task_summary[task_summary['pair'] == 'AB']['mean'].iloc[0]
    ac = task_summary[task_summary['pair'] == 'AC']['mean'].iloc[0]
    bc = task_summary[task_summary['pair'] == 'BC']['mean'].iloc[0]
    assert np.isclose(ab, expected[task]), f'{task}: incorrect AB interaction'
    assert np.isclose(ac, 0.0), f'{task}: incorrect AC interaction'
    assert np.isclose(bc, 0.0), f'{task}: incorrect BC interaction'
    print(f'{task}: I(A,B)={ab:+.1f}, I(A,C)={ac:+.1f}, I(B,C)={bc:+.1f} — PASS')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, task in zip(axes, tasks):
    draw_graph(task, ax, summary_df)
plt.suptitle('Exact Synthetic Concept Interaction Graphs', fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / 'exact_concept_interaction_graphs.png', dpi=200)
plt.show()

example = find_example((1, 1, 1))
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
interventions = [(), ('A',), ('B',), ('A', 'B')]
titles = ['Original', 'Remove A', 'Remove B', 'Remove A+B']
for ax, removed, title in zip(axes, interventions, titles):
    bits, modified_image = intervene(example, removed)
    ax.imshow(modified_image)
    ax.set_title(title)
    ax.axis('off')
plt.suptitle('Exact concept interventions')
plt.tight_layout()
plt.savefig(output_dir / 'intervention_examples.png', dpi=200)
plt.show()

final_table = summary_df.pivot(index='pair', columns='task', values='mean')
final_table = final_table[['ADD', 'AND', 'XOR']]
print('\nFINAL RESULTS')
print('\nAll theoretical checks passed.')
print('\nSaved files:')