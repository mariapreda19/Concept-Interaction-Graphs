import os, random, json, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import networkx as nx

SEED = 42
ROOT = Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird')
OUT = Path('/kaggle/working/waterbirds_real1')
SIZE = 224
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-4
WORKERS = 2
MAX_EVAL_PER_SPLIT = None
CHECKPOINT = None
MASK_MODE = 'auto'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if not (ROOT / 'metadata.csv').exists():
    candidates = list(Path('/kaggle/input').rglob('metadata.csv'))
    candidates = [p for p in candidates if 'waterbird' in str(p).lower()]
    if not candidates:
        raise FileNotFoundError('Waterbirds metadata.csv not found')
    ROOT = candidates[0].parent
meta = pd.read_csv(ROOT / 'metadata.csv').copy()
for col in ('img_filename', 'y', 'place', 'split'):
    if col not in meta:
        raise ValueError('Missing metadata column: ' + col)
meta['path'] = meta['img_filename'].map(lambda p: str(ROOT / str(p)))
if not Path(meta.iloc[0]['path']).exists():
    alt = ROOT / 'dataset'
    if (alt / str(meta.iloc[0]['img_filename'])).exists():
        ROOT = alt
        meta['path'] = meta['img_filename'].map(lambda p: str(ROOT / str(p)))
if not all(os.path.isfile(p) for p in meta['path']):
    raise FileNotFoundError('Not all images referenced by metadata exist')
meta['y'] = meta['y'].astype(int)
meta['place'] = meta['place'].astype(int)
meta['split'] = meta['split'].astype(int)
print('Device:', DEVICE, '| Dataset:', ROOT, '| Images:', len(meta))
print(pd.crosstab([meta['split'], meta['y']], meta['place']))

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_transform = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
eval_transform = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(mean, std)])

class BirdDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(row['y'])

train_df = meta[meta.split == 0].reset_index(drop=True)
val_df = meta[meta.split == 1].reset_index(drop=True)
test_df = meta[meta.split == 2].reset_index(drop=True)
train_loader = DataLoader(BirdDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=DEVICE.type == 'cuda')
val_loader = DataLoader(BirdDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

@torch.no_grad()
def basic_evaluate(loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += len(labels)
    return correct / max(total, 1)

checkpoint_path = OUT / 'resnet18_waterbirds.pt'
if CHECKPOINT is not None:
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    print('Loaded existing binary Waterbirds checkpoint:', CHECKPOINT)
elif checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    print('Loaded checkpoint from previous run')
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        val_accuracy = basic_evaluate(val_loader)
        print('Epoch', epoch + 1, '| train loss', round(loss_sum / len(train_df), 4), '| val accuracy', round(val_accuracy, 4))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

segmenter = None
mask_method = None
if MASK_MODE == 'auto':
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        segmenter = deeplabv3_resnet50(weights=weights).to(DEVICE).eval()
        mask_method = 'DeepLabV3 VOC bird mask (approximate)'
        print('Segmentation:', mask_method)
    except Exception as exc:
        print('Pretrained segmentation weights unavailable:', str(exc)[:180])
        print('Falling back to GrabCut; this is a weaker approximation.')
if segmenter is None:
    mask_method = 'GrabCut central rectangle (approximate)'

seg_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def bird_mask(rgb):
    if segmenter is not None:
        image = Image.fromarray(rgb).resize((SIZE, SIZE))
        tensor = seg_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prediction = segmenter(tensor)['out'].argmax(1)[0].cpu().numpy()
        mask = (prediction == 3).astype(np.uint8)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if mask.mean() > 0.005 and mask.mean() < 0.8:
            return mask
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, (int(SIZE*.1), int(SIZE*.1), int(SIZE*.8), int(SIZE*.8)), bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    except cv2.error:
        mask[:] = 0
    if mask.mean() < 0.005 or mask.mean() > 0.85:
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.ellipse(mask, (SIZE//2, SIZE//2), (SIZE//4, SIZE//3), 0, 0, 360, 1, -1)
    return mask

def read_rgb(path):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((SIZE, SIZE))
        return np.array(image)

def make_interventions(rgb, mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 1.5)[..., None]
    reference = np.full_like(rgb, 127)
    background_changed = np.uint8(np.clip(alpha * rgb + (1 - alpha) * reference, 0, 255))
    inpaint_mask = (mask * 255).astype(np.uint8)
    object_removed = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), inpaint_mask, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    both_removed = reference.copy()
    return object_removed, background_changed, both_removed

def normalize_batch(images):
    tensors = []
    for image in images:
        tensors.append(eval_transform(Image.fromarray(image)))
    return torch.stack(tensors)

@torch.no_grad()
def margins(images):
    logits = model(normalize_batch(images).to(DEVICE))
    return (logits[:, 1] - logits[:, 0]).cpu().numpy()

def choose_subset(frame, limit):
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True)
    groups = []
    for _, group in frame.groupby(['y', 'place']):
        n = max(1, round(limit * len(group) / len(frame)))
        groups.append(group.sample(min(n, len(group)), random_state=SEED))
    return pd.concat(groups).sample(frac=1, random_state=SEED).reset_index(drop=True)

def run_interventions(frame, name):
    frame = choose_subset(frame, MAX_EVAL_PER_SPLIT)
    records = []
    examples = []
    for index, row in frame.iterrows():
        rgb = read_rgb(row['path'])
        mask = bird_mask(rgb)
        without_bird, without_bg, without_both = make_interventions(rgb, mask)
        m, m_o, m_b, m_ob = margins([rgb, without_bird, without_bg, without_both])
        prediction = int(m > 0)
        sign = 2 * prediction - 1
        delta_o = float(m_o - m)
        delta_b = float(m_b - m)
        delta_ob = float(m_ob - m)
        b_o = -sign * delta_o
        b_b = -sign * delta_b
        interaction = delta_ob - delta_o - delta_b
        records.append({'img_filename': row['img_filename'], 'y': int(row['y']), 'place': int(row['place']), 'pred': prediction, 'error': int(prediction != row['y']), 'm': float(m), 'confidence': float(abs(m)), 'm_without_bird': float(m_o), 'm_without_background': float(m_b), 'm_without_both': float(m_ob), 'delta_o': delta_o, 'delta_b': delta_b, 'delta_ob': delta_ob, 'B_o': float(b_o), 'B_b': float(b_b), 'I_ob': float(interaction), 'mask_fraction': float(mask.mean()), 'background_support_ratio': float(max(b_b, 0) / (max(b_b, 0) + max(b_o, 0) + 1e-8)), 'semantic_mismatch': float(max(b_b, 0) + max(-b_o, 0))})
        if len(examples) < 8:
            examples.append((rgb, without_bird, without_bg, without_both, mask, row['img_filename']))
        if (index + 1) % 100 == 0:
            print(name, index + 1, '/', len(frame))
    result = pd.DataFrame(records)
    result.to_csv(OUT / (name + '_interventions.csv'), index=False)
    fig, axes = plt.subplots(len(examples), 5, figsize=(13, 2.7 * len(examples)))
    for i, (original, no_bird, no_bg, no_both, mask, filename) in enumerate(examples):
        for j, image in enumerate((original, mask, no_bird, no_bg, no_both)):
            axes[i, j].imshow(image, cmap='gray' if j == 1 else None)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(Path(filename).name[:22])
    for ax, title in zip(axes[0], ('Original', 'Estimated bird mask', 'Remove bird', 'Replace background', 'Remove both')):
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / (name + '_intervention_examples.png'), dpi=140)
    plt.show()
    plt.close(fig)
    return result

validation = run_interventions(val_df, 'validation')
testing = run_interventions(test_df, 'test')

features = {
    'confidence': ['confidence'],
    'cnn_graph': ['confidence', 'B_o', 'B_b', 'I_ob'],
    'semantic_rule': ['confidence', 'background_support_ratio', 'semantic_mismatch'],
    'combined': ['confidence', 'B_o', 'B_b', 'I_ob', 'background_support_ratio', 'semantic_mismatch']
}
metrics = []
for name, columns in features.items():
    if validation['error'].nunique() < 2:
        print('Skipping detectors: validation has only one error class')
        break
    detector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))
    detector.fit(validation[columns], validation['error'])
    scores = detector.predict_proba(testing[columns])[:, 1]
    testing['risk_' + name] = scores
    if testing['error'].nunique() == 2:
        auroc = roc_auc_score(testing['error'], scores)
        auprc = average_precision_score(testing['error'], scores)
    else:
        auroc = float('nan')
        auprc = float('nan')
    metrics.append({'detector': name, 'AUROC': auroc, 'AUPRC': auprc})
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT / 'detector_metrics.csv', index=False)
testing.to_csv(OUT / 'test_with_risk_scores.csv', index=False)
print('\nERROR DETECTION')
print(metrics_df.to_string(index=False))

rows = []
for split_name, frame in (('validation', validation), ('test', testing)):
    for (y, place), group in frame.groupby(['y', 'place']):
        rows.append({'split': split_name, 'bird': int(y), 'background': int(place), 'n': len(group), 'accuracy': 1 - group['error'].mean(), 'mean_B_o': group['B_o'].mean(), 'mean_B_b': group['B_b'].mean(), 'mean_I_ob': group['I_ob'].mean(), 'mean_semantic_mismatch': group['semantic_mismatch'].mean()})
groups_df = pd.DataFrame(rows)
groups_df.to_csv(OUT / 'group_results.csv', index=False)
print('\nGROUP RESULTS')
print(groups_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes, ('B_o', 'B_b', 'I_ob'), ('Bird contribution', 'Background contribution', 'Bird-background interaction')):
    correct = testing.loc[testing.error == 0, col].to_numpy()
    wrong = testing.loc[testing.error == 1, col].to_numpy()
    ax.boxplot([correct, wrong], tick_labels=['Correct', 'Incorrect'], showfliers=False)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(OUT / 'correct_vs_incorrect.png', dpi=160)
plt.show()
plt.close(fig)

if len(testing) > 0:
    example = testing.iloc[0]
    graph = nx.Graph()
    graph.add_nodes_from(['bird', 'background', '0', '1'])
    graph.add_edge('bird', 'background', weight=float(example['I_ob']))
    predicted_class = str(int(example['pred']))
    graph.add_edge('bird', predicted_class, weight=float(example['B_o']))
    graph.add_edge('background', predicted_class, weight=float(example['B_b']))
    pos = {'bird': (-1, 0), 'background': (1, 0), '0': (-1, -1.3), '1': (1, -1.3)}
    edge_colors = []
    for u, v, data in graph.edges(data=True):
        edge_colors.append('green' if data['weight'] > 0 else 'red')
    plt.figure(figsize=(7, 5))
    nx.draw_networkx(graph, pos, node_color=['#ef6656', '#7fc9e5', '#dddddd', '#dddddd'], node_size=2100, edge_color=edge_colors, width=2, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): f'{d["weight"]:+.2f}' for u, v, d in graph.edges(data=True)})
    plt.title('Example local graph (edges to predicted class only)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'example_local_graph.png', dpi=160)
    plt.show()

with open(OUT / 'experiment_settings.json', 'w') as file:
    json.dump({'root': str(ROOT), 'seed': SEED, 'image_size': SIZE, 'epochs': EPOCHS, 'checkpoint': str(checkpoint_path), 'mask_method': mask_method, 'background_intervention': 'neutral gray RGB(127,127,127)', 'bird_removal': 'OpenCV Telea inpainting over estimated bird mask', 'warning': 'Interventions are approximations; not exact CUB masks or original Places365 backgrounds.'}, file, indent=2)
print('\nFinished. Files saved to', OUT)
print('CAUTION: These results use approximate segmentation and a neutral reference background; semantic features are hand-defined, not learned NLP semantics.')