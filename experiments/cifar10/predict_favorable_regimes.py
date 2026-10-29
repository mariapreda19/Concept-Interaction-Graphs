import json
import random
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import hashlib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from scipy.stats import spearmanr

def _prepare_resnet():
    SEED = 20261024
    OUT = Path('/kaggle/working/cig_idip_cifar10')
    OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 8
    TRAIN_BATCH = 128
    INFER_BATCH = 160
    ERRORS_TOTAL = 600
    N_REGIONS = 8
    BUDGETS = (2, 3, 4, 5, 6)
    ANALYSIS_BUDGET = 4
    N_SHUFFLES = 30
    N_BOOT = 1000
    ALPHA = 0.1
    DATA_ROOT = Path('/kaggle/working/cifar10_data')
    CKPT = OUT / 'cifar10_resnet18_custom_best.pt'
    CACHE = OUT / 'heldout_intervention_logits.npz'
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    roots = [DATA_ROOT, Path('/kaggle/input')]
    for folder in Path('/kaggle/input').glob('**/cifar-10-batches-py'):
        roots.insert(0, folder.parent)
    raw = None
    used_root = None
    for root in roots:
        try:
            candidate = datasets.CIFAR10(root=str(root), train=True, download=False)
            raw = candidate
            used_root = root
            break
        except Exception:
            pass
    if raw is None:
        try:
            raw = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True)
            used_root = DATA_ROOT
        except Exception as e:
            raise RuntimeError('CIFAR-10 missing. Enable Kaggle internet temporarily, or attach a CIFAR-10 dataset containing cifar-10-batches-py under /kaggle/input.') from e
    idx = np.random.default_rng(SEED).permutation(len(raw))
    fit_ids = idx[:-5000]
    held_ids = idx[-5000:]
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.247, 0.2435, 0.2616)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_dataset = datasets.CIFAR10(root=str(used_root), train=True, download=False, transform=train_tf)
    val_dataset = datasets.CIFAR10(root=str(used_root), train=True, download=False, transform=eval_tf)
    train_loader = DataLoader(Subset(train_dataset, fit_ids.tolist()), batch_size=TRAIN_BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, held_ids.tolist()), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    def make_model():
        net = models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(net.fc.in_features, 10)
        return net
    net = make_model().to(DEVICE)
    if CKPT.exists():
        state = torch.load(CKPT, map_location='cpu', weights_only=False)
        if state.get('seed') != SEED or state.get('epochs') != EPOCHS:
            raise RuntimeError('Existing checkpoint uses different SEED/EPOCHS. Move or delete checkpoint or restore those settings.')
        net.load_state_dict(state['weights'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.12, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        for epoch in range(EPOCHS):
            net.train()
            total = correct = 0
            loss_sum = 0.0
            for x, y in tqdm(train_loader, desc=f'Train CNN {epoch + 1}/{EPOCHS}', leave=False):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = net(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total += len(y)
                correct += int((logits.argmax(1) == y).sum())
                loss_sum += float(loss) * len(y)
            scheduler.step()
        torch.save({'weights': net.state_dict(), 'seed': SEED, 'epochs': EPOCHS, 'train_ids_sha': __import__('hashlib').sha256(fit_ids.tobytes()).hexdigest()}, CKPT)

def _prepare_calibration():
    SEED = 20261024
    ROOT = Path('/kaggle/working/cig_idip_cifar10')
    OUT = Path('/kaggle/working/cig_idip_robustness')
    OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RESNET_CKPT = ROOT / 'cifar10_resnet18_custom_best.pt'
    TINY_CKPT = OUT / 'tinycnn_8epochs.pt'
    N_REGIONS = 8
    BUDGETS = (2, 3, 4, 5, 6)
    FOCAL_N = 4
    ERRORS_PER_MODEL = 600
    EPOCHS_TINY = 8
    BATCH = 128
    INFER_BATCH = 160
    BOOT = 1000
    REFERENCES = ('per_image_mean', 'dataset_mean', 'gaussian_blur')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    assert RESNET_CKPT.exists(), f'Missing previous frozen model: {RESNET_CKPT}. Run cig_interaction_dominated_cifar10.py first.'
    DATA_ROOT = Path('/kaggle/working/cifar10_data')
    roots = [p.parent for p in Path('/kaggle/input').glob('**/cifar-10-batches-py')] + [DATA_ROOT]
    raw = None
    used_root = None
    for root in roots:
        try:
            raw = datasets.CIFAR10(root=str(root), train=True, download=False)
            used_root = root
            break
        except Exception:
            pass
    if raw is None:
        try:
            raw = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True)
            used_root = DATA_ROOT
        except Exception as e:
            raise RuntimeError('Attach CIFAR-10 or enable Kaggle internet for download') from e
    ids = np.random.default_rng(SEED).permutation(len(raw))
    fit_ids, held_ids = (ids[:-5000], ids[-5000:])
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.247, 0.2435, 0.2616)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10(str(used_root), train=True, transform=train_tf)
    val_ds = datasets.CIFAR10(str(used_root), train=True, transform=eval_tf)
    train_loader = DataLoader(Subset(train_ds, fit_ids.tolist()), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    held_loader = DataLoader(Subset(val_ds, held_ids.tolist()), batch_size=256, shuffle=False, num_workers=0)

    def resnet18():
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, 10)
        return m

    class TinyCNN(nn.Module):

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            return self.fc(self.features(x).flatten(1))

    def load_model(name):
        if name == 'resnet18':
            checkpoint = torch.load(RESNET_CKPT, map_location='cpu', weights_only=False)
            if checkpoint.get('seed') != SEED:
                raise RuntimeError('ResNet checkpoint has unexpected seed')
            m = resnet18()
            m.load_state_dict(checkpoint['weights'])
        else:
            m = TinyCNN()
            if TINY_CKPT.exists():
                ck = torch.load(TINY_CKPT, map_location='cpu', weights_only=False)
                if ck.get('seed') != SEED or ck.get('epochs') != EPOCHS_TINY:
                    raise RuntimeError('TinyCNN cache mismatch')
                m.load_state_dict(ck['weights'])
            else:
                m.to(DEVICE)
                opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_TINY)
                for e in range(EPOCHS_TINY):
                    m.train()
                    correct = total = 0
                    for x, y in tqdm(train_loader, desc=f'TinyCNN train {e + 1}/{EPOCHS_TINY}', leave=False):
                        x, y = (x.to(DEVICE), y.to(DEVICE))
                        opt.zero_grad(set_to_none=True)
                        logits = m(x)
                        loss = F.cross_entropy(logits, y)
                        loss.backward()
                        opt.step()
                        correct += int((logits.argmax(1) == y).sum())
                        total += len(y)
                    sched.step()
                torch.save(dict(weights=m.cpu().state_dict(), seed=SEED, epochs=EPOCHS_TINY), TINY_CKPT)
        m.to(DEVICE).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    codes = np.arange(2 ** N_REGIONS)
    inc = (codes[:, None] >> np.arange(N_REGIONS) & 1).astype(np.float32)
    card = inc.sum(axis=1).astype(int)
    pairs = list(itertools.combinations(range(N_REGIONS), 2))
    scodes = np.array([1 << i for i in range(N_REGIONS)])
    pcodes = np.array([1 << i | 1 << j for i, j in pairs])
    mask = np.zeros((len(codes), 1, 32, 32), np.float32)
    for code in codes:
        for k in range(N_REGIONS):
            if inc[code, k]:
                row, col = divmod(k, 4)
                mask[code, 0, 16 * row:16 * (row + 1), 8 * col:8 * (col + 1)] = 1
    mask = torch.from_numpy(mask).to(DEVICE)
    assert np.all(mask[0].cpu().numpy() == 0)
    inc_pair = np.stack([inc[:, i] * inc[:, j] for i, j in pairs], axis=1)
    GAUSS = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=DEVICE) / 16
    G2 = (GAUSS[:, None] * GAUSS[None, :]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)

    def reference(x, which):
        if which == 'per_image_mean':
            return x.mean((2, 3), keepdim=True).expand_as(x)
        if which == 'dataset_mean':
            return torch.zeros_like(x)
        if which == 'gaussian_blur':
            return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), G2, groups=3)
        raise ValueError(which)
    with torch.inference_mode():
        Xall = torch.stack([val_ds[int(held_ids[i])][0] for i in range(len(held_ids))])
        Yall = np.array([val_ds[int(held_ids[i])][1] for i in range(len(held_ids))])

    def original_predictions(net):
        parts = []
        with torch.inference_mode():
            for k in range(0, len(Xall), 256):
                parts.append(net(Xall[k:k + 256].to(DEVICE)).cpu().numpy())
        return np.concatenate(parts)

    def intervention_logits(net, model_name, which, chosen):
        cache = OUT / f'{model_name}_{which}_all256.npz'
        ck = RESNET_CKPT if model_name == 'resnet18' else TINY_CKPT
        token = hashlib.sha256(ck.read_bytes()).hexdigest()
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                if np.array_equal(z['chosen'], chosen) and str(z['checkpoint_sha']) == token:
                    return z['logits'].astype(np.float32)
        all_logits = np.empty((len(chosen), len(codes), 10), np.float32)
        with torch.inference_mode():
            for t, ix in enumerate(tqdm(chosen, desc=f'{model_name} {which}: 256 subsets')):
                x = Xall[int(ix)][None].to(DEVICE)
                ref = reference(x, which)
                for st in range(0, len(codes), INFER_BATCH):
                    m = mask[st:st + INFER_BATCH]
                    batch = x * (1 - m) + ref * m
                    all_logits[t, st:st + len(m)] = net(batch).cpu().numpy()
        np.savez_compressed(cache, logits=all_logits, chosen=chosen, checkpoint_sha=np.array(token))
        return all_logits
    rng_boot = np.random.default_rng(SEED + 333)

    def ci(vals):
        vals = np.asarray(vals, float)
        if len(vals) < 12:
            return (np.nan, np.nan)
        b = rng_boot.integers(len(vals), size=(BOOT, len(vals)))
        return tuple(np.quantile(vals[b].mean(1), [0.025, 0.975]))

    def unpack(logits, y, pred):
        L = len(y)
        K = len(codes)
        idx = np.arange(L)[:, None]
        sub = np.arange(K)[None, :]
        q = logits[idx, sub, y[:, None]] - logits[idx, sub, pred[:, None]]
        delta = q - q[:, [0]]
        corrected = logits.argmax(2) == y[:, None]
        single = delta[:, scodes]
        inter = delta[:, pcodes] - single[:, [i for i, j in pairs]] - single[:, [j for i, j in pairs]]
        ap1 = single @ inc.T
        ap2 = ap1 + inter @ inc_pair.T
        rem = delta - ap2
        assert float(np.abs(rem[:, card == 2]).max()) < 0.0002
        assert np.all(~corrected | (delta > 1e-06)), 'Corrected => fixed true-class margin improved'
        return (delta, corrected, ap1, ap2, rem, single, inter)
    for model_name in ('resnet18', 'tinycnn'):
        net = load_model(model_name)
        logits_original = original_predictions(net)
        preds = logits_original.argmax(1)
        bad = np.flatnonzero(preds != Yall)
        chosen = np.random.default_rng(SEED + 3).permutation(bad)[:min(ERRORS_PER_MODEL, len(bad))]
        for refname in REFERENCES:
            intervention_logits(net, model_name, refname, chosen)
        del net
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

def main():
    if not Path('/kaggle/working/cig_idip_cifar10/cifar10_resnet18_custom_best.pt').is_file():
        _prepare_resnet()
    if not all(((Path('/kaggle/working/cig_idip_robustness') / f'{m}_{r}_all256.npz').is_file() for m in ('resnet18', 'tinycnn') for r in ('per_image_mean', 'dataset_mean', 'gaussian_blur'))):
        _prepare_calibration()
    SEED = 20261024
    PREV = Path('/kaggle/working/cig_idip_robustness')
    OUT = PREV / 'prospective_idip_predictor'
    OUT.mkdir(parents=True, exist_ok=True)
    REFS = ('per_image_mean', 'dataset_mean', 'gaussian_blur')
    MODELS = ('resnet18', 'tinycnn')
    NREG = 8
    N = 4
    TARGET_ADVANTAGE = 0.25
    BOOT = 1000
    rng = np.random.default_rng(SEED + 404)
    DATA_ROOT = Path('/kaggle/working/cifar10_data')
    roots = [p.parent for p in Path('/kaggle/input').glob('**/cifar-10-batches-py')] + [DATA_ROOT]
    data = None
    for root in roots:
        try:
            data = datasets.CIFAR10(root=str(root), train=True, download=False)
            break
        except Exception:
            pass
    if data is None:
        raise FileNotFoundError('CIFAR-10 TRAIN data unavailable. Attach the same dataset used previously.')
    y_all = np.asarray(data.targets, dtype=int)
    held_ids = np.random.default_rng(SEED).permutation(len(y_all))[-5000:]
    y_held = y_all[held_ids]
    codes = np.arange(1 << NREG)
    inc = (codes[:, None] >> np.arange(NREG) & 1).astype(np.float64)
    card = inc.sum(1).astype(int)
    pairs = list(itertools.combinations(range(NREG), 2))
    single_codes = np.array([1 << i for i in range(NREG)])
    pair_codes = np.array([1 << i | 1 << j for i, j in pairs])
    pair_inc = np.column_stack([inc[:, i] * inc[:, j] for i, j in pairs])
    candidates = np.flatnonzero(card == N)

    def graph_features(logits, y, pred):
        size = logits.shape[0]
        if logits.shape[1:] != (1 << NREG, 10):
            raise ValueError(f'Unexpected cache logits dimensions: {logits.shape}')
        row = np.arange(size)[:, None]
        subset = np.arange(1 << NREG)[None, :]
        score = logits[row, subset, y[:, None]] - logits[row, subset, pred[:, None]]
        delta = score - score[:, [0]]
        single = delta[:, single_codes]
        edge = delta[:, pair_codes] - single[:, [i for i, j in pairs]] - single[:, [j for i, j in pairs]]
        first = single @ inc.T
        second = first + edge @ pair_inc.T
        top = candidates[np.argmax(first[:, candidates], axis=1)]
        cig = candidates[np.argmax(second[:, candidates], axis=1)]
        i = np.arange(size)
        gamma = second[i, cig] - second[i, top]
        assert np.all(gamma >= -0.0001)
        actual = delta[i, cig] - delta[i, top]
        corrected = logits.argmax(2) == y[:, None]
        assert np.all(~corrected | (delta > 1e-05)), 'Sanity failed: correction must improve true-class margin'
        absedge = np.abs(edge)
        sumedge = absedge.sum(1)
        strength = sumedge / (np.abs(single).sum(1) + 1e-06)
        signed_edge = edge.sum(1)
        maxedge = absedge.max(1)
        meanedge = absedge.mean(1)
        edge_std = edge.std(1)
        posfrac = (edge > 0).mean(1)
        graph_spread = second[:, candidates].std(1)
        node_spread = first[:, candidates].std(1)
        pred_score_gap = second[i, cig] - np.partition(second[:, candidates], -2, axis=1)[:, -2]
        edit_different = (top != cig).astype(float)
        feat = pd.DataFrame({'gamma': gamma, 'strength': strength, 'edge_L1': sumedge, 'edge_signed_sum': signed_edge, 'edge_max_abs': maxedge, 'edge_mean_abs': meanedge, 'edge_sd': edge_std, 'edge_positive_fraction': posfrac, 'node_L1': np.abs(single).sum(1), 'node_max_abs': np.abs(single).max(1), 'node_sum': single.sum(1), 'node_sd': single.std(1), 'graph_candidate_spread': graph_spread, 'node_candidate_spread': node_spread, 'selection_differs': edit_different, 'predicted_best_margin_gain': second[i, cig], 'topn_predicted_second_order_gain': second[i, top], 'graph_opt_gap': pred_score_gap})
        target = pd.DataFrame({'real_advantage': actual, 'topn_corrected': corrected[i, top].astype(int), 'cig_corrected': corrected[i, cig].astype(int), 'cig_only_corrects': (corrected[i, cig] & ~corrected[i, top]).astype(int), 'topn_only_corrects': (corrected[i, top] & ~corrected[i, cig]).astype(int), 'material_advantage': (actual > TARGET_ADVANTAGE).astype(int), 'positive_advantage': (actual > 1e-06).astype(int)})
        return (feat, target)

    def classifier_scores(Xtr, ytr, Xte):
        if np.unique(ytr).size < 2:
            return np.full(len(Xte), np.mean(ytr), dtype=float)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]

    def normalized_gamma_score(g_cal, y_cal, g_eval):
        return classifier_scores(g_cal.reshape(-1, 1), y_cal, g_eval.reshape(-1, 1))

    def evaluate_binary(label, pred, config, task, model_name):
        label = np.asarray(label, dtype=int)
        pred = np.asarray(pred, dtype=float)
        if len(np.unique(label)) < 2:
            return dict(model=model_name, reference=config, task=task, predictor=model_name, n=len(label), positives=int(label.sum()), AUROC=np.nan, AUPRC=np.nan, Brier=np.nan)
        return dict(model=model_name, reference=config, task=task, n=len(label), positives=int(label.sum()), AUROC=roc_auc_score(label, pred), AUPRC=average_precision_score(label, pred), Brier=brier_score_loss(label, np.clip(pred, 0, 1)))
    summary = []
    subgroup = []
    predictions = []
    calibration_info = []
    for model_name in MODELS:
        original_chosen = None
        for ref in REFS:
            cache = PREV / f'{model_name}_{ref}_all256.npz'
            if not cache.is_file():
                raise FileNotFoundError(f'Missing {cache}; run cig_idip_robustness_validation.py first.')
            with np.load(cache, allow_pickle=False) as z:
                chosen = z['chosen'].astype(int)
                logits = z['logits'].astype(np.float32)
            if original_chosen is not None and (not np.array_equal(chosen, original_chosen)):
                raise ValueError('Inconsistent chosen images across references')
            original_chosen = chosen
            if len(chosen) < 40 or len(chosen) % 2:
                raise ValueError('Expected even number of model-specific errors')
            n_cal = len(chosen) // 2
            cal = np.arange(n_cal)
            ev = np.arange(n_cal, len(chosen))
            labels = y_held[chosen]
            pred = logits[:, 0, :].argmax(1)
            if np.any(pred == labels):
                raise ValueError('Cache includes originally correct examples; unexpected protocol mismatch')
            features, target = graph_features(logits, labels, pred)
            assert np.isfinite(features.to_numpy()).all()
            spec = {'gamma_only': ['gamma'], 'strength_only': ['strength'], 'gamma_and_strength': ['gamma', 'strength'], 'full_graph_predictor': list(features.columns)}
            for task in ('material_advantage', 'cig_only_corrects'):
                ytr = target.iloc[cal][task].to_numpy(dtype=int)
                yte = target.iloc[ev][task].to_numpy(dtype=int)
                for method, cols in spec.items():
                    p = classifier_scores(features.iloc[cal][cols].to_numpy(), ytr, features.iloc[ev][cols].to_numpy())
                    if np.unique(yte).size > 1:
                        roc = roc_auc_score(yte, p)
                        ap = average_precision_score(yte, p)
                        brier = brier_score_loss(yte, np.clip(p, 0, 1))
                    else:
                        roc = ap = np.nan
                        brier = brier_score_loss(yte, np.clip(p, 0, 1))
                    summary.append(dict(model=model_name, reference=ref, task=task, predictor=method, n=len(ev), positives=int(yte.sum()), prevalence=float(yte.mean()), AUROC=roc, AUPRC=ap, Brier=brier))
                    if task == 'material_advantage':
                        pc = classifier_scores(features.iloc[cal][cols].to_numpy(), ytr, features.iloc[cal][cols].to_numpy())
                        threshold = float(np.quantile(pc, 0.75))
                        selected = p >= threshold
                        selected_delta = target.iloc[ev].real_advantage.to_numpy()[selected]
                        subgroup.append(dict(model=model_name, reference=ref, predictor=method, selection_threshold_cal=threshold, n=int(selected.sum()), coverage=float(selected.mean()), real_advantage_mean=float(selected_delta.mean()) if len(selected_delta) else np.nan, real_advantage_positive_rate=float((selected_delta > 1e-06).mean()) if len(selected_delta) else np.nan, real_advantage_over_025_rate=float((selected_delta > TARGET_ADVANTAGE).mean()) if len(selected_delta) else np.nan, correction_gain=float((target.iloc[ev].cig_corrected.to_numpy()[selected] - target.iloc[ev].topn_corrected.to_numpy()[selected]).mean()) if selected.any() else np.nan))
                    predictions.append(pd.DataFrame(dict(model=model_name, reference=ref, task=task, predictor=method, heldout_index=chosen[ev], predicted_score=p, true_outcome=yte, real_advantage=target.iloc[ev].real_advantage.to_numpy(), gamma=features.iloc[ev].gamma.to_numpy(), strength=features.iloc[ev].strength.to_numpy())))
            corr, pval = spearmanr(features.iloc[ev].gamma, target.iloc[ev].real_advantage)
            calibration_info.append(dict(model=model_name, reference=ref, N=N, CAL=len(cal), EVAL=len(ev), gamma_spearman_eval=float(corr), gamma_spearman_p_exploratory=float(pval), gamma_mean_cal=float(features.iloc[cal].gamma.mean()), gamma_mean_eval=float(features.iloc[ev].gamma.mean()), actual_gain_eval=float(target.iloc[ev].real_advantage.mean()), actual_gain_positive_eval=float(target.iloc[ev].positive_advantage.mean()), factual_cig_only_corrects_eval=float(target.iloc[ev].cig_only_corrects.mean()), factual_topn_only_corrects_eval=float(target.iloc[ev].topn_only_corrects.mean())))
    summary = pd.DataFrame(summary)
    subgroup = pd.DataFrame(subgroup)
    calibration_info = pd.DataFrame(calibration_info)
    summary.to_csv(OUT / 'predictor_metrics.csv', index=False)
    subgroup.to_csv(OUT / 'prospective_top_quartile.csv', index=False)
    calibration_info.to_csv(OUT / 'gamma_rank_correlations.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'prediction_scores_eval.csv', index=False)
    (OUT / 'protocol.json').write_text(json.dumps({'seed': SEED, 'N': N, 'target_advantage_logit_margin': TARGET_ADVANTAGE, 'datasets': 'CIFAR-10 TRAIN partition only; original heldout split, no official TEST', 'train': '300 model-specific CAL errors per configuration; same CAL across references of a model', 'evaluate': '300 model-specific EVAL errors; same EVAL across references of a model', 'features': 'single and pair intervention effects only, before multi-concept intervention outcomes', 'tasks': ['real advantage > 0.25', 'CIG corrects but Top-N does not'], 'caveat': 'Exploratory; earlier runs repeatedly inspected EVAL and influenced experiment design; no independent confirmation', 'oracle': 'True label defines target margin; this is diagnostic intervention analysis, not deployable repair', 'note': 'Multiple combinations are correlated, and classifier fitting on CAL has only 300 observations; do not interpret as independent replications.'}, indent=2))
    print('Results:', OUT)
if __name__ == '__main__':
    main()
