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
import hashlib
from scipy.stats import spearmanr
from scipy.stats import binomtest

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

def _prepare_frozen_test():
    SEED = 20261024
    PREV = Path('/kaggle/working/cig_idip_robustness')
    RESNET = Path('/kaggle/working/cig_idip_cifar10/cifar10_resnet18_custom_best.pt')
    TINY = PREV / 'tinycnn_8epochs.pt'
    OUT = Path('/kaggle/working/cig_idip_official_test_FROZEN_N4')
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS = ('resnet18', 'tinycnn')
    REFS = ('per_image_mean', 'dataset_mean', 'gaussian_blur')
    N = 4
    R = 8
    SUBSETS = 1 << R
    N_TEST_ERRORS = 600
    BOOT = 2000
    SHUFFLES = 20
    INFER_BATCH = 160
    D = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if D.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    assert RESNET.is_file() and TINY.is_file(), 'Missing pretrained checkpoints. Keep prior Kaggle /working outputs or attach them as Kaggle input.'
    for name in MODELS:
        for ref in REFS:
            assert (PREV / f'{name}_{ref}_all256.npz').is_file(), f'Missing CAL cache: {name} / {ref}'
    DATA_ROOT = Path('/kaggle/working/cifar10_data')
    roots = [p.parent for p in Path('/kaggle/input').glob('**/cifar-10-batches-py')] + [DATA_ROOT]
    raw = None
    root_used = None
    for root in roots:
        try:
            raw = datasets.CIFAR10(root=str(root), train=False, download=False)
            root_used = root
            break
        except Exception:
            pass
    if raw is None:
        raw = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True)
        root_used = DATA_ROOT
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.247, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test = datasets.CIFAR10(root=str(root_used), train=False, transform=tf, download=False)
    assert len(test) == 10000, f'Unexpected official TEST length {len(test)}'
    Yall = np.asarray(test.targets, dtype=int)

    class TinyCNN(nn.Module):

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            return self.fc(self.features(x).flatten(1))

    def load_model(name):
        if name == 'resnet18':
            c = torch.load(RESNET, map_location='cpu', weights_only=False)
            assert c.get('seed') == SEED, 'ResNet checkpoint seed mismatch'
            net = models.resnet18(weights=None)
            net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
            net.fc = nn.Linear(net.fc.in_features, 10)
            net.load_state_dict(c['weights'], strict=True)
        else:
            c = torch.load(TINY, map_location='cpu', weights_only=False)
            assert c.get('seed') == SEED and c.get('epochs') == 8, 'TinyCNN mismatch'
            net = TinyCNN()
            net.load_state_dict(c['weights'], strict=True)
        net.to(D).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net
    codes = np.arange(SUBSETS)
    inc = (codes[:, None] >> np.arange(R) & 1).astype(np.float32)
    card = inc.sum(1).astype(int)
    pairs = list(itertools.combinations(range(R), 2))
    scodes = np.array([1 << i for i in range(R)])
    pcodes = np.array([1 << i | 1 << j for i, j in pairs])
    pair_inc = np.stack([inc[:, i] * inc[:, j] for i, j in pairs], axis=1)
    can = np.flatnonzero(card == N)
    mask = np.zeros((SUBSETS, 1, 32, 32), dtype=np.float32)
    for code in codes:
        for k in range(R):
            if inc[code, k]:
                row, col = divmod(k, 4)
                mask[code, 0, 16 * row:16 * (row + 1), 8 * col:8 * (col + 1)] = 1
    mask = torch.from_numpy(mask).to(D)
    GAUSS = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=D) / 16
    G2 = (GAUSS[:, None] * GAUSS[None, :]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)

    def replacement(x, ref):
        if ref == 'per_image_mean':
            return x.mean((2, 3), keepdim=True).expand_as(x)
        if ref == 'dataset_mean':
            return torch.zeros_like(x)
        if ref == 'gaussian_blur':
            return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), G2, groups=3)
        raise ValueError(ref)

    def metrics_from_logits(logits, labels, pred):
        n = len(labels)
        a = np.arange(n)[:, None]
        s = np.arange(SUBSETS)[None, :]
        q = logits[a, s, labels[:, None]] - logits[a, s, pred[:, None]]
        delta = q - q[:, [0]]
        single = delta[:, scodes]
        edge = delta[:, pcodes] - single[:, [i for i, j in pairs]] - single[:, [j for i, j in pairs]]
        additive = single @ inc.T
        second = additive + edge @ pair_inc.T
        row = np.arange(n)
        top = can[additive[:, can].argmax(1)]
        cig = can[second[:, can].argmax(1)]
        gamma = second[row, cig] - second[row, top]
        actual = delta[row, cig] - delta[row, top]
        true_after = logits.argmax(2) == labels[:, None]
        assert np.all(~true_after | (delta > -1e-05))
        assert float(np.max(np.abs(delta[:, card == 2] - second[:, card == 2]))) < 0.0002
        strength = np.abs(edge).sum(1) / (np.abs(single).sum(1) + 1e-06)
        return dict(delta=delta, additive=additive, second=second, edge=edge, top=top, cig=cig, gamma=gamma, adv=actual, correct=true_after, strength=strength, residual=delta - second)

    def ci_mean(values, seed):
        x = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        b = rng.integers(0, len(x), size=(BOOT, len(x)))
        return np.quantile(x[b].mean(1), [0.025, 0.975])

    def threshold_from_prior_cal(name, ref):
        zfile = PREV / f'{name}_{ref}_all256.npz'
        with np.load(zfile, allow_pickle=False) as z:
            old = z['logits'].astype(np.float32)
            chosen = z['chosen'].astype(int)
        if len(chosen) < 600:
            raise ValueError('CAL cache unexpectedly short')
        train_raw = datasets.CIFAR10(root=str(root_used), train=True, download=False)
        train_labels = np.asarray(train_raw.targets, dtype=int)
        held = np.random.default_rng(SEED).permutation(50000)[-5000:]
        yy = train_labels[held[chosen]]
        pred = old[:, 0, :].argmax(1)
        assert np.all(pred != yy), 'CAL cache mismatch or wrong dataset split'
        m = metrics_from_logits(old, yy, pred)
        cal = np.arange(len(chosen) // 2)
        return dict(gamma75=float(np.quantile(m['gamma'][cal], 0.75)), strength75=float(np.quantile(m['strength'][cal], 0.75)), gamma50=float(np.quantile(m['gamma'][cal], 0.5)), cal_n=len(cal))
    THRESHOLDS = {(name, ref): threshold_from_prior_cal(name, ref) for name in MODELS for ref in REFS}
    summary = []
    subgroups = []
    controls = []
    fidelity = []
    perimage = []
    for model_name in MODELS:
        net = load_model(model_name)
        origin = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(test), 256), desc=f'{model_name} original TEST logits'):
                xx = torch.stack([test[i][0] for i in range(start, min(start + 256, len(test)))]).to(D)
                origin.append(net(xx).cpu().numpy())
        origin = np.concatenate(origin, axis=0)
        predall = origin.argmax(1)
        bad = np.flatnonzero(predall != Yall)
        assert len(bad) >= N_TEST_ERRORS, 'Insufficient errors for frozen cohort sample'
        sample = np.random.default_rng(SEED + 1001).permutation(bad)[:N_TEST_ERRORS]
        yy = Yall[sample]
        pp = predall[sample]
        X = torch.stack([test[int(ix)][0] for ix in sample])
        for refname in REFS:
            cached = OUT / f'{model_name}_{refname}_official_test_logits.npz'
            sha = hashlib.sha256((RESNET if model_name == 'resnet18' else TINY).read_bytes()).hexdigest()
            logits = None
            if cached.exists():
                with np.load(cached, allow_pickle=False) as z:
                    if np.array_equal(z['test_indices'], sample) and str(z['checkpoint_sha']) == sha:
                        logits = z['logits'].astype(np.float32)
            if logits is None:
                logits = np.empty((len(sample), SUBSETS, 10), dtype=np.float32)
                with torch.inference_mode():
                    for a, im in enumerate(tqdm(X, desc=f'{model_name} / {refname}: all 256 interventions')):
                        x = im[None].to(D)
                        z = replacement(x, refname)
                        for k in range(0, SUBSETS, INFER_BATCH):
                            mm = mask[k:k + INFER_BATCH]
                            edited = x * (1 - mm) + z * mm
                            logits[a, k:k + len(mm)] = net(edited).cpu().numpy()
                np.savez_compressed(cached, logits=logits, test_indices=sample, checkpoint_sha=np.array(sha))
            assert np.allclose(logits[:, 0, :], origin[sample], atol=0.001), 'Original output mismatch'
            m = metrics_from_logits(logits, yy, pp)
            row = np.arange(len(sample))
            actual = m['adv']
            g = m['gamma']
            t = m['top']
            c = m['cig']
            corr = m['correct']
            cc = corr[row, c].astype(int)
            tc = corr[row, t].astype(int)
            imprc = m['delta'][row, c] > 1e-06
            imprt = m['delta'][row, t] > 1e-06
            assert np.all(cc <= imprc) and np.all(tc <= imprt), 'Correction/margin sanity failed'
            ci = ci_mean(actual, SEED + 202)
            diffcorr = cc - tc
            ci_corr = ci_mean(diffcorr, SEED + 203)
            only_c = int(((cc == 1) & (tc == 0)).sum())
            only_t = int(((cc == 0) & (tc == 1)).sum())
            mcnemar_p = binomtest(min(only_c, only_t), n=only_c + only_t, p=0.5, alternative='two-sided').pvalue if only_c + only_t else 1.0
            rho = spearmanr(g, actual).statistic
            summary.append(dict(model=model_name, reference=refname, n=len(sample), N=N, mean_margin_adv=float(actual.mean()), margin_ci_lo=float(ci[0]), margin_ci_hi=float(ci[1]), correction_CIG=float(cc.mean()), correction_TopN=float(tc.mean()), correction_gain=float(diffcorr.mean()), correction_ci_lo=float(ci_corr[0]), correction_ci_hi=float(ci_corr[1]), mcnemar_p=float(mcnemar_p), cig_only=only_c, top_only=only_t, gamma_spearman=float(rho), different_selection=float((c != t).mean()), margin_improved_CIG=float(imprc.mean()), margin_improved_TopN=float(imprt.mean())))
            thresholds = THRESHOLDS[model_name, refname]
            for group, flag in [('ALL', np.ones(len(sample), dtype=bool)), ('HIGH_GAMMA_CAL_Q75', g >= thresholds['gamma75']), ('LOW_GAMMA_CAL_Q75', g < thresholds['gamma75']), ('HIGH_STRENGTH_CAL_Q75', m['strength'] >= thresholds['strength75'])]:
                ids = np.flatnonzero(flag)
                if len(ids) == 0:
                    continue
                subgroups.append(dict(model=model_name, reference=refname, group=group, n=len(ids), coverage=len(ids) / len(sample), actual_gain=float(actual[ids].mean()), actual_positive=float((actual[ids] > 1e-06).mean()), correction_delta=float(diffcorr[ids].mean()), correction_CIG=float(cc[ids].mean()), correction_TopN=float(tc[ids].mean())))
            fidelity.append(dict(model=model_name, reference=refname, n=len(sample), MAE_nodes=float(np.abs(m['delta'][:, can] - m['additive'][:, can]).mean()), MAE_CIG=float(np.abs(m['delta'][:, can] - m['second'][:, can]).mean()), RMSE_CIG=float(np.sqrt(np.mean((m['delta'][:, can] - m['second'][:, can]) ** 2)))))
            rng = np.random.default_rng(SEED + 3000)
            for k in range(SHUFFLES):
                order = np.argsort(rng.random(m['edge'].shape), axis=1)
                shuffled_edges = np.take_along_axis(m['edge'], order, axis=1)
                shuffled_score = m['additive'] + shuffled_edges @ pair_inc.T
                selected = can[shuffled_score[:, can].argmax(1)]
                shuffled_gain = m['delta'][row, c] - m['delta'][row, selected]
                shuffled_corr = cc - corr[row, selected].astype(int)
                controls.append(dict(model=model_name, reference=refname, repeat=k, mean_advantage=float(shuffled_gain.mean()), correction_gain=float(shuffled_corr.mean())))
            perimage.append(pd.DataFrame(dict(model=model_name, reference=refname, test_index=sample, gamma=g, strength=m['strength'], real_advantage=actual, CIG_corrected=cc, TopN_corrected=tc, selected_CIG_code=c, selected_TopN_code=t, CIG_margin_improved=imprc.astype(int), TopN_margin_improved=imprt.astype(int), high_gamma=(g >= thresholds['gamma75']).astype(int))))
        del net
        if D.type == 'cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(OUT / 'primary_global_results.csv', index=False)
    pd.DataFrame(subgroups).to_csv(OUT / 'frozen_idip_subgroups.csv', index=False)
    pd.DataFrame(fidelity).to_csv(OUT / 'fidelity_all_N4_subsets.csv', index=False)
    pd.DataFrame(controls).to_csv(OUT / 'shuffled_edge_controls.csv', index=False)
    pd.concat(perimage, ignore_index=True).to_csv(OUT / 'per_image_official_test_results.csv', index=False)
    (OUT / 'FROZEN_PROTOCOL.json').write_text(json.dumps(dict(protocol_frozen=True, seed=SEED, dataset='CIFAR-10 OFFICIAL TEST', n_test=10000, sampled_original_errors_per_model=N_TEST_ERRORS, models=MODELS, references=REFS, budget_N=N, visual_units='eight fixed 16x8 nonoverlapping grid regions', target='oracle true-class logit minus ORIGINAL incorrect predicted-class logit', gamma='argmax second-order effect minus same second-order effect at additive TopN choice', prospective_rule='gamma >= prior CAL 75th percentile; no test-label tuning of threshold', null_control='20 predetermined within-image edge permutations', controls='strength CAL 75th percentile; no test-based subgroup selection', main_metric='paired realized target-margin advantage CIG minus additive TopN', second_metric='difference in corrected-original-error rates', statistical_methods='paired bootstrap 2000 samples; exact McNemar discordance test', caveats=['OFFICIAL TEST may already have been used historically for unrelated model development; disclose if so.', 'True-label oracle interventions, not autonomous error correction.', 'Fixed visual grid regions are not validated human-interpretable semantic concepts.', 'Two models use different error cohorts.', 'Three references and models are correlated, not independent replications.', 'No claim of universal superiority, deterministic per-image guarantee, or semantic graph validity.'], prior_calibration_thresholds={f'{a}|{b}': v for (a, b), v in THRESHOLDS.items()}), indent=2))

def main():
    if not Path('/kaggle/working/cig_idip_cifar10/cifar10_resnet18_custom_best.pt').is_file():
        _prepare_resnet()
    if not all(((Path('/kaggle/working/cig_idip_robustness') / f'{m}_{r}_all256.npz').is_file() for m in ('resnet18', 'tinycnn') for r in ('per_image_mean', 'dataset_mean', 'gaussian_blur'))):
        _prepare_calibration()
    if not all(((Path('/kaggle/working/cig_idip_official_test_FROZEN_N4') / f'{m}_{r}_official_test_logits.npz').is_file() for m in ('resnet18', 'tinycnn') for r in ('per_image_mean', 'dataset_mean', 'gaussian_blur'))):
        _prepare_frozen_test()
    SEED = 20261024
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    D = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT = Path('/kaggle/working/cig_three_limitations')
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN = Path('/kaggle/working/cig_idip_official_test_FROZEN_N4')
    REFS = ('per_image_mean', 'dataset_mean', 'gaussian_blur')
    ARCHS = ('resnet18', 'tinycnn')
    RUN_CIFAR_A_B = True
    RUN_WATERBIRDS_C = False
    WATERBIRDS_MODEL = 'old_spurious'
    WB_MAX_ERRORS = 120
    WB_IMG_SIZE = 224
    WB_BATCH = 12
    N_PERM = 20
    B = 4

    def report_ci(values, seed=0, rep=1500):
        a = np.asarray(values, dtype=float)
        if not len(a):
            return (float('nan'), float('nan'))
        rng = np.random.default_rng(SEED + seed)
        vals = []
        for k in range(rep):
            vals.append(a[rng.integers(0, len(a), len(a))].mean())
        return tuple(np.quantile(vals, [0.025, 0.975]))

    def evaluate_all_subsets(logits, y, pred, nodes, budget):
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        pred = np.asarray(pred, dtype=int)
        n = logits.shape[0]
        nc = 1 << nodes
        assert logits.shape[1] == nc and budget <= nodes
        assert len(y) == len(pred) == n and np.all(pred != y)
        ii = np.arange(n)
        q = logits[ii[:, None], np.arange(nc)[None, :], y[:, None]] - logits[ii[:, None], np.arange(nc)[None, :], pred[:, None]]
        d = q - q[:, [0]]
        bits = (np.arange(nc)[:, None] >> np.arange(nodes)[None, :] & 1).astype(np.float64)
        pop = bits.sum(1)
        single = d[:, [1 << i for i in range(nodes)]]
        pairs = list(itertools.combinations(range(nodes), 2))
        pcodes = np.array([1 << i | 1 << j for i, j in pairs], dtype=int)
        edges = d[:, pcodes] - single[:, [i for i, j in pairs]] - single[:, [j for i, j in pairs]]
        pi = np.column_stack([bits[:, i] * bits[:, j] for i, j in pairs])
        add = single @ bits.T
        second = add + edges @ pi.T
        cand = np.flatnonzero(pop == budget)
        top = cand[np.argmax(add[:, cand], axis=1)]
        cig = cand[np.argmax(second[:, cand], axis=1)]
        oracle = cand[np.argmax(d[:, cand], axis=1)]
        adv = d[ii, cig] - d[ii, top]
        gain = d[ii, oracle] - d[ii, cig]
        assert np.min(gain) >= -1e-08
        gamma = second[ii, cig] - second[ii, top]
        assert np.min(gamma) >= -1e-08
        corr = logits.argmax(axis=2) == y[:, None]
        out = dict(n=n, nodes=nodes, budget=budget, logits=logits, d=d, add=add, second=second, single=single, pairs=pairs, edges=edges, bits=bits, pop=pop, candidate=cand, top=top, cig=cig, oracle=oracle, adv=adv, regret=gain, gamma=gamma, corr=corr, correct_cig=corr[ii, cig], correct_top=corr[ii, top], correct_oracle=corr[ii, oracle])
        triples = list(itertools.combinations(range(nodes), 3))
        triple_codes = np.array([sum((1 << i for i in t)) for t in triples], dtype=int)
        pair_col = {p: z for z, p in enumerate(pairs)}
        tri = np.stack([d[:, sum((1 << i for i in t))] - sum((single[:, i] for i in t)) - sum((edges[:, pair_col[p]] for p in itertools.combinations(t, 2))) for t in triples], axis=1)
        ti = np.column_stack([np.prod(bits[:, list(t)], axis=1) for t in triples])
        third = second + tri @ ti.T
        out.update(triples=triples, tri=tri, third=third)
        return out

    def _row_to_dict(model, ref, m):
        n = m['n']
        a = np.arange(n)
        d = m['d']
        can = m['candidate']
        c = m['cig']
        t = m['top']
        o = m['oracle']
        epsmax = np.abs(d[:, can] - m['second'][:, can]).max(axis=1)
        real_bound = (m['regret'] <= 2 * epsmax + 1e-07).mean()
        return dict(model=model, reference=ref, n=n, N=m['budget'], mean_advantage=float(m['adv'].mean()), ci_margin_low=report_ci(m['adv'], 1)[0], ci_margin_high=report_ci(m['adv'], 1)[1], top_correction=float(m['correct_top'].mean()), cig_correction=float(m['correct_cig'].mean()), oracle_correction=float(m['correct_oracle'].mean()), avg_regret_CIG=float(m['regret'].mean()), avg_regret_TopN=float((d[a, o] - d[a, t]).mean()), median_regret_CIG=float(np.median(m['regret'])), exact_oracle_match_CIG=float((c == o).mean()), exact_oracle_match_TopN=float((t == o).mean()), regret_bound_holds=float(real_bound), mean_abs_triple=float(np.abs(m['tri']).mean()), MAE_nodes=float(np.abs(d[:, can] - m['add'][:, can]).mean()), MAE_2=float(np.abs(d[:, can] - m['second'][:, can]).mean()), MAE_3=float(np.abs(d[:, can] - m['third'][:, can]).mean()))
    AB = []
    TR = []
    PREDS = []
    if RUN_CIFAR_A_B:
        meta = pd.read_csv(FROZEN / 'per_image_official_test_results.csv')
        data_dirs = [Path('/kaggle/working/cifar10_data')] + [Path(z).parent for z in Path('/kaggle/input').glob('**/cifar-10-batches-py')]
        ds = None
        for path in data_dirs:
            try:
                ds = datasets.CIFAR10(root=str(path), train=False, download=False)
                break
            except Exception:
                pass
        if ds is None:
            raise FileNotFoundError('CIFAR-10 TEST files not found; attach CIFAR-10 or preserve /kaggle/working/cifar10_data')
        yy = np.asarray(ds.targets)
        for model in ARCHS:
            for ref in REFS:
                path = FROZEN / f'{model}_{ref}_official_test_logits.npz'
                if not path.exists():
                    raise FileNotFoundError(f'Missing official TEST intervention cache: {path}')
                with np.load(path, allow_pickle=False) as z:
                    logits = z['logits']
                    indices = z['test_indices'].astype(int)
                pred = logits[:, 0, :].argmax(1)
                lab = yy[indices]
                m = evaluate_all_subsets(logits, lab, pred, 8, B)
                row = _row_to_dict(model, ref, m)
                AB.append(row)
                selected = np.arange(len(indices))
                tri3 = m['tri']
                triple_regressors = np.abs(tri3).sum(axis=1)
                exact_residual = np.abs(m['d'][selected, m['cig']] - m['second'][selected, m['cig']])
                corr_spear = pd.Series(triple_regressors).corr(pd.Series(exact_residual), method='spearman')
                TR.append(dict(model=model, reference=ref, spearman_triple_strength_vs_CIG_residual=float(corr_spear), residual_at_CIG_MAE=float(exact_residual.mean()), triple_interaction_mean_abs=float(np.abs(tri3).mean()), fraction_cig_oracle=float((m['cig'] == m['oracle']).mean())))
                PREDS.append(pd.DataFrame(dict(model=model, reference=ref, test_index=indices, selected_top=m['top'], selected_cig=m['cig'], selected_oracle=m['oracle'], gamma=m['gamma'], regret_cig=m['regret'], margin_advantage=m['adv'], top_correct=m['correct_top'].astype(int), cig_correct=m['correct_cig'].astype(int), oracle_correct=m['correct_oracle'].astype(int))))
        pd.DataFrame(AB).to_csv(OUT / 'A_oracle_vs_CIG.csv', index=False)
        pd.DataFrame(TR).to_csv(OUT / 'B_third_order_analysis.csv', index=False)
        pd.concat(PREDS, ignore_index=True).to_csv(OUT / 'A_B_per_image.csv', index=False)
    print('Results:', OUT)
if __name__ == '__main__':
    main()
