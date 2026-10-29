import json
import random
import itertools
import math
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

def main():
    if not Path('/kaggle/working/cig_idip_cifar10/cifar10_resnet18_custom_best.pt').is_file():
        _prepare_resnet()
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
    rows = []
    fidelity = []
    subgroup = []
    certifications = []
    controls = []
    for model_name in ('resnet18', 'tinycnn'):
        net = load_model(model_name)
        logits_original = original_predictions(net)
        preds = logits_original.argmax(1)
        bad = np.flatnonzero(preds != Yall)
        if len(bad) < 100:
            raise RuntimeError(f'{model_name}: too few errors ({len(bad)})')
        chosen = np.random.default_rng(SEED + 3).permutation(bad)[:min(ERRORS_PER_MODEL, len(bad))]
        ncal = len(chosen) // 2
        cal = np.arange(ncal)
        ev = np.arange(ncal, len(chosen))
        iy = Yall[chosen]
        ip = preds[chosen]
        for refname in REFERENCES:
            lg = intervention_logits(net, model_name, refname, chosen)
            delta, correct, ap1, ap2, rem, single, inter = unpack(lg, iy, ip)
            strength = np.abs(inter).sum(1) / (np.abs(single).sum(1) + 1e-06)
            qcuts = np.quantile(strength[cal], [0.25, 0.5, 0.75])
            quartile = np.searchsorted(qcuts, strength, side='right') + 1
            for n in BUDGETS:
                candidates = np.flatnonzero(card == n)
                top = candidates[ap1[:, candidates].argmax(1)]
                cig = candidates[ap2[:, candidates].argmax(1)]
                A = np.arange(len(chosen))
                gamma = ap2[A, cig] - ap2[A, top]
                advantage = delta[A, cig] - delta[A, top]
                gamma_cut = float(np.quantile(gamma[cal], 0.5))
                rsel = np.maximum(np.abs(rem[A, cig]), np.abs(rem[A, top]))
                alpha = 0.1
                k = math.ceil((len(cal) + 1) * (1 - alpha))
                eps = float(np.partition(rsel[cal], k - 1)[k - 1]) if k <= len(cal) else float('inf')
                covered = rsel[ev] <= eps
                certified = gamma[ev] > 2 * eps
                assert np.all(advantage[ev][covered] >= gamma[ev][covered] - 2 * eps - 0.0002)
                certifications.append(dict(model=model_name, reference=refname, N=n, epsilon=eps, coverage=float(covered.mean()), certified_n=int(certified.sum()), certified_rate=float(certified.mean()), certified_correct=float((advantage[ev][certified] > 0).mean()) if certified.any() else np.nan))
                for group, sel in [('ALL', np.ones(len(chosen), bool)), ('Q1', quartile == 1), ('Q2', quartile == 2), ('Q3', quartile == 3), ('Q4', quartile == 4), ('HIGH_GAMMA_CAL', gamma >= gamma_cut), ('LOW_GAMMA_CAL', gamma < gamma_cut)]:
                    ix = ev[sel[ev]]
                    if len(ix) < 12:
                        continue
                    ci0, ci1 = ci(advantage[ix])
                    rows.append(dict(model=model_name, reference=refname, N=n, group=group, n=len(ix), advantage=float(advantage[ix].mean()), ci_low=ci0, ci_high=ci1, CIG_correction=float(correct[ix, cig[ix]].mean()), TopN_correction=float(correct[ix, top[ix]].mean()), corrected_delta=float(correct[ix, cig[ix]].mean() - correct[ix, top[ix]].mean()), CIG_improved=float((delta[ix, cig[ix]] > 1e-06).mean()), TopN_improved=float((delta[ix, top[ix]] > 1e-06).mean()), different_selection=float((cig[ix] != top[ix]).mean()), gamma_mean=float(gamma[ix].mean()), strength_mean=float(strength[ix].mean())))
                for co, ix in [('CAL', cal), ('EVAL', ev)]:
                    fidelity.append(dict(model=model_name, reference=refname, N=n, cohort=co, node_MAE=float(np.abs(delta[ix][:, candidates] - ap1[ix][:, candidates]).mean()), CIG_MAE=float(np.abs(delta[ix][:, candidates] - ap2[ix][:, candidates]).mean()), CIG_RMSE=float(np.sqrt(np.mean((delta[ix][:, candidates] - ap2[ix][:, candidates]) ** 2)))))
                srng = np.random.default_rng(SEED + 23 + n)
                for rep in range(10):
                    perm = np.argsort(srng.random(inter.shape), axis=1)
                    shinter = np.take_along_axis(inter, perm, axis=1)
                    shscore = ap1 + shinter @ inc_pair.T
                    sh = candidates[shscore[:, candidates].argmax(1)]
                    controls.append(dict(model=model_name, reference=refname, N=n, shuffle=rep, advantage=float(np.mean(delta[ev, cig[ev]] - delta[ev, sh[ev]])), correction_delta=float(np.mean(correct[ev, cig[ev]]) - np.mean(correct[ev, sh[ev]]))))
                if n == FOCAL_N:
                    subgroup.append(pd.DataFrame(dict(model=model_name, reference=refname, heldout_index=chosen[ev], true_label=iy[ev], original_prediction=ip[ev], N=n, quartile=quartile[ev], strength=strength[ev], gamma=gamma[ev], gamma_cut=gamma_cut, advantage=advantage[ev], TopN_set=top[ev], CIG_set=cig[ev], TopN_corrected=correct[ev, top[ev]], CIG_corrected=correct[ev, cig[ev]], TopN_delta=delta[ev, top[ev]], CIG_delta=delta[ev, cig[ev]], CIG_residual=rem[ev, cig[ev]], TopN_residual=rem[ev, top[ev]])))
            del lg
        del net
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    results = pd.DataFrame(rows)
    fid = pd.DataFrame(fidelity)
    certificate = pd.DataFrame(certifications)
    ctrl = pd.DataFrame(controls)
    results.to_csv(OUT / 'robustness_main.csv', index=False)
    fid.to_csv(OUT / 'approximation_fidelity.csv', index=False)
    certificate.to_csv(OUT / 'selected_pair_conformal.csv', index=False)
    ctrl.to_csv(OUT / 'shuffled_graph_control.csv', index=False)
    pd.concat(subgroup, ignore_index=True).to_csv(OUT / 'idip_focal_eval_per_image.csv', index=False)
    (OUT / 'protocol.json').write_text(json.dumps(dict(seed=SEED, reference_types=REFERENCES, dataset='CIFAR-10 train split: frozen original train/heldout partition; official test not used', calibration='Half of sampled model-specific heldout errors; prospective strength quartiles and gamma threshold', intervention='eight equal-area nonoverlapping grid regions; 256 subsets; three replacement references', model_control='trained-from-scratch TinyCNN separate architecture, same training examples', target='oracle true-class logit minus initial incorrectly predicted class logit, fixed across edits', correctness='edited argmax across all 10 classes equals true class', conformal='selected-pair max residual on calibration, 90% marginal under exchangeability; candidate selection based on singles/pairs only', caveats=['No semantic concept discovery; grid regions are visual units.', 'Reference interventions can still create off-manifold artifacts.', 'Model-specific error cohorts differ; do not compare absolute performance across models.', 'Validation cohort was involved in previous exploratory work; independent confirmation needed.', 'This is true-label oracle diagnosis, not deployable automatic correction.', 'Conformal is probabilistic marginal, not per-image deterministic.']), indent=2))
    print('Results:', OUT)
if __name__ == '__main__':
    main()
