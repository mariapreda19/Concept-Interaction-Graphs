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

def main():
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
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        original_logits = []
        original_labels = []
        for x, y in tqdm(val_loader, desc='Original classifier on untouched heldout'):
            original_logits.append(net(x.to(DEVICE)).cpu().numpy())
            original_labels.append(y.numpy())
    original_logits = np.concatenate(original_logits).astype(np.float32)
    original_labels = np.concatenate(original_labels).astype(np.int64)
    original_pred = original_logits.argmax(axis=1)
    error_all = np.flatnonzero(original_pred != original_labels)
    if len(error_all) < 100:
        raise RuntimeError('Too few errors for this diagnostic; train fewer epochs or use a harder CNN.')
    chosen = np.random.default_rng(SEED + 3).permutation(error_all)[:min(ERRORS_TOTAL, len(error_all))]
    if len(chosen) < 120:
        raise RuntimeError('Need at least 120 held-out classification errors.')
    cal_n = len(chosen) // 2
    cal_positions = np.arange(cal_n)
    eval_positions = np.arange(cal_n, len(chosen))
    subsets = np.arange(1 << N_REGIONS, dtype=np.int64)
    incidence = (subsets[:, None] >> np.arange(N_REGIONS) & 1).astype(np.float32)
    card = incidence.sum(axis=1).astype(int)
    mask = torch.zeros((1 << N_REGIONS, 1, 32, 32), dtype=torch.float32)
    for s in subsets:
        for k in range(N_REGIONS):
            if incidence[s, k]:
                row, col = divmod(k, 4)
                mask[s, 0, row * 16:(row + 1) * 16, col * 8:(col + 1) * 8] = 1
    mask = mask.to(DEVICE)
    mask_np = mask[:, 0].cpu().numpy()
    assert (mask[0] == 0).all() and (mask[-1] == 1).all()
    cache_valid = False
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        cache_valid = np.array_equal(z['chosen'], chosen) and int(z['n_regions']) == N_REGIONS and (str(z['checkpoint_sha']) == __import__('hashlib').sha256(CKPT.read_bytes()).hexdigest())
        if cache_valid:
            all_logits = z['logits'].astype(np.float32)
    if not cache_valid:
        all_logits = np.empty((len(chosen), len(subsets), 10), dtype=np.float32)
        with torch.inference_mode():
            for t, pos in enumerate(tqdm(chosen, desc='ALL 256 intervention subsets per selected error')):
                image, _ = val_dataset[int(held_ids[pos])]
                x = image[None].to(DEVICE)
                reference = x.mean(dim=(2, 3), keepdim=True)
                for a in range(0, len(subsets), INFER_BATCH):
                    m = mask[a:a + INFER_BATCH]
                    altered = x * (1 - m) + reference * m
                    all_logits[t, a:a + len(m)] = net(altered).cpu().numpy()
        np.savez_compressed(CACHE, logits=all_logits, chosen=chosen, n_regions=np.array(N_REGIONS), checkpoint_sha=np.array(__import__('hashlib').sha256(CKPT.read_bytes()).hexdigest()))
    y = original_labels[chosen]
    pred = original_pred[chosen]
    assert np.all(y != pred)
    rows = np.arange(len(chosen))[:, None]
    q_fixed = all_logits[np.arange(len(chosen))[:, None], np.arange(len(subsets))[None, :], y[:, None]] - all_logits[np.arange(len(chosen))[:, None], np.arange(len(subsets))[None, :], pred[:, None]]
    comp = all_logits.copy()
    comp[np.arange(len(chosen))[:, None], np.arange(len(subsets))[None, :], y[:, None]] = -np.inf
    q_true = all_logits[np.arange(len(chosen))[:, None], np.arange(len(subsets))[None, :], y[:, None]] - comp.max(axis=2)
    delta = q_fixed - q_fixed[:, [0]]
    real_margin_gain = q_true - q_true[:, [0]]
    corrected = all_logits.argmax(axis=-1) == y[:, None]
    assert np.all(~corrected[:, 0])
    assert np.all(~corrected | (delta > 1e-06)), 'Every corrected example must improve fixed target margin!'
    assert np.all(~corrected | (real_margin_gain > 1e-06)), 'Every corrected example must improve true multiclass margin!'
    bits = np.arange(N_REGIONS)
    single_codes = (1 << bits).astype(int)
    pairs = list(itertools.combinations(bits, 2))
    pair_codes = np.array([1 << i | 1 << j for i, j in pairs], dtype=int)
    single = delta[:, single_codes]
    I = np.column_stack([delta[:, code] - single[:, i] - single[:, j] for code, (i, j) in zip(pair_codes, pairs)])
    approx1 = incidence @ single.T
    approx2 = approx1.copy()
    for col, (i, j) in enumerate(pairs):
        approx2 += (incidence[:, i] * incidence[:, j])[:, None] * I[:, col][None, :]
    approx1 = approx1.T
    approx2 = approx2.T
    assert np.max(np.abs(approx2[:, card == 2] - delta[:, card == 2])) < 0.0002
    remainder = delta - approx2
    choices = {}
    for n in BUDGETS:
        cand = np.flatnonzero(card == n)
        top_idx = cand[np.argmax(approx1[:, cand], axis=1)]
        cig_idx = cand[np.argmax(approx2[:, cand], axis=1)]
        choices[n] = {'top': top_idx, 'cig': cig_idx, 'cand': cand}
        assert np.all(card[top_idx] == n) and np.all(card[cig_idx] == n)
        if n == 2:
            assert np.all(delta[np.arange(len(y)), cig_idx] >= delta[np.arange(len(y)), top_idx] - 0.0002)
    allowed = np.isin(card, BUDGETS)
    max_err = np.abs(remainder[:, allowed]).max(axis=1)
    cal_vals = max_err[cal_positions]
    k = math.ceil((cal_n + 1) * (1 - ALPHA))
    epsilon = float(np.partition(cal_vals, k - 1)[k - 1]) if k <= cal_n else float('inf')
    coverage = float(np.mean(max_err[eval_positions] <= epsilon))
    focal = choices[ANALYSIS_BUDGET]
    gamma = approx2[np.arange(len(y)), focal['cig']] - approx2[np.arange(len(y)), focal['top']]
    strength = np.sum(np.abs(I), axis=1) / (np.sum(np.abs(single), axis=1) + 1e-06)
    quartile_edges = np.quantile(strength[cal_positions], [0.25, 0.5, 0.75])
    quartile = np.searchsorted(quartile_edges, strength, side='right') + 1
    rng = np.random.default_rng(SEED + 19)

    def boot_ci(diff):
        a = np.asarray(diff, dtype=float)
        if len(a) < 8:
            return (np.nan, np.nan)
        draws = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        means = a[draws].mean(axis=1)
        return tuple(np.quantile(means, [0.025, 0.975]))
    records = []
    per_image = []
    shuffle_records = []
    null_rng = np.random.default_rng(SEED + 29)
    for n in BUDGETS:
        top = choices[n]['top']
        cig = choices[n]['cig']
        cand = choices[n]['cand']
        actual_top = delta[np.arange(len(y)), top]
        actual_cig = delta[np.arange(len(y)), cig]
        true_top = real_margin_gain[np.arange(len(y)), top]
        true_cig = real_margin_gain[np.arange(len(y)), cig]
        corr_top = corrected[np.arange(len(y)), top]
        corr_cig = corrected[np.arange(len(y)), cig]
        df = pd.DataFrame(dict(index_in_heldout=chosen, source_y=y, predicted=pred, cohort=np.where(np.arange(len(y)) < cal_n, 'calibration', 'evaluation'), N=n, strength=strength, quartile=quartile, gamma=gamma, epsilon=epsilon, TopN_set=top, CIG_set=cig, selection_differs=top != cig, TopN_delta_fixed=actual_top, CIG_delta_fixed=actual_cig, TopN_delta_true=true_top, CIG_delta_true=true_cig, TopN_corrected=corr_top, CIG_corrected=corr_cig, TopN_improved_fixed=actual_top > 1e-06, CIG_improved_fixed=actual_cig > 1e-06, approx2_CIG=approx2[np.arange(len(y)), cig], approximation_error_CIG=remainder[np.arange(len(y)), cig]))
        per_image.append(df)
        for cohort, coh_mask in [('evaluation', np.arange(len(y)) >= cal_n)]:
            for group, selector in [('ALL', np.ones(len(y), dtype=bool)), *[(f'Q{q}', quartile == q) for q in range(1, 5)], ('different_selections', top != cig), ('gamma_gt_2eps', gamma > 2 * epsilon)]:
                sel = coh_mask & selector
                if sel.sum() < 8:
                    continue
                advantage = (actual_cig - actual_top)[sel]
                lo, hi = boot_ci(advantage)
                records.append(dict(N=n, group=group, n=int(sel.sum()), mean_margin_adv=float(advantage.mean()), delta_CI_low=lo, delta_CI_high=hi, pct_CIG_wins=float(np.mean(advantage > 1e-06) * 100), pct_TopN_wins=float(np.mean(advantage < -1e-06) * 100), pct_ties=float(np.mean(np.abs(advantage) <= 1e-06) * 100), CIG_margin_improved_pct=float(np.mean(actual_cig[sel] > 1e-06) * 100), TopN_margin_improved_pct=float(np.mean(actual_top[sel] > 1e-06) * 100), CIG_errors_corrected_pct=float(np.mean(corr_cig[sel]) * 100), TopN_errors_corrected_pct=float(np.mean(corr_top[sel]) * 100), CIG_mean_abs_resid=float(np.abs(remainder[np.arange(len(y)), cig])[sel].mean()), node_only_MAE=float(np.abs((delta - approx1)[sel][:, cand]).mean()), second_order_MAE=float(np.abs((delta - approx2)[sel][:, cand]).mean()), fraction_different=float(np.mean((top != cig)[sel]))))
        for ksh in range(N_SHUFFLES):
            perm = np.argsort(null_rng.random(I.shape), axis=1)
            shuffled_I = np.take_along_axis(I, perm, axis=1)
            surr = approx1.copy()
            for edge, (i, j) in enumerate(pairs):
                surr += (incidence[:, i] * incidence[:, j])[None, :] * shuffled_I[:, edge, None]
            sh_choice = cand[np.argmax(surr[:, cand], axis=1)]
            held = eval_positions
            sh_adv = delta[held, cig[held]] - delta[held, sh_choice[held]]
            shuffle_records.append(dict(N=n, shuffle=ksh, CIG_minus_shuffled_mean=float(sh_adv.mean()), CIG_minus_shuffled_correction=float(corrected[held, cig[held]].mean() - corrected[held, sh_choice[held]].mean()), shuffled_mean_gain=float(delta[held, sh_choice[held]].mean())))
    per_image = pd.concat(per_image, ignore_index=True)
    summary = pd.DataFrame(records)
    shuffles = pd.DataFrame(shuffle_records)
    summary.to_csv(OUT / 'idip_main_results.csv', index=False)
    per_image.to_csv(OUT / 'idip_per_image.csv', index=False)
    shuffles.to_csv(OUT / 'idip_shuffled_edge_control.csv', index=False)
    fidelity = []
    for n in range(2, 7):
        where = card == n
        for cohort, ids in [('calibration', cal_positions), ('evaluation', eval_positions)]:
            err1 = delta[ids][:, where] - approx1[ids][:, where]
            err2 = delta[ids][:, where] - approx2[ids][:, where]
            fidelity.append(dict(N=n, cohort=cohort, MAE_nodes=float(np.abs(err1).mean()), MAE_CIG=float(np.abs(err2).mean()), RMSE_nodes=float(np.sqrt(np.mean(err1 ** 2))), RMSE_CIG=float(np.sqrt(np.mean(err2 ** 2))), signed_remainder_mean=float(err2.mean())))
    pd.DataFrame(fidelity).to_csv(OUT / 'idip_approximation_fidelity.csv', index=False)
    triples = list(itertools.combinations(range(N_REGIONS), 3))
    triple_out = []
    for i, j, k3 in triples:
        code = 1 << i | 1 << j | 1 << k3
        triple_interaction = remainder[:, code]
        triple_out.append(dict(triple=f'{i}-{j}-{k3}', mean_abs=float(np.abs(triple_interaction[eval_positions]).mean()), mean_signed=float(triple_interaction[eval_positions].mean())))
    pd.DataFrame(triple_out).to_csv(OUT / 'idip_third_order_interactions.csv', index=False)
    eval_cig = focal['cig'][eval_positions]
    eval_top = focal['top'][eval_positions]
    actual_adv = delta[eval_positions, eval_cig] - delta[eval_positions, eval_top]
    theory_bound = gamma[eval_positions] - 2 * epsilon
    covered = max_err[eval_positions] <= epsilon
    assert np.all(actual_adv[covered] >= theory_bound[covered] - 0.0002)
    cert = gamma[eval_positions] > 2 * epsilon
    certificate = dict(calibration_count=cal_n, evaluation_count=len(eval_positions), alpha=ALPHA, simultaneous_error_bound=epsilon, empirical_coverage=coverage, certified_fraction=float(cert.mean()), certified_n=int(cert.sum()), certified_success_rate=float(np.mean(actual_adv[cert] > 0)) if cert.any() else None, caveat='90% marginal, image-wise simultaneous split-conformal guarantee under exchangeability; not deterministic per-image guarantee')
    (OUT / 'idip_protocol.json').write_text(json.dumps(dict(seed=SEED, epochs=EPOCHS, checkpoint=str(CKPT), dataset='CIFAR-10 training images; 45k fitting, 5k heldout, no official test', sampled_errors=len(chosen), n_regions=N_REGIONS, budgets=BUDGETS, analysis_budget=ANALYSIS_BUDGET, intervention='8 nonoverlapping fixed rectangular RGB regions, each replaced by per-image channel mean', objective='oracle true-label logit minus original incorrect predicted-class logit (fixed competitor)', corrected='argmax of all ten edited logits matches true label', theory='second order exact for pairs; residual for 3+; split conformal max residual bound', quartile_thresholds=quartile_edges.tolist(), certificate=certificate, limitations=['Grid regions are not validated semantic concepts.', 'Uses true class for selecting edits; offline diagnostic only.', 'Per-image mean replacement can make off-distribution artifacts.', 'Calibration and evaluation errors come from same CIFAR training distribution.', 'All subset edits are exhaustively evaluated, including multi-concept interventions.', 'Reference and architecture are held fixed; robustness remains a future experiment.']), indent=2))
    cols = ['N', 'group', 'n', 'mean_margin_adv', 'delta_CI_low', 'delta_CI_high', 'CIG_margin_improved_pct', 'TopN_margin_improved_pct', 'CIG_errors_corrected_pct', 'TopN_errors_corrected_pct', 'fraction_different']
    for n in BUDGETS:
        sub = per_image[(per_image.N == n) & (per_image.cohort == 'evaluation')]
        for mode in ('CIG', 'TopN'):
            assert (~sub[f'{mode}_corrected'] | sub[f'{mode}_improved_fixed']).all()
    print('Results:', OUT)
if __name__ == '__main__':
    main()
