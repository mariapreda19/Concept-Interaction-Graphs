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
from torchvision import transforms
from torchvision import models
from tqdm.auto import tqdm
import gc
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torch.utils.data import Dataset

def main():
    SEED = 20261024
    OUT = Path('/kaggle/working/cig_waterbirds_standalone')
    OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 224
    EPOCHS = 8
    TRAIN_BATCH = 32
    EVAL_BATCH = 32
    MASK_BATCH = 16
    MAX_EVAL_ERRORS = 200
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 0.0001
    NUM_WORKERS = 0
    RETRAIN = False
    MIN_BIRD_FRAC = 0.01
    MAX_BIRD_FRAC = 0.85
    MIN_NODE_FRAC = 0.005
    SHUFFLES = 20
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    candidates = [Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'), Path('/kaggle/input/waterbird/waterbird'), Path('/kaggle/input/waterbird')]
    candidates += [p.parent for p in Path('/kaggle/input').glob('**/metadata.csv')]
    ROOT = next((p for p in candidates if (p / 'metadata.csv').exists() and 'img_filename' in pd.read_csv(p / 'metadata.csv', nrows=1).columns), None)
    if ROOT is None:
        raise FileNotFoundError('Waterbirds not attached. Kaggle -> Add Input -> search bahardibaie/waterbird; metadata.csv required.')
    raw = pd.read_csv(ROOT / 'metadata.csv')
    assert {'img_filename', 'split', 'y'}.issubset(raw.columns)

    def resolve(rel):
        rel = Path(str(rel))
        for p in (ROOT / rel, ROOT / 'images' / rel, ROOT.parent / rel):
            if p.is_file():
                return str(p)
        return None
    raw['path'] = raw.img_filename.map(resolve)
    if raw.path.isna().any():
        raise FileNotFoundError(f"Failed to find {int(raw.path.isna().sum())} Waterbirds files under {ROOT}; example {raw.loc[raw.path.isna(), 'img_filename'].iloc[0]}")
    train = raw.loc[raw.split == 0].reset_index(drop=True)
    val = raw.loc[raw.split == 1].reset_index(drop=True)
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    val_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])

    class WaterbirdsDataset(Dataset):

        def __init__(self, df, tf):
            self.df = df
            self.tf = tf

        def __len__(self):
            return len(self.df)

        def __getitem__(self, k):
            r = self.df.iloc[k]
            with Image.open(r.path) as im:
                x = self.tf(im.convert('RGB'))
            return (x, int(r.y))
    train_dl = DataLoader(WaterbirdsDataset(train, train_tf), batch_size=TRAIN_BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda')
    val_dl = DataLoader(WaterbirdsDataset(val, val_tf), batch_size=EVAL_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda')
    CP = OUT / 'resnet18_waterbirds_trainonly.pt'
    PRETRAINED_BACKBONE = True
    TRAINED_THIS_RUN = not (CP.exists() and (not RETRAIN))
    if CP.exists() and (not RETRAIN):
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 2)
        state = torch.load(CP, map_location='cpu', weights_only=False)
        net.load_state_dict(state['state_dict'])
        net.to(DEVICE).eval()
    else:
        try:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            PRETRAINED_BACKBONE = False
            net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 2)
        net.to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for ep in range(1, EPOCHS + 1):
            net.train()
            correct = seen = 0
            loss_sum = 0.0
            for x, y in tqdm(train_dl, desc=f'TRAIN epoch {ep}/{EPOCHS}', leave=False):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = net(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                n = len(y)
                seen += n
                correct += int((logits.argmax(1) == y).sum())
                loss_sum += float(loss) * n
        torch.save({'state_dict': net.cpu().state_dict(), 'epochs': EPOCHS, 'pretrained_backbone': PRETRAINED_BACKBONE, 'seed': SEED, 'trained_on': 'Waterbirds split=0 only', 'architecture': 'torchvision resnet18'}, CP)
        net.to(DEVICE).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    val_predictions = []
    with torch.inference_mode():
        for x, y in tqdm(val_dl, desc='Original VAL predictions'):
            logits = net(x.to(DEVICE)).cpu().numpy()
            val_predictions.extend(logits.tolist())
    val_predictions = np.asarray(val_predictions, dtype=np.float32)
    label = val.y.to_numpy(dtype=int)
    pred = val_predictions.argmax(1)
    wrong_idx = np.flatnonzero(pred != label)
    if not len(wrong_idx):
        raise RuntimeError('No validation errors, cannot evaluate error diagnostics')
    chosen = np.random.default_rng(SEED + 491).permutation(wrong_idx)[:min(MAX_EVAL_ERRORS, len(wrong_idx))]
    try:
        SEG_WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT
        segmenter = deeplabv3_resnet50(weights=SEG_WEIGHTS).to(DEVICE).eval()
        seg_tf = SEG_WEIGHTS.transforms()
    except Exception as e:
        raise RuntimeError(f'Cannot load pretrained DeepLabV3 bird segmenter. Enable Kaggle Internet for first run or attach cached torchvision weights. The ResNet checkpoint was saved at {CP}. Original failure: {e}') from e
    for p in segmenter.parameters():
        p.requires_grad_(False)
    bird_frac = []
    paths = []
    ys = []
    ps = []
    images = []
    masks = []
    index_kept = []
    for idx in tqdm(chosen, desc='Segment bird and build four grounded regions'):
        rec = val.iloc[int(idx)]
        with Image.open(rec.path) as im:
            original_pil = im.convert('RGB')
            imgs = val_tf(original_pil)
            pil = transforms.ToPILImage()(torch.clamp(imgs * torch.tensor(NORM_STD)[:, None, None] + torch.tensor(NORM_MEAN)[:, None, None], 0, 1))
        with torch.inference_mode():
            out = segmenter(seg_tf(pil).unsqueeze(0).to(DEVICE))['out']
            out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        bird = (out.argmax(1)[0] == 3).cpu().numpy()
        area = float(bird.mean())
        if not MIN_BIRD_FRAC <= area <= MAX_BIRD_FRAC:
            continue
        rows = np.flatnonzero(bird.any(axis=1))
        if not len(rows):
            continue
        med = int(np.median(np.where(bird)[0]))
        yy = np.arange(IMG_SIZE)[:, None]
        scene = ~bird
        mask = np.stack((bird & (yy <= med), bird & (yy > med), scene & (yy < IMG_SIZE // 2), scene & (yy >= IMG_SIZE // 2)))
        if np.min(mask.mean(axis=(1, 2))) < MIN_NODE_FRAC:
            continue
        images.append(imgs)
        masks.append(mask.astype(np.float32))
        ys.append(int(rec.y))
        ps.append(int(pred[idx]))
        paths.append(str(rec.path))
        bird_frac.append(area)
        index_kept.append(int(idx))
    del segmenter
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    if len(images) < 15:
        raise RuntimeError('Too few usable bird masks. The segmenter may be unsuitable for this set; inspect mask quality.')
    node_count = 4
    codes = np.arange(1 << node_count)
    bits = codes[:, None] >> np.arange(node_count)[None, :] & 1
    intervened_logits = np.zeros((len(images), 16, 2), dtype=np.float32)
    with torch.inference_mode():
        for i in tqdm(range(len(images)), desc='Waterbirds 16 intervention subsets'):
            x = images[i].unsqueeze(0).to(DEVICE)
            mk = torch.tensor(masks[i], device=DEVICE)
            m = (torch.as_tensor(bits, device=DEVICE, dtype=torch.float32)[:, :, None, None] * mk[None]).amax(1)[:, None]
            edited = x * (1 - m)
            for k in range(0, 16, MASK_BATCH):
                intervened_logits[i, k:k + MASK_BATCH] = net(edited[k:k + MASK_BATCH]).cpu().numpy()
            original_idx = index_kept[i]
            max_abs_logit_diff = float(np.max(np.abs(intervened_logits[i, 0] - val_predictions[original_idx])))
            if max_abs_logit_diff > 0.002:
                raise AssertionError(f'Zero-mask preprocessing mismatch: val_index={original_idx}, max_abs_logit_diff={max_abs_logit_diff:.6g}')
            if np.argmax(intervened_logits[i, 0]) != ps[i]:
                raise AssertionError(f'Zero-mask prediction mismatch: val_index={original_idx}')
    np.savez_compressed(OUT / 'waterbirds_grounded_16_subsets.npz', logits=intervened_logits, y=np.asarray(ys), pred=np.asarray(ps), val_indices=np.asarray(index_kept), paths=np.asarray(paths, dtype=str), mask_areas=np.asarray([m.mean((1, 2)) for m in masks]))
    y = np.asarray(ys)
    p = np.asarray(ps)
    n = len(y)
    a = np.arange(n)
    q = intervened_logits[a[:, None], codes[None, :], y[:, None]] - intervened_logits[a[:, None], codes[None, :], p[:, None]]
    d = q - q[:, [0]]
    node = d[:, [1 << i for i in range(4)]]
    pairs = list(itertools.combinations(range(4), 2))
    pcode = np.array([(1 << i) + (1 << j) for i, j in pairs])
    edges = d[:, pcode] - node[:, [i for i, j in pairs]] - node[:, [j for i, j in pairs]]
    add = node @ bits.T
    pfeatures = np.stack([bits[:, i] * bits[:, j] for i, j in pairs], axis=1)
    second = add + edges @ pfeatures.T
    assert np.max(np.abs(d[:, np.bitwise_or.outer(2 ** np.arange(4), 2 ** np.arange(4)).reshape(-1)] - second[:, np.bitwise_or.outer(2 ** np.arange(4), 2 ** np.arange(4)).reshape(-1)])) < 0.0001, 'Pair identity failed'
    correct = intervened_logits.argmax(axis=2) == y[:, None]
    rng = np.random.default_rng(SEED + 202)
    rows = []
    imgs = []
    for budget in (2, 3):
        cand = np.flatnonzero(bits.sum(axis=1) == budget)
        top = cand[np.argmax(add[:, cand], axis=1)]
        cig = cand[np.argmax(second[:, cand], axis=1)]
        oracle = cand[np.argmax(d[:, cand], axis=1)]
        adv = d[a, cig] - d[a, top]
        regret = d[a, oracle] - d[a, cig]
        shuffled = []
        for rep in range(SHUFFLES):
            perm = np.stack([rng.permutation(len(pairs)) for _ in range(n)])
            e = edges[np.arange(n)[:, None], perm]
            s = add + e @ pfeatures.T
            chosen_shuffle = cand[np.argmax(s[:, cand], axis=1)]
            shuffled.append(d[a, cig] - d[a, chosen_shuffle])
        draws = np.stack([adv[rng.integers(0, n, n)].mean() for _ in range(1500)])
        ci = np.quantile(draws, [0.025, 0.975])
        rows.append(dict(budget=budget, n=n, mean_margin_adv=float(adv.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]), top_correction=float(correct[a, top].mean()), cig_correction=float(correct[a, cig].mean()), oracle_correction=float(correct[a, oracle].mean()), regret_CIG=float(regret.mean()), regret_TopN=float((d[a, oracle] - d[a, top]).mean()), oracle_match_CIG=float((cig == oracle).mean()), MAE_nodes=float(np.abs(d[:, cand] - add[:, cand]).mean()), MAE_CIG=float(np.abs(d[:, cand] - second[:, cand]).mean()), CIG_minus_shuffle=float(np.mean(shuffled)), different_selections=float((cig != top).mean())))
        imgs.append(pd.DataFrame(dict(budget=budget, val_index=index_kept, selected_CIG=cig, selected_TopN=top, selected_oracle=oracle, margin_adv=adv, regret=regret, CIG_correct=correct[a, cig].astype(int), TopN_correct=correct[a, top].astype(int))))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'C_grounded_bird_scene_selection.csv', index=False)
    pd.concat(imgs, ignore_index=True).to_csv(OUT / 'C_grounded_bird_scene_per_image.csv', index=False)
    pd.DataFrame({'val_index': index_kept, 'path': paths, 'bird_fraction': bird_frac}).to_csv(OUT / 'C_mask_quality.csv', index=False)
    (OUT / 'C_PROTOCOL.json').write_text(json.dumps({'dataset_root': str(ROOT), 'train_rows': len(train), 'validation_rows': len(val), 'checkpoint': str(CP), 'trained_in_this_run': TRAINED_THIS_RUN, 'training_epochs': EPOCHS, 'no_val_checkpoint_selection': True, 'original_val_errors': len(wrong_idx), 'selected_errors': len(chosen), 'usable_masks': n, 'node_names': ['bird_upper', 'bird_lower', 'background_upper', 'background_lower'], 'intervention_reference': 'ImageNet channel means in RGB (0 in normalized space)', 'limitations': 'Segmentation is pretrained DeepLabV3 VOC bird; slices are spatial, not four independently annotated concepts. Oracle label used for target margin. Validation cohort used for exploratory assessment. Official TEST untouched.'}, indent=2))
    print('Results:', OUT)
if __name__ == '__main__':
    main()
