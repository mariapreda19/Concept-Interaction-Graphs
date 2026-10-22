import os, glob, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

SEED=42
N_TRAIN=1200
N_VAL=220
N_DONORS=2
SIZE=224
BATCH=16
MIN_BIRD=0.012
MAX_BIRD=0.75
MIN_CHANGED_BG=0.15
BOOTSTRAPS=1000
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
OUT=Path('/kaggle/working/waterbirds_v9_counterfactual_graph');OUT.mkdir(parents=True,exist_ok=True)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
if DEVICE=='cuda':torch.cuda.manual_seed_all(SEED)
print('DEVICE:',DEVICE,'OUTPUT:',OUT)
ROOT=next((p for p in [Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),Path('/kaggle/input/waterbird/waterbird'),Path('/kaggle/input/waterbird')] if (p/'metadata.csv').exists()),None)
if ROOT is None:
    candidates=[Path(s).parent for s in glob.glob('/kaggle/input/**/metadata.csv',recursive=True) if 'waterbird' in s.lower()]
    if not candidates:raise FileNotFoundError('Waterbirds metadata.csv missing')
    ROOT=candidates[0]
meta=pd.read_csv(ROOT/'metadata.csv')
assert {'y','place','split','img_filename'}.issubset(meta.columns)
train=meta[meta['split']==0].sample(frac=1,random_state=SEED).head(N_TRAIN).reset_index(drop=True)
valid=meta[meta['split']==1].sample(frac=1,random_state=SEED).head(N_VAL).reset_index(drop=True)

def image_file(name):
    p=Path(str(name))
    for candidate in [ROOT/p,ROOT/'dataset'/p]:
        if candidate.is_file():return candidate
    raise FileNotFoundError(str(name))

def rgb(name):
    with Image.open(image_file(name)) as im:
        return np.asarray(im.convert('RGB').resize((SIZE,SIZE),resample=Image.Resampling.BILINEAR))

ckpts=[Path('/kaggle/working/waterbirds_human_concepts_v2/resnet18_waterbirds_best.pt'),Path('/kaggle/working/waterbirds_human_concepts/resnet18_waterbirds_best.pt')]
ckpts += [Path(p) for p in glob.glob('/kaggle/input/**/resnet18_waterbirds_best.pt',recursive=True)]
ckpt=next((p for p in ckpts if p.is_file()),None)
if ckpt is None:raise FileNotFoundError('Required fixed ResNet18 checkpoint not found')
cnn=models.resnet18(weights=None);cnn.fc=torch.nn.Linear(cnn.fc.in_features,2)
state=torch.load(ckpt,map_location='cpu',weights_only=False)
if isinstance(state,dict):
    for key in ['model_state_dict','state_dict','model']:
        if isinstance(state.get(key),dict):state=state[key];break
state={k.removeprefix('module.'):v for k,v in state.items()}
cnn.load_state_dict(state,strict=True);cnn.to(DEVICE).eval()
for param in cnn.parameters():param.requires_grad_(False)
segmenter=deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(DEVICE).eval()
for param in segmenter.parameters():param.requires_grad_(False)
normalizer=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def tensor(images):
    a=np.stack(images)
    return normalizer(torch.from_numpy(a.copy()).permute(0,3,1,2).float().to(DEVICE)/255.0)

def margins(images):
    result=[]
    for start in range(0,len(images),BATCH):
        with torch.inference_mode():z=cnn(tensor(images[start:start+BATCH]))
        result.extend((z[:,1]-z[:,0]).float().cpu().tolist())
    return np.asarray(result,dtype=float)

def masks(images):
    result=[]
    for start in range(0,len(images),BATCH):
        with torch.inference_mode():logits=segmenter(tensor(images[start:start+BATCH]))['out']
        result.extend(((logits.softmax(1)[:,3]>=0.5)&(logits.argmax(1)==3)).cpu().numpy())
    return result

def build_donor_bank(frame):
    bank={}
    for place in [0,1]:
        choices=frame[frame.place==place].sample(frac=1,random_state=SEED+place).head(24)
        if len(choices)<N_DONORS:raise ValueError('Insufficient donors for place '+str(place))
        bank[place]=[rgb(s) for s in choices.img_filename]
    return bank

fill=np.array([123,116,103],dtype=np.uint8)

def remove_bird(im,mask):
    out=im.copy();out[mask]=fill;return out

def counterfactual(source,donor,mask):
    out=donor.copy();out[mask]=source[mask];return out

def process_split(frame,split_name,save_examples):
    donor_bank=build_donor_bank(frame)
    records=[]; examples=0
    for start in tqdm(range(0,len(frame),BATCH),desc=split_name+' paired counterfactuals'):
        part=frame.iloc[start:start+BATCH]
        imgs=[rgb(n) for n in part.img_filename]
        segs=masks(imgs)
        for j,(original,mask) in enumerate(zip(imgs,segs)):
            item=part.iloc[j]
            area=float(mask.mean())
            if not (MIN_BIRD<=area<=MAX_BIRD):continue
            options=donor_bank[1-int(item.place)]
            rng=np.random.default_rng(SEED + (0 if split_name=='train' else 100000) + start+j)
            choices=rng.choice(len(options),size=N_DONORS,replace=False)
            cfs=[];changes=[]
            for ix in choices:
                cf=counterfactual(original,options[int(ix)],mask)
                diff=np.abs(cf.astype(np.float32)-original.astype(np.float32)).mean(axis=2)/255
                changed=float(np.mean(diff[~mask]>0.08))
                if changed>=MIN_CHANGED_BG:
                    cfs.append(cf);changes.append(changed)
            if not cfs:continue
            original_no_bird=remove_bird(original,mask)
            batch=[original,original_no_bird]
            for cf in cfs:batch.extend([cf,remove_bird(cf,mask)])
            z=margins(batch);m0,m_no_bird=z[:2]
            sign=1. if m0>0 else -1.
            if split_name=='validation' and save_examples and examples<8:
                montage=np.concatenate([original,cfs[0]],axis=1)
                Image.fromarray(montage).save(OUT/('example_swap_'+str(examples).zfill(2)+'.png'))
                examples+=1
            individual=[];sens=[];interactions=[];swapped_wrong=[];flips=[]
            for k in range(len(cfs)):
                mc=z[2+2*k];mc_no_bird=z[3+2*k]
                interaction=(m0-m_no_bird)-(mc-mc_no_bird)
                interactions.append(float(sign*interaction))
                sens.append(float(abs(m0-mc)))
                individual.append(float(sign*(m0-m_no_bird)))
                swapped_wrong.append(int((mc>0)!=int(item.y)))
                flips.append(int((mc>0)!=(m0>0)))
            records.append(dict(split=split_name,source_index=int(start+j),y=int(item.y),place=int(item.place),
                 original_prediction=int(m0>0),original_error=int((m0>0)!=int(item.y)),
                 margin=float(m0),confidence=float(-abs(m0)),bird_area=area,
                 n_donors=len(cfs),mean_changed_bg=float(np.mean(changes)),
                 background_sensitivity=float(np.mean(sens)),max_background_sensitivity=float(np.max(sens)),
                 bird_support=float(np.mean(individual)),
                 interaction_signed=float(np.mean(interactions)),
                 interaction_abs=float(np.mean(np.abs(interactions))),
                 interaction_variability=float(np.std(interactions)),
                 swap_flip_rate=float(np.mean(flips)),swap_error_rate=float(np.mean(swapped_wrong)),
                 any_swap_flip=int(any(flips)),any_swap_error=int(any(swapped_wrong))))
    return pd.DataFrame(records)

tr=process_split(train,'train',False)
va=process_split(valid,'validation',True)
if len(tr)<100 or len(va)<70 or tr.original_error.nunique()<2 or va.original_error.nunique()<2:
    raise RuntimeError('Insufficient valid samples or error variation; inspect masks and dataset')
tr.to_csv(OUT/'train_counterfactual_features.csv',index=False)
va.to_csv(OUT/'validation_counterfactual_features.csv',index=False)

base=['confidence']
node=['confidence','bird_area','bird_support']
sensitivity=node+['background_sensitivity','max_background_sensitivity']
graph=sensitivity+['interaction_signed','interaction_abs','interaction_variability']
columns={'confidence':base,'nodes':node,'background_sensitivity':sensitivity,'graph_plus_interaction':graph}

def fit_predict(cols):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced',random_state=SEED))
    model.fit(tr[cols],tr.original_error.astype(int))
    return model.predict_proba(va[cols])[:,1]

scores={'confidence_untrained':va.confidence.to_numpy()}
for name,cols in columns.items():scores[name]=fit_predict(cols)
rng=np.random.default_rng(SEED+100)
permtr=tr.copy();permva=va.copy();interaction_cols=graph[len(sensitivity):]
for df in [permtr,permva]:
    perm=rng.permutation(len(df))
    df.loc[:,interaction_cols]=df[interaction_cols].to_numpy()[perm]
null=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced',random_state=SEED))
null.fit(permtr[graph],permtr.original_error.astype(int))
scores['shuffled_interaction']=null.predict_proba(permva[graph])[:,1]

def metric(y,p):
    if np.unique(y).size<2:return {'AUROC':None,'AUPRC':None}
    return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p))}

def paired_ci(y,a,b,seed):
    rng=np.random.default_rng(seed)
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    draws=[]
    for k in range(BOOTSTRAPS):
        ids=np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)])
        draws.append([roc_auc_score(y[ids],a[ids])-roc_auc_score(y[ids],b[ids]),
            average_precision_score(y[ids],a[ids])-average_precision_score(y[ids],b[ids])])
    draws=np.asarray(draws)
    return dict(AUROC_delta=float(roc_auc_score(y,a)-roc_auc_score(y,b)),
       AUROC_CI95=np.quantile(draws[:,0],[.025,.975]).round(6).tolist(),
       AUPRC_delta=float(average_precision_score(y,a)-average_precision_score(y,b)),
       AUPRC_CI95=np.quantile(draws[:,1],[.025,.975]).round(6).tolist())

y=va.original_error.astype(int).to_numpy()
report={'protocol':{'train_partition':0,'validation_partition':1,'test_accessed':False,
      'n_train':len(tr),'train_errors':int(tr.original_error.sum()),'n_val':len(va),
      'val_errors':int(y.sum()),'donors_per_source_attempted':N_DONORS,
      'source_and_donor_same_partition':True,'donor_place':'opposite',
      'fixed_cnn':str(ckpt),'masker':'DeepLabV3 bird, approximate',
      'warning':'Counterfactual swaps can include donor birds, seams and distribution shifts. Results are exploratory, not proof of causal or semantic correctness. Validation has been inspected repeatedly.'},
      'metrics':{},'paired':{},'swap_diagnostics':{},'subgroup':{}}
for name,p in scores.items():report['metrics'][name]=metric(y,p)
report['paired']['graph_minus_background_sensitivity']=paired_ci(y,scores['graph_plus_interaction'],scores['background_sensitivity'],SEED+1)
report['paired']['graph_minus_nodes']=paired_ci(y,scores['graph_plus_interaction'],scores['nodes'],SEED+2)
report['paired']['graph_minus_shuffled']=paired_ci(y,scores['graph_plus_interaction'],scores['shuffled_interaction'],SEED+3)
report['paired']['background_sensitivity_minus_confidence']=paired_ci(y,scores['background_sensitivity'],scores['confidence_untrained'],SEED+4)
report['swap_diagnostics']={'original_error_rate':float(va.original_error.mean()),
    'mean_swap_flip_rate':float(va.swap_flip_rate.mean()),'mean_swap_error_rate':float(va.swap_error_rate.mean()),
    'mean_background_change_fraction':float(va.mean_changed_bg.mean())}
threshold=float(tr.interaction_abs.quantile(0.75));keep=va.interaction_abs.to_numpy()>=threshold
report['subgroup']={'train_q75_interaction_abs':threshold,'n':int(keep.sum()),'errors':int(y[keep].sum()),'metrics':{}}
for name,p in scores.items():report['subgroup']['metrics'][name]=metric(y[keep],p[keep])

prediction=va.copy()
for name,p in scores.items():prediction['score_'+name]=p
prediction.to_csv(OUT/'validation_predictions.csv',index=False)
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print('\nTRAIN',len(tr),'errors',int(tr.original_error.sum()),'VAL',len(va),'errors',int(y.sum()))
print('\nMETRICS\n',pd.DataFrame(report['metrics']).T.to_string())
print('\nPAIRED DELTAS\n',json.dumps(report['paired'],indent=2))
print('\nSWAP DIAGNOSTICS\n',json.dumps(report['swap_diagnostics'],indent=2))
print('\nDONE:',OUT)
print('Send report.json and validation_predictions.csv; optionally example_swap_00.png')