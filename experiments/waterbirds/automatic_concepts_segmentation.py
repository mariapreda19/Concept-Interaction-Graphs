
import os, glob, json, random, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

SEED=2026
N_CONCEPTS=12
N_DISCOVERY=1000
N_SEMANTIC=250
TOP_K=4
BATCH_SIZE=48
MAX_VAL=None
MAX_TEST=None
SAVE=Path('/kaggle/working/waterbirds_auto_concepts')
SAVE.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',DEVICE)

roots=[Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),
       Path('/kaggle/input/waterbird/waterbird'),Path('/kaggle/input/waterbird')]
root=next((p for p in roots if (p/'metadata.csv').exists()),None)
if root is None:
    hits=glob.glob('/kaggle/input/**/metadata.csv',recursive=True)
    if not hits: raise FileNotFoundError('metadata.csv not found in /kaggle/input')
    root=Path(hits[0]).parent
print('Dataset:',root)
meta=pd.read_csv(root/'metadata.csv').copy()
for c in ['img_filename','y','split','place']:
    if c not in meta: raise ValueError(f'Missing metadata column: {c}')
meta['path']=meta.img_filename.map(lambda s: str(root/str(s)))
missing=meta.loc[~meta.path.map(os.path.isfile)]
if len(missing): raise FileNotFoundError(f'{len(missing)} images missing, e.g. {missing.path.iloc[0]}')

ckpt=Path('/kaggle/working/waterbirds_real1_20epochs/resnet18_waterbirds_best.pt')
if not ckpt.exists():
    candidates=glob.glob('/kaggle/input/**/resnet18_waterbirds_best.pt',recursive=True)
    if not candidates: raise FileNotFoundError('Upload the trained resnet18_waterbirds_best.pt to Kaggle input or use the existing /kaggle/working checkpoint')
    ckpt=Path(candidates[0])
print('CNN checkpoint:',ckpt)
model=models.resnet18(weights=None)
model.fc=nn.Linear(model.fc.in_features,2)
obj=torch.load(ckpt,map_location='cpu',weights_only=False)
state=obj.get('model_state_dict',obj.get('state_dict',obj)) if isinstance(obj,dict) else obj
if any(k.startswith('module.') for k in state): state={k.removeprefix('module.'):v for k,v in state.items()}
model.load_state_dict(state,strict=True)
model=model.to(DEVICE).eval()
for p in model.parameters(): p.requires_grad_(False)

pre=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
raw_pre=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

def load_img(path):
    with Image.open(path) as im: return pre(im.convert('RGB'))

def layer3_features(x):
    z=model.conv1(x); z=model.bn1(z); z=model.relu(z); z=model.maxpool(z)
    z=model.layer1(z); z=model.layer2(z); return model.layer3(z)

def tail(z):
    z=model.layer4(z); z=model.avgpool(z); z=torch.flatten(z,1); return model.fc(z)

def infer_features(rows,desc):
    for i in tqdm(range(0,len(rows),BATCH_SIZE),desc=desc):
        chunk=rows.iloc[i:i+BATCH_SIZE]
        x=torch.stack([load_img(p) for p in chunk.path]).to(DEVICE)
        with torch.inference_mode(): z=layer3_features(x)
        yield chunk,z

train=meta[meta.split==0].reset_index(drop=True)
val=meta[meta.split==1].reset_index(drop=True)
test=meta[meta.split==2].reset_index(drop=True)
if MAX_VAL is not None: val=val.sample(n=min(MAX_VAL,len(val)),random_state=SEED).reset_index(drop=True)
if MAX_TEST is not None: test=test.sample(n=min(MAX_TEST,len(test)),random_state=SEED).reset_index(drop=True)

sample=train.sample(n=min(N_DISCOVERY,len(train)),random_state=SEED).reset_index(drop=True)
patches=[]
for _,z in infer_features(sample,'Extract train CNN patches'):
    a=z.permute(0,2,3,1).reshape(-1,z.shape[1]).cpu().numpy().astype(np.float32)
    patches.append(a)
patches=np.concatenate(patches)
mu=patches.mean(axis=0); sd=np.maximum(patches.std(axis=0),1e-4)
Z=np.clip((patches-mu)/sd,-8,8)
km=MiniBatchKMeans(n_clusters=N_CONCEPTS,batch_size=4096,n_init=5,max_iter=120,random_state=SEED)
km.fit(Z)
centers=torch.tensor(km.cluster_centers_,device=DEVICE,dtype=torch.float32)
mu_t=torch.tensor(mu,device=DEVICE); sd_t=torch.tensor(sd,device=DEVICE)
print('Concept discovery complete:',N_CONCEPTS,'clusters; train patches:',len(patches))

def concept_assign(z):
    b,c,h,w=z.shape
    a=z.permute(0,2,3,1).reshape(-1,c)
    a=((a-mu_t)/sd_t).clamp(-8,8)
    d=(a*a).sum(1,keepdim=True)+(centers*centers).sum(1)[None,:]-2*a@centers.T
    return d.argmin(1).reshape(b,h,w)

try:
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    seg_weights=DeepLabV3_ResNet50_Weights.DEFAULT
    segmenter=deeplabv3_resnet50(weights=seg_weights).to(DEVICE).eval()
    for p in segmenter.parameters(): p.requires_grad_(False)
except Exception as e:
    raise RuntimeError('Cannot load DeepLabV3 pretrained VOC weights. In Kaggle enable Internet once to cache weights, or attach cached weights. This semantic step is mandatory; no invented labels.') from e

sem_sample=train.sample(n=min(N_SEMANTIC,len(train)),random_state=SEED+1).reset_index(drop=True)
bird_pixels=np.zeros(N_CONCEPTS,dtype=np.float64)
concept_pixels=np.zeros(N_CONCEPTS,dtype=np.float64)
examples={i:[] for i in range(N_CONCEPTS)}
for chunk,z in infer_features(sem_sample,'Semantic concept labeling'):
    with torch.inference_mode():
        labels=concept_assign(z)
        x=torch.stack([load_img(p) for p in chunk.path]).to(DEVICE)
        seg=segmenter(x)['out'].argmax(1)
        seg=F.interpolate(seg[:,None].float(),size=labels.shape[-2:],mode='nearest')[:,0].long()==3
        for k in range(N_CONCEPTS):
            mask=labels==k
            concept_pixels[k]+=mask.sum().item()
            bird_pixels[k]+=(mask&seg).sum().item()
        for j,(_,row) in enumerate(chunk.iterrows()):
            ids,counts=labels[j].unique(return_counts=True)
            for cid,count in zip(ids.tolist(),counts.tolist()):
                if len(examples[cid])<8: examples[cid].append({'path':row.path,'coverage':round(count/labels[j].numel(),3)})
bird_fraction=bird_pixels/np.maximum(concept_pixels,1)
semantic=pd.DataFrame({'concept':np.arange(N_CONCEPTS),'bird_fraction':bird_fraction,
                       'background_fraction':1-bird_fraction,'train_patch_count':concept_pixels.astype(int)})
semantic.to_csv(SAVE/'concept_semantics.csv',index=False)
with open(SAVE/'concept_examples.json','w') as f: json.dump(examples,f,indent=2)
print('\nSemantic concept proxy (fraction of concept locations overlapping a segmented bird):')
print(semantic.to_string(index=False,float_format=lambda a:f'{a:.3f}'))

fig,axes=plt.subplots(3,4,figsize=(14,10))
for ax,(_,row) in zip(axes.flat,sem_sample.head(12).iterrows()):
    with Image.open(row.path) as im: rgb=np.asarray(im.convert('RGB').resize((224,224)))
    with torch.inference_mode():
        z=layer3_features(load_img(row.path)[None].to(DEVICE))
        ids=concept_assign(z)[0].cpu().numpy()
    ids=np.asarray(Image.fromarray(ids.astype(np.uint8)).resize((224,224),Image.Resampling.NEAREST))
    ax.imshow(rgb); ax.imshow(ids,alpha=.32,cmap='tab20',vmin=0,vmax=max(19,N_CONCEPTS-1)); ax.axis('off')
plt.tight_layout(); fig.savefig(SAVE/'train_concept_maps.png',dpi=160); plt.close(fig)

def process_split(rows,desc):
    records=[]
    for chunk,z in infer_features(rows,desc):
        ids=concept_assign(z)
        b,c,h,w=z.shape
        with torch.inference_mode():
            logits=tail(z)
            margin=logits[:,1]-logits[:,0]
            prob=logits.softmax(1).max(1).values
            pred=logits.argmax(1)
            sign=torch.where(pred==1,1.,-1.)
            for j,(_,r) in enumerate(chunk.iterrows()):
                zi=z[j:j+1]
                base=float(margin[j]); s=float(sign[j]); predicted=int(pred[j]); confidence=float(prob[j])
                counts=torch.bincount(ids[j].flatten(),minlength=N_CONCEPTS)
                active=torch.nonzero(counts>0).flatten()
                selected=counts.argsort(descending=True)[:min(TOP_K,len(active))].tolist()
                sets=[()] + [(k,) for k in selected] + [(a,b) for ix,a in enumerate(selected) for b in selected[ix+1:]]
                stack=zi.expand(len(sets),-1,-1,-1).clone()
                for t,group in enumerate(sets[1:],1):
                    mask=torch.zeros((h,w),dtype=torch.bool,device=DEVICE)
                    for k in group: mask|=(ids[j]==k)
                    stack[t,:,mask]=0
                scores=[]
                for part in stack.split(24):
                    out=tail(part); scores.extend((out[:,1]-out[:,0]).tolist())
                delta={group:score-base for group,score in zip(sets,scores)}
                B={k:-s*delta[(k,)] for k in selected}
                pairs={(a,b):delta[(a,b)]-delta[(a,)]-delta[(b,)] for ix,a in enumerate(selected) for b in selected[ix+1:]}

                bg_support=sum((1-bird_fraction[k])*max(B[k],0) for k in selected)
                bird_opposition=sum(bird_fraction[k]*max(-B[k],0) for k in selected)
                bird_support=sum(bird_fraction[k]*max(B[k],0) for k in selected)
                cross_interaction=sum(abs(v)*abs(bird_fraction[a]-bird_fraction[b]) for (a,b),v in pairs.items())
                interaction_abs=sum(abs(v) for v in pairs.values())
                rec={'img_filename':r.img_filename,'y':int(r.y),'place':int(r.place),
                     'pred':predicted,'error':int(predicted!=int(r.y)),
                     'confidence':confidence,'margin_abs':abs(base),
                     'bg_support':bg_support,'bird_opposition':bird_opposition,
                     'bird_support':bird_support,'semantic_discrepancy':bg_support+bird_opposition,
                     'cross_interaction':cross_interaction,'interaction_abs':interaction_abs,
                     'n_active':len(active),'top_concepts':json.dumps(selected)}
                for k in range(N_CONCEPTS):
                    rec[f'B_{k}']=float(B.get(k,0.))
                    rec[f'coverage_{k}']=float(counts[k]/(h*w))
                records.append(rec)
    return pd.DataFrame(records)

val_df=process_split(val,'Build validation graphs')
val_df.to_csv(SAVE/'validation_graph_features.csv',index=False)
test_df=process_split(test,'Build test graphs (one-time evaluation)')
test_df.to_csv(SAVE/'test_graph_features.csv',index=False)

feature_sets={
    'confidence':['confidence'],
    'cnn_contributions':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)],
    'cnn_graph':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['interaction_abs'],
    'semantic_only':['semantic_discrepancy'],
    'semantic_components':['bg_support','bird_opposition','bird_support'],
    'semantic_plus_confidence':['confidence','semantic_discrepancy'],
    'cnn_plus_semantic':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['semantic_discrepancy'],
    'full_without_interaction':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['bg_support','bird_opposition','bird_support'],
    'full_with_interaction':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['bg_support','bird_opposition','bird_support','interaction_abs','cross_interaction'],
}
if val_df.error.nunique()<2 or test_df.error.nunique()<2: raise ValueError('Need both correct and incorrect predictions in validation and test')
rows_out=[]; predictions=test_df[['img_filename','y','place','pred','error']].copy()
for name,cols in feature_sets.items():
    if name=='confidence':
        p=1-test_df.confidence.to_numpy()
    elif name=='semantic_only':
        p=test_df.semantic_discrepancy.to_numpy()
    else:
        detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0,class_weight=None,random_state=SEED))
        detector.fit(val_df[cols].to_numpy(),val_df.error.to_numpy())
        p=detector.predict_proba(test_df[cols].to_numpy())[:,1]
    predictions[name]=p
    rows_out.append({'detector':name,'n_test':len(test_df),'AUROC':roc_auc_score(test_df.error,p),
                     'AUPRC':average_precision_score(test_df.error,p)})
results=pd.DataFrame(rows_out).sort_values('AUROC',ascending=False)
print('\n=== TEST RESULTS (do not tune on these) ===')
print(results.to_string(index=False,float_format=lambda a:f'{a:.6f}'))
results.to_csv(SAVE/'ablation_results.csv',index=False)
predictions.to_csv(SAVE/'test_detector_predictions.csv',index=False)

rng=np.random.default_rng(SEED)
y=test_df.error.to_numpy(); n=len(y)
for name_a,name_b in [('full_with_interaction','full_without_interaction'),
                      ('cnn_plus_semantic','cnn_contributions'),
                      ('semantic_plus_confidence','confidence')]:
    a=predictions[name_a].to_numpy(); b=predictions[name_b].to_numpy()
    auc_d=[]; ap_d=[]
    for _ in range(1000):
        ix=rng.integers(0,n,n)
        if np.unique(y[ix]).size<2: continue
        auc_d.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
        ap_d.append(average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],b[ix]))
    print(f'\n{name_a} - {name_b}')
    print(' AUROC diff CI:',np.quantile(auc_d,[.025,.5,.975]))
    print(' AUPRC diff CI:',np.quantile(ap_d,[.025,.5,.975]))

print('\n=== TEST SUBGROUP AUROC ===')
for (bird,place),g in predictions.groupby(['y','place']):
    if g.error.nunique()<2: continue
    print(f'bird={bird} place={place} n={len(g)} errors={int(g.error.sum())}',
          'confidence=',round(roc_auc_score(g.error,g.confidence),4),
          'semantic=',round(roc_auc_score(g.error,g.semantic_plus_confidence),4),
          'full=',round(roc_auc_score(g.error,g.full_with_interaction),4))

with open(SAVE/'run_config.json','w') as f:
    json.dump({'seed':SEED,'n_concepts':N_CONCEPTS,'n_discovery':N_DISCOVERY,
               'n_semantic':N_SEMANTIC,'top_k':TOP_K,'checkpoint':str(ckpt),
               'dataset':str(root),'n_val':len(val_df),'n_test':len(test_df),
               'semantic_model':'DeepLabV3 Pascal VOC bird mask; not VLM/NLP',
               'intervention':'zero layer3 spatial activations for each concept'},f,indent=2)
print('\nSaved outputs:',SAVE)
print('IMPORTANT: Concept labels are segmentation-based proxies. This is NOT a fully independent semantic relation model.')