
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
SAVE=Path('/kaggle/working/waterbirds_auto_concepts_clip')
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

from transformers import CLIPModel, CLIPProcessor
CLIP_ID='openai/clip-vit-base-patch32'
try:
    clip=CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    clip_processor=CLIPProcessor.from_pretrained(CLIP_ID)
except Exception as e:
    raise RuntimeError(
        f'CLIP {CLIP_ID} could not load. Enable Kaggle Internet or attach a cached HuggingFace model.'
    ) from e
for p in clip.parameters(): p.requires_grad_(False)

LABELS=[
    ('bird feathers',1.0), ('bird wing',1.0), ('bird head',1.0),
    ('bird beak',1.0), ('bird body',1.0), ('bird legs',1.0),
    ('water surface',0.0), ('grass',0.0), ('tree branches',0.0),
    ('green leaves',0.0), ('blue sky',0.0), ('rocks',0.0),
    ('sand',0.0), ('forest',0.0), ('reeds',0.0), ('mud',0.0),
    ('clouds',0.0), ('shoreline',0.0), ('water reflections',0.0),
    ('background texture',0.0)
]
N_LABELS=len(LABELS)
label_names=[v[0] for v in LABELS]
label_relevance=np.array([v[1] for v in LABELS],dtype=np.float32)
single_prompts=[f'a close-up photograph of {name}' for name in label_names]
pair_prompts=[
    f'a photograph containing {label_names[i]} and {label_names[j]}'
    for i in range(N_LABELS) for j in range(i+1,N_LABELS)
]
pair_label_ids=[(i,j) for i in range(N_LABELS) for j in range(i+1,N_LABELS)]

def clip_text_vectors(texts):
    parts=[]
    for k in range(0,len(texts),64):
        inp=clip_processor(text=texts[k:k+64],padding=True,truncation=True,
                           return_tensors='pt').to(DEVICE)
        with torch.inference_mode():
            t=clip.get_text_features(**inp)
            if not isinstance(t,torch.Tensor): t=t.pooler_output
            t=F.normalize(t.float(),dim=-1)
        parts.append(t)
    return torch.cat(parts,dim=0)

text_single=clip_text_vectors(single_prompts)
text_pairs=clip_text_vectors(pair_prompts)
TOP_EXAMPLES=32
CROP_SIZE=96
sem_sample=train.sample(n=min(900,len(train)),random_state=SEED+1).reset_index(drop=True)
reservoir={k:[] for k in range(N_CONCEPTS)}
for chunk,z in infer_features(sem_sample,'Collect representative TRAIN concept crops'):
    ids=concept_assign(z)
    b,c,h,w=z.shape
    a=((z.permute(0,2,3,1).reshape(-1,c)-mu_t)/sd_t).clamp(-8,8)
    dist=(a*a).sum(1,keepdim=True)+(centers*centers).sum(1)[None,:]-2*a@centers.T
    dist=dist.reshape(b,h,w,N_CONCEPTS)
    for j,(_,r) in enumerate(chunk.iterrows()):
        assigned=ids[j]
        for k in range(N_CONCEPTS):
            loc=(assigned==k).nonzero(as_tuple=False)
            if not len(loc): continue
            scores=dist[j,:,:,k][assigned==k]
            take=min(3,len(scores))
            near=torch.topk(scores,take,largest=False)
            for index,d in zip(near.indices.tolist(),near.values.tolist()):
                yy,xx=loc[index].tolist()
                reservoir[k].append((float(d),str(r.path),int(xx),int(yy),w,h))
    for k in range(N_CONCEPTS):
        reservoir[k]=sorted(reservoir[k],key=lambda x:x[0])[:TOP_EXAMPLES*3]

def crop_cell(path,xx,yy,w,h):
    with Image.open(path) as im:
        im=im.convert('RGB').resize((224,224))
        cx=int((xx+.5)*224/w); cy=int((yy+.5)*224/h)
        r=CROP_SIZE//2
        left=max(0,min(224-CROP_SIZE,cx-r))
        top=max(0,min(224-CROP_SIZE,cy-r))
        return im.crop((left,top,left+CROP_SIZE,top+CROP_SIZE))

chosen={}
for k in range(N_CONCEPTS):
    used=set(); chosen[k]=[]
    for entry in sorted(reservoir[k]):
        if entry[1] in used: continue
        used.add(entry[1]); chosen[k].append(entry)
        if len(chosen[k])>=TOP_EXAMPLES: break
    if not chosen[k]: raise RuntimeError(f'Concept {k} has no representative crops')

fig,axes=plt.subplots(N_CONCEPTS,8,figsize=(16,N_CONCEPTS*2.1),squeeze=False)
for k in range(N_CONCEPTS):
    for j,ax in enumerate(axes[k]):
        ax.axis('off')
        if j<len(chosen[k]):
            _,p,xx,yy,w,h=chosen[k][j]
            ax.imshow(crop_cell(p,xx,yy,w,h))
        if j==0: ax.set_ylabel(f'C{k}',rotation=0,labelpad=20,fontsize=11)
plt.tight_layout()
fig.savefig(SAVE/'concept_representative_crops.png',dpi=150)
plt.close(fig)

concept_similarity=np.zeros((N_CONCEPTS,N_LABELS),dtype=np.float32)
concept_coherence=np.zeros(N_CONCEPTS,dtype=np.float32)
for k in tqdm(range(N_CONCEPTS),desc='CLIP concept labeling'):
    crops=[crop_cell(p,xx,yy,w,h) for _,p,xx,yy,w,h in chosen[k]]
    sims=[]
    for q in range(0,len(crops),32):
        inp=clip_processor(images=crops[q:q+32],return_tensors='pt').to(DEVICE)
        with torch.inference_mode():
            vec=clip.get_image_features(**inp)
            if not isinstance(vec,torch.Tensor): vec=vec.pooler_output
            vec=F.normalize(vec.float(),dim=-1)
            sims.append((vec@text_single.T).cpu().numpy())
    sims=np.concatenate(sims)
    concept_similarity[k]=sims.mean(axis=0)
    concept_coherence[k]=float((sims.argmax(1)==sims.mean(0).argmax()).mean())

temperature=0.05
shift=(concept_similarity-concept_similarity.max(axis=1,keepdims=True))/temperature
weights=np.exp(shift); weights/=weights.sum(axis=1,keepdims=True)
concept_relevance=weights@label_relevance
concept_labels=[label_names[i] for i in concept_similarity.argmax(axis=1)]
semantic=pd.DataFrame({
    'concept':np.arange(N_CONCEPTS),'clip_label':concept_labels,
    'clip_similarity':concept_similarity.max(axis=1),
    'label_agreement':concept_coherence,
    'bird_relevance_proxy':concept_relevance,
    'background_relevance_proxy':1-concept_relevance,
    'representative_crops':[len(chosen[k]) for k in range(N_CONCEPTS)]
})
semantic.to_csv(SAVE/'concept_semantics_clip.csv',index=False)
print('\nCLIP concept labels (inspect gallery before treating these as semantic concepts):')
print(semantic.to_string(index=False,float_format=lambda a:f'{a:.3f}'))
with open(SAVE/'concept_label_vocabulary.json','w') as f:
    json.dump({'labels':LABELS,'clip_model':CLIP_ID,
               'note':'Curated vocabulary; CLIP labels and soft weights are heuristic, not calibrated probabilities.'},f,indent=2)

pair_index={pair:i for i,pair in enumerate(pair_label_ids)}
best_label_ids=concept_similarity.argmax(axis=1).astype(int)
def clip_global_features(paths):
    vectors=[]
    for q in range(0,len(paths),BATCH_SIZE):
        images=[]
        for p in paths[q:q+BATCH_SIZE]:
            with Image.open(p) as im: images.append(im.convert('RGB').resize((224,224)))
        inp=clip_processor(images=images,return_tensors='pt').to(DEVICE)
        with torch.inference_mode():
            vec=clip.get_image_features(**inp)
            if not isinstance(vec,torch.Tensor): vec=vec.pooler_output
            vec=F.normalize(vec.float(),dim=-1)
        vectors.append(vec)
    return torch.cat(vectors,dim=0)
def process_split(rows,desc):
    records=[]
    for chunk,z in infer_features(rows,desc):
        ids=concept_assign(z)
        global_clip=clip_global_features(chunk.path.tolist())
        clip_single_scores=global_clip@text_single.T
        clip_pair_scores=global_clip@text_pairs.T
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
                bg_support=sum((1-concept_relevance[k])*max(B[k],0) for k in selected)
                bird_opposition=sum(concept_relevance[k]*max(-B[k],0) for k in selected)
                bird_support=sum(concept_relevance[k]*max(B[k],0) for k in selected)
                cross_interaction=sum(abs(v)*abs(concept_relevance[a]-concept_relevance[b]) for (a,b),v in pairs.items())

                pair_anomaly=0.0
                pair_discrepancy=0.0
                for (a,b),interaction in pairs.items():
                    la,lb=int(best_label_ids[a]),int(best_label_ids[b])
                    if la==lb: continue
                    idx=pair_index[tuple(sorted((la,lb)))]
                    joint=float(clip_pair_scores[j,idx])
                    singles=.5*float(clip_single_scores[j,la]+clip_single_scores[j,lb])
                    anomaly=max(0.,singles-joint)
                    pair_anomaly+=anomaly
                    pair_discrepancy+=abs(interaction)*anomaly
                interaction_abs=sum(abs(v) for v in pairs.values())
                rec={'img_filename':r.img_filename,'y':int(r.y),'place':int(r.place),
                     'pred':predicted,'error':int(predicted!=int(r.y)),
                     'confidence':confidence,'margin_abs':abs(base),
                     'bg_support':bg_support,'bird_opposition':bird_opposition,
                     'bird_support':bird_support,'semantic_discrepancy':bg_support+bird_opposition,
                     'cross_interaction':cross_interaction,'interaction_abs':interaction_abs,
                     'vlm_pair_anomaly':pair_anomaly,'vlm_pair_discrepancy':pair_discrepancy,
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
    'vlm_pair_only':['vlm_pair_anomaly'],
    'vlm_semantic_only':['semantic_discrepancy','vlm_pair_anomaly'],
    'semantic_components':['bg_support','bird_opposition','bird_support'],
    'semantic_plus_confidence':['confidence','semantic_discrepancy'],
    'cnn_plus_semantic':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['semantic_discrepancy'],
    'cnn_plus_vlm_pair':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['vlm_pair_anomaly'],
    'full_without_interaction':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['bg_support','bird_opposition','bird_support'],
    'full_with_interaction':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['bg_support','bird_opposition','bird_support','interaction_abs','cross_interaction'],
    'full_with_vlm_relation':['confidence']+[f'B_{k}' for k in range(N_CONCEPTS)]+['bg_support','bird_opposition','bird_support','interaction_abs','cross_interaction','vlm_pair_anomaly','vlm_pair_discrepancy'],
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
                      ('semantic_plus_confidence','confidence'),
                      ('full_with_vlm_relation','full_with_interaction'),
                      ('cnn_plus_vlm_pair','cnn_contributions')]:
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
               'semantic_model':CLIP_ID+'; curated text prompts and image/text compatibility, not calibrated conditional probabilities',
               'intervention':'zero layer3 spatial activations for each concept'},f,indent=2)
print('\nSaved outputs:',SAVE)
print('IMPORTANT: CLIP provides independent image/text compatibility, NOT a calibrated conditional semantic graph. Inspect the concept gallery.')