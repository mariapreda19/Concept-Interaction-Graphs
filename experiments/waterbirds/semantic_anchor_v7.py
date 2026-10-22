import os, glob, json, random, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
from skimage.segmentation import slic
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

SEED=42; IMAGE_SIZE=224; N_VAL=220; BIRD_THRESHOLD=0.50
MAX_ACE_CONCEPTS=20; TOP_CONCEPTS_PER_IMAGE=8
MASK_COLOR=np.array([123,116,103],dtype=np.uint8)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
OUT=Path('/kaggle/working/waterbirds_v7_semantic_anchors'); OUT.mkdir(parents=True,exist_ok=True)
V4=Path('/kaggle/working/waterbirds_ace_semantic_graph_v4')
if not (V4/'concept_summary.csv').exists():
    candidates=[p.parent for p in Path('/kaggle/input').rglob('concept_summary.csv') if (p.parent/'train_region_vectors.npy').exists()]
    if not candidates: raise FileNotFoundError('Need V4 output: concept_summary.csv, train_region_vectors.npy, train_region_assignments.csv, and validation files. Run V4 or attach saved V4 output.')
    V4=candidates[0]
print('DEVICE:',DEVICE,'V4:',V4,'OUTPUT:',OUT)

roots=[Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),Path('/kaggle/input/waterbird/waterbird'),Path('/kaggle/input/waterbird')]
ROOT=next((p for p in roots if (p/'metadata.csv').exists()),None)
if ROOT is None:
    hits=[Path(s).parent for s in glob.glob('/kaggle/input/**/metadata.csv',recursive=True) if 'waterbird' in s.lower()]
    if not hits: raise FileNotFoundError('Waterbirds metadata.csv not found')
    ROOT=hits[0]
meta=pd.read_csv(ROOT/'metadata.csv')
assert {'split','y','place','img_filename'}.issubset(meta.columns)
val=meta.loc[meta['split']==1].sample(frac=1,random_state=SEED).head(N_VAL).reset_index(drop=True)
if len(val)<40: raise RuntimeError('Not enough validation images')

def resolve_image(name):
    p=Path(str(name))
    if p.is_file(): return p
    for q in [ROOT/p,ROOT/'dataset'/p, ROOT/p.name,ROOT/'dataset'/p.name]:
        if q.is_file():return q
    raise FileNotFoundError('Cannot resolve image: '+str(name))

def load_rgb(name):
    with Image.open(resolve_image(name)) as im:return np.asarray(im.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE)))

ckpts=[Path('/kaggle/working/waterbirds_human_concepts_v2/resnet18_waterbirds_best.pt'),
       Path('/kaggle/working/waterbirds_human_concepts/resnet18_waterbirds_best.pt')]
ckpts.extend(Path(p) for p in glob.glob('/kaggle/input/**/resnet18_waterbirds_best.pt',recursive=True))
ckpt=next((p for p in ckpts if p.is_file()),None)
if ckpt is None:raise FileNotFoundError('Attach exact ResNet18 checkpoint resnet18_waterbirds_best.pt from V4')
cnn=models.resnet18(weights=None); cnn.fc=torch.nn.Linear(cnn.fc.in_features,2)
state=torch.load(ckpt,map_location='cpu',weights_only=False)
if isinstance(state,dict):
    for k in ['model_state_dict','state_dict','model']:
        if k in state and isinstance(state[k],dict): state=state[k]; break
if not isinstance(state,dict):raise ValueError('Invalid CNN checkpoint')
state={k.removeprefix('module.'):v for k,v in state.items()}
cnn.load_state_dict(state,strict=True); cnn=cnn.to(DEVICE).eval()
for p in cnn.parameters():p.requires_grad_(False)
print('Fixed classifier:',ckpt)

norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
def cnn_batch(arrays):
    t=torch.stack([norm(torch.from_numpy(np.array(a,copy=True)).permute(2,0,1).float()/255.0) for a in arrays]).to(DEVICE)
    with torch.inference_mode():z=cnn(t).float().cpu().numpy()
    return z[:,1]-z[:,0]

try:
    seg_weights=DeepLabV3_ResNet50_Weights.DEFAULT
    segmenter=deeplabv3_resnet50(weights=seg_weights).to(DEVICE).eval()
except Exception as exc:
    raise RuntimeError('Cannot load torchvision pretrained DeepLabV3 weights. Enable Kaggle Internet temporarily or attach the pretrained weights. '+str(exc))
for p in segmenter.parameters():p.requires_grad_(False)
seg_norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
def predict_bird(arr):
    x=seg_norm(torch.from_numpy(np.array(arr,copy=True)).permute(2,0,1).float()/255.0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        z=segmenter(x)['out']
        prob=z.softmax(1)[0,3].cpu().numpy()
        pred=z.argmax(1)[0].cpu().numpy()
    mask=(prob>=BIRD_THRESHOLD)&(pred==3)
    return mask,prob

concepts=pd.read_csv(V4/'concept_summary.csv').head(MAX_ACE_CONCEPTS).copy()
regions=pd.read_csv(V4/'train_region_assignments.csv')
vecfile=V4/'train_region_vectors.npy'
if not vecfile.exists():raise FileNotFoundError('Missing V4 train_region_vectors.npy')
X=np.load(vecfile,mmap_mode='r')
if len(X)!=len(regions):raise ValueError('V4 vector-region alignment failed')
protos=pd.read_csv(V4/'concept_prototypes.csv')
centers={}
for cid in concepts.concept_id.astype(str):
    ids=protos.loc[protos.concept_id.astype(str)==cid,'region_id'].astype(int).to_numpy()
    if len(ids)==0:continue
    v=np.asarray(X[ids],dtype=np.float32).mean(0);v=v/max(np.linalg.norm(v),1.e-8)
    centers[cid]=v
print('ACE prototypes available:',len(centers),'from',len(concepts),'concept candidates')

activ=[]
def hookfn(module,inputs,output):activ.append(output.detach().flatten(1))
hook=cnn.avgpool.register_forward_hook(hookfn)
def descriptor_batch(patches):
    if not patches:return np.empty((0,512),np.float32)
    out=[]
    for start in range(0,len(patches),48):
        t=torch.stack([norm(torch.from_numpy(np.array(a,copy=True)).permute(2,0,1).float()/255.) for a in patches[start:start+48]]).to(DEVICE)
        activ.clear()
        with torch.inference_mode():cnn(t)
        out.append(F.normalize(activ[-1],dim=1).float().cpu().numpy())
    return np.concatenate(out)

def intervention(arr,mask):
    z=arr.copy(); z[mask]=MASK_COLOR;return z

def sscale(y_true,score):
    if len(set(y_true))<2:return {'AUROC':None,'AUPRC':None}
    return {'AUROC':float(roc_auc_score(y_true,score)),'AUPRC':float(average_precision_score(y_true,score))}

alignment=[]; results=[]; inspected=0
for i,r in enumerate(tqdm(val.itertuples(index=False),total=len(val),desc='V7 grounded graph validation')):
    arr=load_rgb(r.img_filename)
    bird,prob=predict_bird(arr)
    area=float(bird.mean())
    reliable=(area>=.012 and area<=.75)
    if not reliable:
        results.append(dict(image_i=i,y=int(r.y),place=int(r.place),bird_detected=False,bird_area=area,
            error=np.nan,confidence_score=np.nan,node_effect_score=np.nan,interaction_score=np.nan,
            cnn_bird_background_interaction=np.nan,n_ace_aligned=0))
        continue
    background=~bird
    b0,b1,b2,b3=cnn_batch([arr,intervention(arr,bird),intervention(arr,background),intervention(arr,np.ones_like(bird,dtype=bool))])
    pred=int(b0>0);error=int(pred!=int(r.y))
    db=float(b1-b0);dg=float(b2-b0)
    interaction=float(b3-b1-b2+b0)
    direction=1 if pred==1 else -1
    node_support_bird=-direction*db
    node_support_bg=-direction*dg
    results.append(dict(image_i=i,y=int(r.y),place=int(r.place),bird_detected=True,bird_area=area,
        error=error,confidence_score=-abs(float(b0)),node_effect_score=abs(node_support_bird)+abs(node_support_bg),
        interaction_score=abs(interaction),cnn_bird_background_interaction=interaction,
        bird_support=float(node_support_bird),background_support=float(node_support_bg),n_ace_aligned=0))
    best={}
    for scale in [15,50,80]:
        segmentation=slic(arr,n_segments=scale,compactness=10,sigma=1,start_label=0,channel_axis=-1)
        parts=[]; masks=[]
        for sid in np.unique(segmentation):
            mask=segmentation==sid;frac=float(mask.mean())
            if frac<.003 or frac>.65:continue
            yy,xx=np.where(mask)
            y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1
            if x1-x0<5 or y1-y0<5:continue
            canvas=np.full_like(arr,117);canvas[mask]=arr[mask]
            patch=np.asarray(Image.fromarray(canvas[y0:y1,x0:x1]).resize((IMAGE_SIZE,IMAGE_SIZE)))
            parts.append(patch);masks.append(mask)
        if not parts:continue
        embs=descriptor_batch(parts)
        matching=[row.concept_id for row in concepts.itertuples(index=False) if int(row.scale)==scale and str(row.concept_id) in centers]
        if not matching:continue
        C=np.stack([centers[c] for c in matching]);sim=embs@C.T
        for j,mask in enumerate(masks):
            k=int(np.argmax(sim[j]));score=float(sim[j,k]);cid=matching[k]
            if score<.65:continue
            if cid not in best or score>best[cid][0]:best[cid]=(score,mask)
    results[-1]['n_ace_aligned']=len(best)
    for cid,(sim,mask) in best.items():
        bird_overlap=float(np.logical_and(mask,bird).sum()/max(1,mask.sum()))
        bird_iou=float(np.logical_and(mask,bird).sum()/max(1,np.logical_or(mask,bird).sum()))
        alignment.append(dict(image_i=i,concept_id=cid,similarity=sim,bird_region_fraction=bird_overlap,
            bird_iou=bird_iou,background_region_fraction=1-bird_overlap,bird_area=area,error=error))
    if inspected<12:
        inspected+=1
        fig,ax=plt.subplots(1,3,figsize=(11,3.5))
        ax[0].imshow(arr);ax[0].set_title('Original | y='+str(int(r.y))+' pred='+str(pred))
        ax[1].imshow(arr);ax[1].imshow(bird,cmap='spring',alpha=bird.astype(float)*.45);ax[1].set_title('Independent proposed bird mask')
        ax[2].imshow(arr);ax[2].imshow(prob,cmap='viridis',alpha=.50);ax[2].set_title('Bird probability | '+str(round(area,2)))
        for a in ax:a.axis('off')
        fig.tight_layout();fig.savefig(OUT/('grounding_%03d.png'%i),dpi=120);plt.close(fig)
hook.remove()
res=pd.DataFrame(results);aln=pd.DataFrame(alignment)
res.to_csv(OUT/'image_level_graph.csv',index=False)
aln.to_csv(OUT/'ace_semantic_alignment_individual.csv',index=False)
if not aln.empty:
    agg=aln.groupby('concept_id',as_index=False).agg(
        n_images=('image_i','nunique'),mean_similarity=('similarity','mean'),
        mean_bird_fraction=('bird_region_fraction','mean'),mean_bird_iou=('bird_iou','mean'))
    agg['anchor_candidate']=np.where(agg.mean_bird_fraction>=.8,'bird-associated',np.where(agg.mean_bird_fraction<=.2,'background-associated','mixed / uncertain'))
    agg['enough_observations']=agg.n_images>=10
else:
    agg=pd.DataFrame(columns=['concept_id','n_images','mean_similarity','mean_bird_fraction','mean_bird_iou','anchor_candidate','enough_observations'])
agg.to_csv(OUT/'ace_semantic_alignment_summary.csv',index=False)

ok=res[res.bird_detected].dropna(subset=['error']).copy()
report={'setup':{'model':str(ckpt),'semantics':'DeepLabV3 VOC bird mask vs complement background',
         'ACE_centroids':'means of V4 stored prototype representations, approximate V4 centers',
         'n_validation_selected':len(res),'n_masks_valid':len(ok),'n_masks_rejected':int((~res.bird_detected).sum()),
         'n_errors_in_valid_masks':int(ok.error.sum()),'test_used':False},
         'analysis':{'note':'Exploratory. Fixed non-trained scores: higher means higher suspected error. Image masks and double-masking can introduce out-of-distribution artifacts; do not claim causal or semantic correctness.',
         'alignment_concepts_evaluated':len(agg),
         'confidence_only':sscale(ok.error.astype(int),ok.confidence_score),
         'interaction_only':sscale(ok.error.astype(int),ok.interaction_score),
         'confidence_plus_interaction':sscale(ok.error.astype(int),ok.confidence_score+np.log1p(ok.interaction_score)),
         'confidence_plus_nodes':sscale(ok.error.astype(int),ok.confidence_score+np.log1p(ok.node_effect_score)),
         'confidence_plus_nodes_and_interaction':sscale(ok.error.astype(int),ok.confidence_score+np.log1p(ok.node_effect_score)+np.log1p(ok.interaction_score))}}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print('\nREPORT:\n'+json.dumps(report,indent=2))
print('\nACE ALIGNMENT SUMMARY (provisional, not verified):')
print(agg.sort_values('n_images',ascending=False).head(20).to_string(index=False))
print('\nDONE:',OUT)
print('Send report.json, ace_semantic_alignment_summary.csv, image_level_graph.csv, and grounding_*.png')