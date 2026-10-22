import os, json, glob, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.segmentation import slic
import torch
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
V4=Path('/kaggle/working/waterbirds_ace_semantic_graph_v4')
if not (V4/'concept_prototypes.csv').is_file():
    roots=[p.parent for p in Path('/kaggle/input').rglob('concept_prototypes.csv') if (p.parent/'train_region_assignments.csv').exists()]
    if not roots: raise FileNotFoundError('V4 files not found: save/run V4 first or attach its outputs.')
    V4=roots[0]
OUT=Path('/kaggle/working/waterbirds_v6_grounded_audit'); OUT.mkdir(parents=True,exist_ok=True)
print('Device:',DEVICE,'\nV4:',V4,'\nOutput:',OUT)
concepts=pd.read_csv(V4/'concept_summary.csv')
protos=pd.read_csv(V4/'concept_prototypes.csv')
regions=pd.read_csv(V4/'train_region_assignments.csv')
pairs=pd.read_csv(V4/'pair_interactions_individual.csv')
if pairs.empty:raise RuntimeError('No v4 pair measurements: cannot evaluate.')
meta_roots=[Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),Path('/kaggle/input/waterbird/waterbird'),Path('/kaggle/input/waterbird')]
meta_root=next((p for p in meta_roots if (p/'metadata.csv').exists()),None)
def fixpath(s):
    p=Path(str(s))
    if p.exists():return p
    if meta_root is not None:
        candidates=[meta_root/p.name,meta_root/'dataset'/p.name]
        s2=str(s).replace('\\','/')
        if '/waterbird/' in s2:
            tail=s2.split('/waterbird/')[-1]
            candidates.extend([meta_root/tail,meta_root/'dataset'/tail])
        for q in candidates:
            if q.exists():return q
    raise FileNotFoundError('Original training image unavailable: '+str(s)[:150]+'. Attach Waterbirds dataset and rerun.')

LABELS=['bird head','bird beak','bird wing','bird feathers','bird body','bird tail','whole bird',
        'water surface','water waves','sky','clouds','grass','green vegetation','tree leaves',
        'tree branches','reeds','sand','rocks','earth','tree trunk','background texture']
from transformers import CLIPModel, CLIPProcessor
model_id='openai/clip-vit-base-patch32'
try:
    proc=CLIPProcessor.from_pretrained(model_id)
    model=CLIPModel.from_pretrained(model_id).eval().to(DEVICE)
except Exception as e:raise RuntimeError('CLIP unavailable; enable internet or add cached model weights: '+repr(e))
for p in model.parameters():p.requires_grad_(False)
with torch.inference_mode():
    ti=proc(text=['a photograph of '+z for z in LABELS],padding=True,return_tensors='pt').to(DEVICE)
    te=model.get_text_features(**ti)
    if hasattr(te,'pooler_output'):te=te.pooler_output
    te=torch.nn.functional.normalize(te,dim=-1)

cache={}
def views(rec):
    p=fixpath(rec['path']);scale=int(rec['scale']);key=(str(p),scale)
    if key not in cache:
        rgb=np.asarray(Image.open(p).convert('RGB').resize((224,224)))
        seg=slic(rgb,n_segments=scale,compactness=10,sigma=1,start_label=0,channel_axis=-1)
        cache[key]=(rgb,seg)
    rgb,seg=cache[key];mask=seg==int(rec['segment'])
    if not mask.any():return None
    yy,xx=np.where(mask); y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1
    bg=np.full_like(rgb,117);bg[mask]=rgb[mask]
    crop=Image.fromarray(bg[y0:y1,x0:x1]).resize((224,224))
    context=rgb.copy()
    context[~mask]=(0.28*context[~mask]+0.72*np.array([110,110,110])).astype(np.uint8)
    context=Image.fromarray(context)
    return crop,context

per=[]; summary=[]
for cid in tqdm(concepts.concept_id.astype(str).tolist(),desc='Grounded region review'):
    refs=protos[protos.concept_id.astype(str)==cid].sort_values('rank').head(12)
    scorelist=[]; names=[]; reps=[]
    for pr in refs.itertuples(index=False):
        rec=regions.iloc[int(pr.region_id)]
        out=views(rec)
        if out is None:continue
        crop,context=out
        ins=proc(images=[crop,context],return_tensors='pt').to(DEVICE)
        with torch.inference_mode():
            z=model.get_image_features(**ins)
            if hasattr(z,'pooler_output'):z=z.pooler_output
            z=torch.nn.functional.normalize(z,dim=-1)
            sims=(z@te.T).float().cpu().numpy()
        score=0.75*sims[0]+0.25*sims[1]
        top=np.argsort(-score);winner=int(top[0]);gap=float(score[top[0]]-score[top[1]])
        names.append(LABELS[winner]);scorelist.append(score)
        per.append(dict(concept_id=cid,region_id=int(pr.region_id),candidate=LABELS[winner],similarity=float(score[winner]),margin=gap))
        reps.append((crop,context,LABELS[winner]))
    if not scorelist:continue
    mat=np.stack(scorelist)
    avg=mat.mean(0);order=np.argsort(-avg);best=int(order[0]);second=int(order[1]);label=LABELS[best]
    agreement=sum(x==label for x in names)/len(names)
    margin=float(avg[best]-avg[second]);sim=float(avg[best])
    review=(agreement>=0.65 and margin>=0.015 and sim>=0.22 and len(names)>=5)
    summary.append(dict(concept_id=cid,candidate_label=label,agreement=agreement,
      mean_similarity=sim,mean_margin=margin,n_prototypes=len(names),automatic_candidate_pass=review,
      human_verified=False,human_label='',human_coherent_0_to_2='',human_notes=''))
    fig,axes=plt.subplots(min(len(reps),6),2,figsize=(7,2.25*min(len(reps),6)),squeeze=False)
    for j,(crop,context,guess) in enumerate(reps[:6]):
        axes[j,0].imshow(crop);axes[j,0].set_title('Focused region',fontsize=9)
        axes[j,1].imshow(context);axes[j,1].set_title('Context | '+guess,fontsize=9)
        for a in axes[j]:a.axis('off')
    fig.suptitle(cid+' | candidate: '+label+' | agreement '+str(round(agreement,2)))
    fig.tight_layout();fig.savefig(OUT/(cid+'_context_review.png'),dpi=120);plt.close(fig)

review=pd.DataFrame(summary);details=pd.DataFrame(per)
review.to_csv(OUT/'concepts_to_review.csv',index=False)
details.to_csv(OUT/'prototype_grounding.csv',index=False)
print('\nCONCEPT REVIEW (candidate ≠ verified):')
print(review[['concept_id','candidate_label','agreement','mean_margin','automatic_candidate_pass']].to_string(index=False))
print('Candidate passes:',int(review.automatic_candidate_pass.sum()),'/',len(review))

rows=[]
for sid,sub in pairs.groupby('sample_id'):
    base=float(sub.base_margin.iloc[0]); y=int(sub.y.iloc[0]); pred=int(base>0)
    absw=sub.interaction.abs().to_numpy(dtype=float)
    rows.append(dict(sample_id=int(sid),error=int(pred!=y),confidence_abs_margin=abs(base),
        interaction_mean_abs=float(np.mean(absw)),interaction_max_abs=float(np.max(absw)),n_edges=len(sub)))
metrics=pd.DataFrame(rows)
metrics['low_confidence_score']=-metrics.confidence_abs_margin
metrics['interaction_augmented_score']=-np.log1p(metrics.confidence_abs_margin)+np.log1p(metrics.interaction_mean_abs)
metrics.to_csv(OUT/'validation_error_scores.csv',index=False)
result={'n_validation_images':len(metrics),'n_errors':int(metrics.error.sum()),'n_candidate_concepts':len(review),
 'n_automatic_candidate_pass':int(review.automatic_candidate_pass.sum()),'test_used':False}
if metrics.error.nunique()==2:
    for key in ['low_confidence_score','interaction_augmented_score']:
        result[key+'_AUROC']=float(roc_auc_score(metrics.error,metrics[key]))
        result[key+'_AUPRC']=float(average_precision_score(metrics.error,metrics[key]))
else:result['warning']='Only one outcome class in sample: AUROC undefined.'
(OUT/'validation_diagnostic.json').write_text(json.dumps(result,indent=2))
print('\nVALIDATION DIAGNOSTIC (exploratory, not unbiased generalization):\n',json.dumps(result,indent=2))
name=dict(zip(review.concept_id,review.candidate_label))
flag=dict(zip(review.concept_id,review.automatic_candidate_pass))
graph=pairs.copy()
graph['candidate_a']=graph.concept_a.map(name);graph['candidate_b']=graph.concept_b.map(name)
graph['both_candidate_pass']=graph.concept_a.map(flag).fillna(False)&graph.concept_b.map(flag).fillna(False)
graph['human_verified']=False
graph.to_csv(OUT/'cnn_graph_with_provisional_labels.csv',index=False)
print('\nDONE:',OUT,'\nSend: concepts_to_review.csv, validation_diagnostic.json and a few Cxxx_context_review.png files.')
print('Never treat automatic_candidate_pass as human verification. No semantic graph claim until concepts are checked.')