import os, json, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from transformers import CLIPModel, CLIPProcessor
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

SEED=20261014
torch.manual_seed(SEED); random.seed(SEED);np.random.seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
ROOT=Path('/kaggle/working/waterbirds_cig_dual_comparison')
OUT=ROOT/'semantic_vlm_validation_study'
OUT.mkdir(parents=True,exist_ok=True)
MODELS=('old_spurious','new_groupdro')
N_FOLDS=5
IMAGE_BATCH=12
TEXT_BATCH=32
CLIP_NAME='openai/clip-vit-base-patch32'
C=0.1
CACHE=OUT/'semantic_features_validation.csv'
print('Device:',DEVICE,'| output:',OUT)

frames={}
for m in MODELS:
    p=ROOT/m/'validation_features.csv'
    if not p.is_file():
        raise FileNotFoundError(f'Missing {p}. Run prior dual CNN CIG pipeline first.')
    d=pd.read_csv(p)
    for c in ['path','y','place','predicted_class','error','confidence']:
        if c not in d: raise RuntimeError(f'{m}: missing {c}')
    if d.path.duplicated().any(): raise RuntimeError(f'{m}: duplicate paths')
    if not (d.error.to_numpy(dtype=int)==
            (d.y.to_numpy(dtype=int)!=d.predicted_class.to_numpy(dtype=int))).all():
        raise RuntimeError(f'{m}: inconsistent error labels')
    frames[m]=d.reset_index(drop=True)
paths=frames[MODELS[0]].path.astype(str).tolist()
if set(paths)!=set(frames[MODELS[1]].path.astype(str)):
    raise RuntimeError('Two model validation sets differ.')
for m in MODELS:
    frames[m]=frames[m].set_index('path').loc[paths].reset_index()
if any(not Path(p).is_file() for p in paths):
    raise FileNotFoundError('Validation images inaccessible; attach Waterbirds dataset.')
print('Validation images:',len(paths))
TEXTS={
    'bird_land':[
        'a photo of a land bird',
        'a photo of a terrestrial bird',
        'a close-up photo of a bird adapted to living on land'],
    'bird_water':[
        'a photo of a water bird',
        'a photo of an aquatic bird',
        'a close-up photo of a bird adapted to living near water'],
    'bg_land':[
        'a photograph of grass, trees, and dry land',
        'a natural terrestrial background with vegetation'],
    'bg_water':[
        'a photograph of a lake or river water surface',
        'a natural aquatic background with water'],
    'joint_0_0':['a land bird in a grassy or forest environment'],
    'joint_0_1':['a land bird near a lake or river'],
    'joint_1_0':['a water bird in a grassy or forest environment'],
    'joint_1_1':['a water bird near a lake or river'],
}
all_texts=sum(TEXTS.values(),[])
slices={}; pos=0
for k,v in TEXTS.items():
    slices[k]=slice(pos,pos+len(v));pos+=len(v)

def stable_softmax(a,axis=-1):
    a=np.asarray(a,dtype=float)
    z=np.exp(a-a.max(axis=axis,keepdims=True))
    return z/z.sum(axis=axis,keepdims=True)

def check_binary_auc(y,s):
    y=np.asarray(y).astype(int);s=np.asarray(s,dtype=float)
    if len(set(y))<2:return (np.nan,np.nan)
    return float(roc_auc_score(y,s)),float(average_precision_score(y,s))

if CACHE.exists():
    sem=pd.read_csv(CACHE)
    if not set(paths).issubset(set(sem.path)) or len(sem)!=len(paths):
        raise RuntimeError('Cached semantic features do not match validation paths; delete cache to recompute.')
    sem=sem.set_index('path').loc[paths].reset_index()
    print('Loaded existing VLM validation cache.')
else:
    try:
        clip=CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
        proc=CLIPProcessor.from_pretrained(CLIP_NAME)
        seg_weights=DeepLabV3_ResNet50_Weights.DEFAULT
        segmenter=deeplabv3_resnet50(weights=seg_weights).to(DEVICE).eval()
    except Exception as ex:
        raise RuntimeError('Cannot load frozen CLIP / DeepLabV3 pretrained weights. Enable Kaggle Internet or attach models.') from ex
    for p in clip.parameters():p.requires_grad_(False)
    for p in segmenter.parameters():p.requires_grad_(False)
    pre=seg_weights.transforms()
    text_input=proc(text=all_texts,return_tensors='pt',padding=True,truncation=True).to(DEVICE)
    with torch.inference_mode():
        tx=clip.get_text_features(**text_input)
        if not isinstance(tx,torch.Tensor):tx=tx.pooler_output
        tx=F.normalize(tx.float(),dim=-1)

    def make_views(img,mask):
        base=np.asarray(img,dtype=np.uint8)
        neutral=np.full_like(base,127,dtype=np.uint8)
        only_bird=np.where(mask[...,None],base,neutral)
        only_bg=np.where(mask[...,None],neutral,base)
        return [Image.fromarray(base), Image.fromarray(only_bird),
                Image.fromarray(only_bg)]

    def clip_sim(images):
        entries=[]
        for start in range(0,len(images),IMAGE_BATCH):
            chunk=images[start:start+IMAGE_BATCH]
            args=proc(images=chunk,return_tensors='pt').to(DEVICE)
            with torch.inference_mode():
                z=clip.get_image_features(**args)
                if not isinstance(z,torch.Tensor):z=z.pooler_output
                z=F.normalize(z.float(),dim=-1)
                entries.append((z@tx.T).cpu().numpy())
        return np.concatenate(entries)

    records=[]
    for st in tqdm(range(0,len(paths),IMAGE_BATCH),desc='Semantic VLM features on VALIDATION only'):
        these=paths[st:st+IMAGE_BATCH]
        pil=[]
        for p in these:
            with Image.open(p) as im:pil.append(im.convert('RGB').resize((224,224)))
        seg_input=torch.stack([pre(im) for im in pil]).to(DEVICE)
        with torch.inference_mode():
            seg_logits=segmenter(seg_input)['out']
            seg_logits=F.interpolate(seg_logits,size=(224,224),mode='bilinear',align_corners=False)
            bird_masks=(seg_logits.argmax(1)==3).cpu().numpy()
        views=[]
        for im,mask in zip(pil,bird_masks):
            views.extend(make_views(im,mask))
        sim=clip_sim(views).reshape(len(these),3,len(all_texts))
        for k,path in enumerate(these):
            original,bird,background=sim[k]
            def mean_score(arr,key):return float(arr[slices[key]].mean())
            b0=mean_score(bird,'bird_land')
            b1=mean_score(bird,'bird_water')
            bg0=mean_score(background,'bg_land')
            bg1=mean_score(background,'bg_water')
            all_b0=mean_score(original,'bird_land')
            all_b1=mean_score(original,'bird_water')
            joint=np.array([[mean_score(original,'joint_0_0'),mean_score(original,'joint_0_1')],
                            [mean_score(original,'joint_1_0'),mean_score(original,'joint_1_1')]],dtype=float)

            sem_pair=joint-np.array([all_b0,all_b1])[:,None]-np.array([bg0,bg1])[None,:]
            records.append(dict(path=path,mask_bird_fraction=float(bird_masks[k].mean()),
                                vlm_bird_class_margin=b1-b0,
                                vlm_full_class_margin=all_b1-all_b0,
                                vlm_background_class_margin=bg1-bg0,
                                vlm_bg_land_score=bg0,vlm_bg_water_score=bg1,
                                vlm_bird_land_score=b0,vlm_bird_water_score=b1,
                                vlm_subject_bg_coupling=sem_pair[1,1]-sem_pair[1,0]-sem_pair[0,1]+sem_pair[0,0],
                                vlm_context_max=float(sem_pair.max()),
                                vlm_context_range=float(sem_pair.max()-sem_pair.min()),
                                vlm_pair_00=float(sem_pair[0,0]),
                                vlm_pair_01=float(sem_pair[0,1]),
                                vlm_pair_10=float(sem_pair[1,0]),
                                vlm_pair_11=float(sem_pair[1,1])))
    sem=pd.DataFrame(records)
    sem.to_csv(CACHE,index=False)
    del clip,segmenter
    if DEVICE=='cuda':torch.cuda.empty_cache()
    print('Saved VLM features:',CACHE)

semantic_base=['vlm_bird_class_margin','vlm_full_class_margin',
               'vlm_background_class_margin','vlm_bg_land_score',
               'vlm_bg_water_score','vlm_bird_land_score','vlm_bird_water_score',
               'mask_bird_fraction']
semantic_pair=['vlm_subject_bg_coupling','vlm_context_max','vlm_context_range',
               'vlm_pair_00','vlm_pair_01','vlm_pair_10','vlm_pair_11']

def pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C,max_iter=3000,random_state=SEED))
def cv_oof(d,features,conditional=False):
    y=d.error.to_numpy(dtype=int)
    cls=d.predicted_class.to_numpy(dtype=int)
    ans=np.full(len(d),np.nan)
    for which in ([None] if not conditional else [0,1]):
        ids=np.arange(len(d)) if which is None else np.flatnonzero(cls==which)
        if len(np.unique(y[ids]))<2:raise ValueError('One class stratum contains no errors/correct examples')
        n=min(N_FOLDS,np.bincount(y[ids]).min())
        if n<2:raise ValueError('Not enough data for OOF in one class stratum')
        cv=StratifiedKFold(n_splits=int(n),shuffle=True,random_state=SEED)
        X=d.iloc[ids][features].to_numpy(dtype=float)
        if not np.isfinite(X).all():raise ValueError('NaN/inf in '+str(features))
        for itr,ite in cv.split(X,y[ids]):
            p=pipe();p.fit(X[itr],y[ids][itr])
            ans[ids[ite]]=p.predict_proba(X[ite])[:,1]
    if not np.isfinite(ans).all():raise AssertionError('Incomplete OOF')
    return ans

def bootstrap_delta(y,s1,s0,its=400):
    yy=np.asarray(y,dtype=int);a=np.asarray(s1);b=np.asarray(s0)
    rng=np.random.default_rng(SEED)
    one=np.flatnonzero(yy==1);zero=np.flatnonzero(yy==0)
    vals=[]
    for _ in range(its):
        idx=np.r_[rng.choice(one,len(one),replace=True),rng.choice(zero,len(zero),replace=True)]
        vals.append((roc_auc_score(yy[idx],a[idx])-roc_auc_score(yy[idx],b[idx]),
                     average_precision_score(yy[idx],a[idx])-average_precision_score(yy[idx],b[idx])))
    v=np.asarray(vals)
    return {'auroc_diff_lo':np.quantile(v[:,0],.025),'auroc_diff_hi':np.quantile(v[:,0],.975),
            'auprc_diff_lo':np.quantile(v[:,1],.025),'auprc_diff_hi':np.quantile(v[:,1],.975)}

all_results=[];all_pair=[];all_strata=[]
for m in MODELS:
    d=frames[m].merge(sem,on='path',how='left',validate='one_to_one')
    if len(d)!=len(paths) or d[semantic_base+semantic_pair].isna().any().any():
        raise RuntimeError('VLM feature merge failed')
    cls_sign=2*d.predicted_class.to_numpy(dtype=float)-1
    d['vlm_bird_prediction_disagreement']=-cls_sign*d.vlm_bird_class_margin
    d['vlm_full_prediction_disagreement']=-cls_sign*d.vlm_full_class_margin
    d['vlm_bg_prediction_alignment']=cls_sign*d.vlm_background_class_margin
    d['vlm_bird_bg_mismatch']=abs(d.vlm_bird_class_margin-d.vlm_background_class_margin)
    sem_features=semantic_base+[
        'vlm_bird_prediction_disagreement','vlm_full_prediction_disagreement',
        'vlm_bg_prediction_alignment','vlm_bird_bg_mismatch']
    sem_plus_pair=sem_features+semantic_pair
    contributes=sorted([c for c in d if c.startswith('B_') and c[2:].isdigit()],
                       key=lambda c:int(c[2:]))
    activations=sorted([c for c in d if c.startswith('activation_') and
                        c.rsplit('_',1)[-1].isdigit()],
                       key=lambda c:int(c.rsplit('_',1)[-1]))
    cnn_basic=['confidence']+contributes+activations
    cnn_inter=['interaction_abs','interaction_signed','cross_interaction','signed_cross_interaction']
    cnn_sem_old=['semantic_discrepancy','bg_support','bird_opposition','bird_support']
    specs={
        'confidence_conditional':['confidence'],
        'cnn_contributions_conditional':cnn_basic,
        'cnn_interactions_conditional':cnn_basic+cnn_inter,
        'VLM_single_semantics_only':sem_features,
        'VLM_pair_semantics_only':sem_plus_pair,
        'cnn_plus_VLM_single':cnn_basic+sem_features,
        'cnn_plus_VLM_pairs':cnn_basic+sem_plus_pair,
        'cnn_interactions_plus_VLM_single':cnn_basic+cnn_inter+sem_features,
        'cnn_interactions_plus_VLM_pairs':cnn_basic+cnn_inter+sem_plus_pair,
        'previous_graph_no_interaction':cnn_basic+cnn_sem_old,
        'previous_graph_with_interaction':cnn_basic+cnn_sem_old+cnn_inter,
    }
    df_scores=pd.DataFrame({'path':d.path,'predicted_class':d.predicted_class,
                            'y_for_evaluation_only':d.y,'place_for_evaluation_only':d.place,
                            'error':d.error})
    for name,features in tqdm(specs.items(),desc=f'{m} 5-fold validation OOF'):
        missing=set(features)-set(d)
        if missing:raise RuntimeError(f'{name} missing {missing}')
        conditional=name in ('confidence_conditional','cnn_contributions_conditional',
                             'cnn_interactions_conditional','previous_graph_no_interaction',
                             'previous_graph_with_interaction')
        if name.startswith('cnn_plus_') or name.startswith('cnn_interactions_plus_'):
            conditional=True
        df_scores[name]=cv_oof(d,features,conditional=conditional)
    df_scores.to_csv(OUT/f'{m}_validation_oof_scores.csv',index=False)
    for name in specs:
        a,p=check_binary_auc(d.error,df_scores[name])
        all_results.append(dict(model=m,detector=name,n=len(d),n_errors=int(d.error.sum()),
                                AUROC=a,AUPRC=p,split='validation_OOF'))
        for cls in (0,1):
            sel=d.predicted_class.to_numpy()==cls
            a,p=check_binary_auc(d.error.to_numpy()[sel],df_scores.loc[sel,name])
            all_strata.append(dict(model=m,predicted_class=cls,detector=name,
                                   n=int(sel.sum()),errors=int(d.error.to_numpy()[sel].sum()),
                                   AUROC=a,AUPRC=p))
    pairs=[
        ('cnn_plus_VLM_single','cnn_contributions_conditional'),
        ('cnn_plus_VLM_pairs','cnn_plus_VLM_single'),
        ('cnn_interactions_plus_VLM_pairs','cnn_interactions_conditional'),
        ('cnn_interactions_plus_VLM_pairs','cnn_interactions_plus_VLM_single'),
        ('cnn_interactions_conditional','cnn_contributions_conditional'),
        ('previous_graph_with_interaction','previous_graph_no_interaction'),
    ]
    for x0,y0 in pairs:
        a,b=check_binary_auc(d.error,df_scores[x0]),check_binary_auc(d.error,df_scores[y0])
        cis=bootstrap_delta(d.error,df_scores[x0],df_scores[y0])
        all_pair.append(dict(model=m,comparison=f'{x0} MINUS {y0}',
                             delta_AUROC=a[0]-b[0],delta_AUPRC=a[1]-b[1],**cis))

result=pd.DataFrame(all_results)
pairs=pd.DataFrame(all_pair)
strata=pd.DataFrame(all_strata)
result.to_csv(OUT/'validation_oof_results.csv',index=False)
pairs.to_csv(OUT/'validation_oof_paired_bootstrap.csv',index=False)
strata.to_csv(OUT/'validation_oof_predicted_class.csv',index=False)
print('\n=== FIVE-FOLD OUT-OF-FOLD VALIDATION (not TEST) ===')
for model in MODELS:
    print('\n',model)
    print(result[result.model==model][['detector','AUROC','AUPRC']].sort_values('AUROC',ascending=False).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\n=== PAIRED BOOTSTRAP (fixed OOF scores; intervals do not account for retraining) ===')
print(pairs.to_string(index=False,float_format=lambda x:f'{x:+.4f}'))
print('\n=== BY PREDICTED CLASS ===')
print(strata[strata.detector.isin(['confidence_conditional','cnn_contributions_conditional',
                                  'cnn_interactions_conditional','cnn_interactions_plus_VLM_pairs'])]
      .to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if 'mask_bird_fraction' in sem:
    print('\nDeepLab bird-mask fractions (validation images):',
          sem.mask_bird_fraction.describe(percentiles=[.1,.25,.5,.75,.9]).to_dict())
with open(OUT/'PROTOCOL.json','w') as f:
    json.dump(dict(clip_model=CLIP_NAME,cnn_models=list(MODELS),
                   evaluation='VALIDATION ONLY, out-of-fold 5-fold stratified by error within predicted class',
                   test='NOT ACCESSED by this script',
                   semantic_nodes='KNOWN foreground bird and background (via independent pretrained DeepLab); not 16 automatically discovered concept nodes',
                   semantic_edges='CLIP image-text compatibility using fixed natural-language prompts, NOT calibrated conditional probabilities',
                   risks=['CLIP itself may solve binary bird classification; gains need VLM-only and matched baselines',
                          'Segmentation errors, masking artifacts, curated English text prompts',
                          'Same validation set also used earlier for checkpoint selection/detector development; not a fresh independent holdout',
                          'Repeated exploratory experiments introduce selection bias',
                          'Pair prompt residuals are NOT causal interactions',
                          'Paired bootstrap on OOF scores does not include model fitting uncertainty',
                          'This study does not fully establish a learned contextual semantic concept graph']),
              f,indent=2)
print('\nSaved:',OUT)
print('Files: validation_oof_results.csv, validation_oof_paired_bootstrap.csv, validation_oof_predicted_class.csv')
print('No Waterbirds TEST predictions were read or evaluated.')