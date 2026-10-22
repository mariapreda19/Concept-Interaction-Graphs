import os, json, random, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import expit

SEED=20261014
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
ROOT=Path('/kaggle/working/waterbirds_cig_dual_comparison')
OUT=ROOT/'clip_control_contextual_anomaly'
OUT.mkdir(parents=True,exist_ok=True)
MODELS=('old_spurious','new_groupdro')
CLIP_NAME='openai/clip-vit-base-patch32'
BATCH=24
CV=5
C=0.1
EPS=1e-6
print('Device:',DEVICE,'Output:',OUT)

cache=ROOT/'semantic_vlm_validation_study'/'semantic_features_validation.csv'
if not cache.is_file():
    raise FileNotFoundError('Run waterbirds_semantic_vlm_validation_only.py first: '+str(cache))
semantic_val=pd.read_csv(cache)
if semantic_val.path.duplicated().any():raise RuntimeError('Duplicate VLM cached paths')
features={}
for model in MODELS:
    f=ROOT/model/'validation_features.csv'
    if not f.is_file():raise FileNotFoundError(str(f))
    d=pd.read_csv(f)
    assert {'path','confidence','predicted_class','y','place','error'}.issubset(d)
    assert (d.predicted_class.astype(int).to_numpy()!=d.y.astype(int).to_numpy()).astype(int).tolist()==d.error.astype(int).tolist()
    if d.path.duplicated().any():raise RuntimeError('Duplicate graph paths')
    d=d.merge(semantic_val,on='path',validate='one_to_one',how='inner')
    if len(d)!=len(pd.read_csv(f)):raise RuntimeError('Missing semantic image scores')
    features[model]=d
paths=features[MODELS[0]].path.astype(str).tolist()
if set(paths)!=set(features[MODELS[1]].path):raise RuntimeError('Models have different validation images')
for model in MODELS:features[model]=features[model].set_index('path').loc[paths].reset_index()

CONCEPTS=[
    ('bird','a close-up photograph of a bird'),
    ('wing','a close-up photograph of a bird wing'),
    ('feathers','a close-up photograph of bird feathers'),
    ('beak','a close-up photograph of a bird beak'),
    ('water','a photograph of water'),
    ('grass','a photograph of grass'),
    ('trees','a photograph of trees'),
    ('sky','a photograph of blue sky'),
    ('reeds','a photograph of reeds near a lake'),
    ('rocks','a photograph of rocks and stones'),
]
N=len(CONCEPTS)
PROMPTS=[v for _,v in CONCEPTS]+[
    'a photo of a land bird','a photo of a terrestrial bird',
    'a photo of a water bird','a photo of an aquatic bird'
]

roots=[Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),
       Path('/kaggle/input/waterbird/waterbird'),Path('/kaggle/input/waterbird')]
dataset=next((p for p in roots if (p/'metadata.csv').is_file()),None)
if dataset is None:
    candidates=glob.glob('/kaggle/input/**/metadata.csv',recursive=True)
    if not candidates:raise FileNotFoundError('Cannot locate Waterbirds metadata.csv')
    dataset=Path(candidates[0]).parent
meta=pd.read_csv(dataset/'metadata.csv')
meta['path']=meta.img_filename.map(lambda v:str(dataset/str(v)))
train=meta[meta.split==0].copy().reset_index(drop=True)
if len(train)<500:raise RuntimeError('Unexpected Waterbirds TRAIN size')
if set(train.path).intersection(paths):raise RuntimeError('TRAIN / VAL overlap')
print('Train:',len(train),'Validation:',len(paths),'Concepts:',N)

clip_cache=OUT/'clip_train_val_scores.npz'
if clip_cache.is_file():
    npz=np.load(clip_cache,allow_pickle=False)
    train_score=npz['train_score'];val_score=npz['val_score']
    train_paths=npz['train_paths'].astype(str);val_paths=npz['val_paths'].astype(str)
    if not np.array_equal(train_paths,train.path.to_numpy(dtype=str)) or not np.array_equal(val_paths,np.array(paths,dtype=str)):
        raise RuntimeError('Cached CLIP scores correspond to other paths. Delete cache to regenerate.')
    if train_score.shape!=(len(train),len(PROMPTS)) or val_score.shape!=(len(paths),len(PROMPTS)):
        raise RuntimeError('CLIP cache shape mismatch')
    print('Using cached CLIP train/validation scores')
else:
    try:
        clip=CLIPModel.from_pretrained(CLIP_NAME).eval().to(DEVICE)
        proc=CLIPProcessor.from_pretrained(CLIP_NAME)
    except Exception as ex:
        raise RuntimeError('Could not load pretrained CLIP; enable Kaggle Internet or cached weights.') from ex
    for p in clip.parameters():p.requires_grad_(False)
    ti=proc(text=PROMPTS,return_tensors='pt',padding=True,truncation=True).to(DEVICE)
    with torch.inference_mode():
        text_embed=clip.get_text_features(**ti)
        if not isinstance(text_embed,torch.Tensor):text_embed=text_embed.pooler_output
        text_embed=F.normalize(text_embed.float(),dim=-1)
    def embed_paths(pathlist,title):
        scores=[]
        for i in tqdm(range(0,len(pathlist),BATCH),desc=title):
            ims=[]
            for p in pathlist[i:i+BATCH]:
                with Image.open(p) as im:ims.append(im.convert('RGB').resize((224,224)))
            inp=proc(images=ims,return_tensors='pt').to(DEVICE)
            with torch.inference_mode():
                img_emb=clip.get_image_features(**inp)
                if not isinstance(img_emb,torch.Tensor):img_emb=img_emb.pooler_output
                img_emb=F.normalize(img_emb.float(),dim=-1)
                sims=(img_emb@text_embed.T).cpu().numpy()
            scores.append(sims)
        return np.concatenate(scores,axis=0)
    train_score=embed_paths(train.path.tolist(),'CLIP TRAIN concept scores')
    val_score=embed_paths(paths,'CLIP VALIDATION concept scores')
    np.savez_compressed(clip_cache,train_score=train_score,val_score=val_score,
                        train_paths=np.array(train.path.tolist(),dtype=str),val_paths=np.array(paths,dtype=str))
    del clip
    if DEVICE=='cuda':torch.cuda.empty_cache()

thresholds=np.quantile(train_score[:,:N],.70,axis=0)
train_bin=(train_score[:,:N]>=thresholds[None,:]).astype(int)
val_bin=(val_score[:,:N]>=thresholds[None,:]).astype(int)
train_cont=train_score[:,:N].astype(float)
val_cont=val_score[:,:N].astype(float)

train_cls=np.stack([train_score[:,N:N+2].mean(1),train_score[:,N+2:N+4].mean(1)],axis=1)
val_cls=np.stack([val_score[:,N:N+2].mean(1),val_score[:,N+2:N+4].mean(1)],axis=1)
train_clip_margin=train_cls[:,1]-train_cls[:,0]
val_clip_margin=val_cls[:,1]-val_cls[:,0]
clip_pred=(val_clip_margin>0).astype(int)
clip_acc=float(np.mean(clip_pred==features[MODELS[0]].y.astype(int).to_numpy()))
print('Frozen CLIP zero-shot validation classification accuracy:',round(clip_acc,4))

node_models=[];pair_models={}
for i in tqdm(range(N),desc='Train masked-concept conditional models'):
    others=[j for j in range(N) if j!=i]
    X=train_cont[:,others]
    y=train_bin[:,i]
    if len(np.unique(y))<2:raise RuntimeError('Degenerate concept indicator '+str(i))
    md=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=SEED))
    md.fit(X,y)
    node_models.append((others,md))

for i in tqdm(range(N),desc='Train concept-pair conditional models'):
    for j in range(N):
        if i==j:continue
        context=[k for k in range(N) if k not in (i,j)]
        X=np.column_stack([train_cont[:,context],train_bin[:,i]])
        y=train_bin[:,j]
        md=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=SEED))
        md.fit(X,y)
        first=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=SEED))
        first.fit(train_cont[:,context],train_bin[:,i])
        pair_models[i,j]=(context,first,md)

def concept_scores(cont,binary):
    n=len(cont)
    pnode=np.zeros((n,N),dtype=float)
    for i,(others,md) in enumerate(node_models):
        pnode[:,i]=md.predict_proba(cont[:,others])[:,1]
    A_node=-np.log(np.clip(np.where(binary==1,pnode,1-pnode),EPS,1))
    PA=np.zeros((n,N,N),dtype=float)
    for i in range(N):
        for j in range(i+1,N):
            ctx,first,md=pair_models[i,j]
            X=np.column_stack([cont[:,ctx],binary[:,i]])
            pj=md.predict_proba(X)[:,1]
            pi=first.predict_proba(cont[:,ctx])[:,1]
            prob_i=np.where(binary[:,i]==1,pi,1-pi)
            prob_j=np.where(binary[:,j]==1,pj,1-pj)
            a=-np.log(np.clip(prob_i*prob_j,EPS,1))
            PA[:,i,j]=PA[:,j,i]=a
    return A_node,PA

An_train,Ap_train=concept_scores(train_cont,train_bin)
An_val,Ap_val=concept_scores(val_cont,val_bin)

train_y=train.y.astype(int).to_numpy()
train_place=train.place.astype(int).to_numpy()
train_groups=2*train_y+train_place
counts=np.bincount(train_groups,minlength=4)
weights=np.array([len(train)/(4*counts[g]) for g in train_groups])
cn_models=[]
for i in tqdm(range(N),desc='Fit context-conditional concept/class probabilities'):
    md=make_pipeline(StandardScaler(),LogisticRegression(C=0.2,max_iter=1500,random_state=SEED))
    md.fit(train_cont,train_y,logisticregression__sample_weight=weights)
    cn_models.append(md)
    break
cn_model=cn_models[0]
class_prob=cn_model.predict_proba(val_cont)[:,1]

ctx_cols={
    'A_cc_active_mean':[], 'A_cc_active_max':[], 'A_cc_active_top3':[],
    'A_cc_all_mean':[], 'A_cn_predicted':[], 'A_cn_mismatch':[],
    'A_cc_bird_background':[], 'A_cc_activation_weighted':[],
    'A_node_mean':[], 'A_node_max':[],
}
for k in range(len(paths)):
    active=np.flatnonzero(val_bin[k]==1)
    pi,pj=np.triu_indices(N,1)
    pair_values=Ap_val[k,pi,pj]
    active_pairs=np.array([Ap_val[k,i,j] for i in active for j in active if i<j],dtype=float)
    if not len(active_pairs):active_pairs=pair_values
    ctx_cols['A_cc_active_mean'].append(float(active_pairs.mean()))
    ctx_cols['A_cc_active_max'].append(float(active_pairs.max()))
    ctx_cols['A_cc_active_top3'].append(float(np.sort(active_pairs)[-3:].mean()))
    ctx_cols['A_cc_all_mean'].append(float(pair_values.mean()))
    ctx_cols['A_node_mean'].append(float(An_val[k].mean()))
    ctx_cols['A_node_max'].append(float(An_val[k].max()))
    bg=[4,5,6,7,8,9];bird=[0,1,2,3]
    ctx_cols['A_cc_bird_background'].append(float(np.mean([Ap_val[k,i,j] for i in bird for j in bg])))
    activ=1/(1+np.exp(-15*(val_cont[k]-thresholds)))
    wij=np.outer(activ,activ)[pi,pj]
    ctx_cols['A_cc_activation_weighted'].append(float(np.sum(wij*pair_values)/(np.sum(wij)+EPS)))
    cls=int(features[MODELS[0]].predicted_class.iloc[k])
    p=float(np.clip(class_prob[k],EPS,1-EPS))
    ctx_cols['A_cn_predicted'].append(float(-np.log(p if cls==1 else 1-p)))
    ctx_cols['A_cn_mismatch'].append(float(abs(cls-p)))

semantic_context=pd.DataFrame({'path':paths,**ctx_cols})
semantic_context=semantic_context.drop(columns=['A_cn_predicted','A_cn_mismatch'])
semantic_context.to_csv(OUT/'contextual_anomaly_validation.csv',index=False)
semantic_ctx_cols=list(semantic_context.columns.drop('path'))

def new_detector():return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=4000,random_state=SEED))
def oof(d,cols,conditional=True):
    y=d.error.to_numpy(dtype=int);pred=d.predicted_class.to_numpy(dtype=int)
    ans=np.full(len(d),np.nan)
    groups=[np.arange(len(d))] if not conditional else [np.flatnonzero(pred==0),np.flatnonzero(pred==1)]
    for idx in groups:
        labels=y[idx];cnt=np.bincount(labels,minlength=2)
        if min(cnt)<2:raise RuntimeError('Not enough errors in OOF class: '+str(cnt))
        skf=StratifiedKFold(n_splits=min(CV,int(min(cnt))),shuffle=True,random_state=SEED)
        X=d.iloc[idx][cols].to_numpy(dtype=float)
        if not np.isfinite(X).all():raise RuntimeError('Nonfinite input in '+str(cols))
        for tr,te in skf.split(X,labels):
            model=new_detector();model.fit(X[tr],labels[tr])
            ans[idx[te]]=model.predict_proba(X[te])[:,1]
    if not np.isfinite(ans).all():raise RuntimeError('Incomplete OOF')
    return ans

def metrics(y,p):
    return float(roc_auc_score(y,p)),float(average_precision_score(y,p))
def delta_ci(y,a,b,nboot=500):
    rng=np.random.default_rng(SEED)
    y=np.asarray(y,dtype=int);a=np.asarray(a);b=np.asarray(b)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    dif=[]
    for _ in range(nboot):
        ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
        dif.append((roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]),
                    average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],b[ix])))
    z=np.array(dif)
    return [float(np.quantile(z[:,0],.025)),float(np.quantile(z[:,0],.975)),
            float(np.quantile(z[:,1],.025)),float(np.quantile(z[:,1],.975))]

results=[];comparisons=[];strata=[]
for model in MODELS:
    d=features[model].merge(semantic_context,on='path',validate='one_to_one',how='left')
    s=2*d.predicted_class.to_numpy(dtype=float)-1
    d['clip_land_score']=val_cls[:,0]
    d['clip_water_score']=val_cls[:,1]
    d['clip_margin']=val_clip_margin
    d['clip_disagree']=(-s*val_clip_margin)
    d['clip_pred_mismatch']=(clip_pred!=d.predicted_class.to_numpy(dtype=int)).astype(float)
    d['clip_bird_crop_disagree']=-s*d.vlm_bird_class_margin.to_numpy(dtype=float)
    d['cn_anomaly']=-np.log(np.clip(np.where(d.predicted_class.to_numpy()==1,class_prob,1-class_prob),EPS,1))
    d['cn_mismatch']=np.abs(d.predicted_class.to_numpy(dtype=float)-class_prob)
    ce=[]
    for i in range(N):
        z=val_cont.copy();z[:,i]=np.median(train_cont[:,i])
        p=cn_model.predict_proba(z)[:,1]
        base=-np.log(np.clip(np.where(d.predicted_class.to_numpy()==1,class_prob,1-class_prob),EPS,1))
        alt=-np.log(np.clip(np.where(d.predicted_class.to_numpy()==1,p,1-p),EPS,1))
        ce.append(base-alt)
    ce=np.column_stack(ce)
    d['cn_concept_effect_abs']=np.abs(ce).mean(axis=1)
    d['cn_concept_effect_max']=np.abs(ce).max(axis=1)
    d['cn_concept_support_conflict']=np.maximum(ce,0).mean(axis=1)
    concept_class_cols=['cn_anomaly','cn_mismatch','cn_concept_effect_abs',
                        'cn_concept_effect_max','cn_concept_support_conflict']
    b=sorted([c for c in d if c.startswith('B_') and c[2:].isdigit()],key=lambda x:int(x[2:]))
    a=sorted([c for c in d if c.startswith('activation_') and c.rsplit('_',1)[-1].isdigit()],key=lambda x:int(x.rsplit('_',1)[-1]))
    interactions=['interaction_abs','interaction_signed','cross_interaction','signed_cross_interaction']
    independent_clip=['clip_land_score','clip_water_score','clip_margin','clip_disagree','clip_pred_mismatch']
    bird_clip=['vlm_bird_class_margin','clip_bird_crop_disagree']
    simple=['confidence']
    cnn=['confidence']+b+a
    cnn_graph=cnn+interactions
    semcn=semantic_ctx_cols+concept_class_cols
    specs={
        'confidence':simple,
        'CLIP_second_classifier_only':independent_clip,
        'confidence_plus_CLIP':simple+independent_clip,
        'confidence_CLIP_bird_crop':simple+independent_clip+bird_clip,
        'CNN_graph_without_CLIP':cnn_graph,
        'CNN_graph_plus_CLIP':cnn_graph+independent_clip,
        'CNN_graph_plus_CLIP_crop':cnn_graph+independent_clip+bird_clip,
        'CNN_graph_CLIP_plus_Acn':cnn_graph+independent_clip+bird_clip+concept_class_cols,
        'CNN_graph_CLIP_plus_Acc':cnn_graph+independent_clip+bird_clip+semantic_ctx_cols,
        'CNN_graph_CLIP_plus_Acc_Acn':cnn_graph+independent_clip+bird_clip+semcn,
        'confidence_CLIP_plus_Acc_Acn':simple+independent_clip+bird_clip+semcn,
        'CNN_contributions_CLIP_plus_Acc_Acn':cnn+independent_clip+bird_clip+semcn,
    }
    y=d.error.to_numpy(dtype=int)
    pred=pd.DataFrame({'path':d.path,'error':y,'predicted_class':d.predicted_class,
                       'true_class_EVAL_ONLY':d.y,'background_EVAL_ONLY':d.place})
    for key,cols in tqdm(specs.items(),desc=model+' OOF error-detector controls'):
        if len(set(cols))!=len(cols):raise RuntimeError('Duplicate features for '+key)
        missing=set(cols)-set(d)
        if missing:raise RuntimeError('Missing '+str(missing))
        pred[key]=oof(d,cols,conditional=True)
        roc,pr=metrics(y,pred[key])
        results.append(dict(model=model,detector=key,n_val=len(d),errors=int(y.sum()),AUROC=roc,AUPRC=pr))
        for pc in (0,1):
            ix=d.predicted_class==pc
            roc1,pr1=metrics(y[ix],pred.loc[ix,key])
            strata.append(dict(model=model,predicted_class=pc,detector=key,n=int(ix.sum()),errors=int(y[ix].sum()),AUROC=roc1,AUPRC=pr1))
    pred.to_csv(OUT/f'{model}_validation_oof_predictions.csv',index=False)
    audit_pairs=[
        ('confidence_plus_CLIP','confidence'),
        ('CNN_graph_plus_CLIP','confidence_plus_CLIP'),
        ('CNN_graph_plus_CLIP_crop','confidence_CLIP_bird_crop'),
        ('CNN_graph_CLIP_plus_Acn','CNN_graph_plus_CLIP_crop'),
        ('CNN_graph_CLIP_plus_Acc','CNN_graph_plus_CLIP_crop'),
        ('CNN_graph_CLIP_plus_Acc_Acn','CNN_graph_plus_CLIP_crop'),
        ('CNN_graph_CLIP_plus_Acc_Acn','CNN_graph_CLIP_plus_Acn'),
        ('CNN_graph_CLIP_plus_Acc_Acn','CNN_graph_CLIP_plus_Acc'),
        ('CNN_graph_CLIP_plus_Acc_Acn','confidence_CLIP_plus_Acc_Acn'),
        ('CNN_graph_CLIP_plus_Acc_Acn','CNN_contributions_CLIP_plus_Acc_Acn'),
    ]
    for upper,lower in audit_pairs:
        au,pu=metrics(y,pred[upper]);al,pl=metrics(y,pred[lower])
        c=delta_ci(y,pred[upper],pred[lower])
        comparisons.append(dict(model=model,contrast=upper+' MINUS '+lower,
                                delta_AUROC=au-al,delta_AUPRC=pu-pl,
                                AUROC_CI_low=c[0],AUROC_CI_high=c[1],
                                AUPRC_CI_low=c[2],AUPRC_CI_high=c[3]))

res=pd.DataFrame(results);comp=pd.DataFrame(comparisons);sub=pd.DataFrame(strata)
res.to_csv(OUT/'validation_OOF_detector_results.csv',index=False)
comp.to_csv(OUT/'validation_OOF_paired_ablation.csv',index=False)
sub.to_csv(OUT/'validation_OOF_predicted_class_subgroups.csv',index=False)
with open(OUT/'experiment_protocol.json','w') as f:
    json.dump(dict(description='CLIP classifier controls and TRAIN-fitted contextual concept anomaly proxies',
                   clip=CLIP_NAME,concepts=[k for k,_ in CONCEPTS],concept_threshold_quantile=.70,
                   train_samples=len(train),validation_samples=len(paths),
                   folds=CV,regularization_C=C,
                   training='semantic probability models trained on TRAIN; detectors OOF on VALIDATION',
                   evaluation='NO TEST consulted or read',
                   caution=[
                       'Vocabulary is curated rather than automatically identified CNN activation concepts.',
                       'Concept presence is CLIP threshold pseudo-label, not validated semantic presence.',
                       'A_cc is a fitted context-conditional joint pseudo-likelihood; not grounded semantic truth.',
                       'A_cn is a group-balanced TRAIN-learned P(y|CLIP concept scores), may retain spurious correlations.',
                       'No alignment between semantic pair nodes and CNN cluster IDs; not the full h_ij graph detector.',
                       'Independent CLIP may classify birds; strict matched baselines test incremental value.',
                       'Validation was previously used for CNN selection; OOF is not fresh independent confirmation.',
                       'Bootstrap intervals hold fixed OOF predictions and do not account for retraining/hyperparameter selection.',
                       'Repeated exploratory comparisons can cause selection bias.'
                   ]),f,indent=2)
for model in MODELS:
    print('\n===== MODEL',model,'=====')
    print(res[res.model==model][['detector','AUROC','AUPRC']].sort_values('AUROC',ascending=False).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nPAIRED ABLATIONS')
    print(comp[comp.model==model].drop(columns=['model']).to_string(index=False,float_format=lambda x:f'{x:+.4f}'))
    print('\nPREDICTED-CLASS METRICS (main controls)')
    interesting=['confidence_plus_CLIP','CNN_graph_plus_CLIP_crop','CNN_graph_CLIP_plus_Acc_Acn']
    print(sub[(sub.model==model)&sub.detector.isin(interesting)].drop(columns=['model']).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nSaved:',OUT)
print('IMPORTANT: This is a partial proxy for contextual anomaly graph, not a validated full semantic graph.')