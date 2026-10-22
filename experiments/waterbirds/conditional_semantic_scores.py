import json, random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import expit

SEED=20261014; EPS=1e-8; CV=5; BOOT=500; C=0.1
np.random.seed(SEED);random.seed(SEED)
ROOT=Path('/kaggle/working/waterbirds_cig_dual_comparison')
INPUT=ROOT/'clip_control_contextual_anomaly'/'clip_train_val_scores.npz'
OUT=ROOT/'anomaly_score_family_validation'
OUT.mkdir(parents=True,exist_ok=True)
MODELS=['old_spurious','new_groupdro']
CONCEPTS=['bird','wing','feathers','beak','water','grass','trees','sky','reeds','rocks']
N=len(CONCEPTS);BIRD=[0,1,2,3];BG=[4,5,6,7,8,9]
if not INPUT.exists():
    raise FileNotFoundError(f'Run waterbirds_clip_control_contextual_anomaly.py first. Missing {INPUT}')
cache=np.load(INPUT,allow_pickle=False)
tr_s=np.asarray(cache['train_score'],dtype=float)
va_s=np.asarray(cache['val_score'],dtype=float)
tr_paths=cache['train_paths'].astype(str);va_paths=cache['val_paths'].astype(str)
if tr_s.shape!=(len(tr_paths),N+4) or va_s.shape!=(len(va_paths),N+4):
    raise RuntimeError('CLIP feature cache has unexpected shape. Check concept vocabulary / prompts.')
if len(set(tr_paths))!=len(tr_paths) or len(set(va_paths))!=len(va_paths):
    raise RuntimeError('Duplicate paths in CLIP cache')
if set(tr_paths)&set(va_paths):raise RuntimeError('TRAIN and VAL images overlap')
import glob
roots=[Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),
       Path('/kaggle/input/waterbird/waterbird'),Path('/kaggle/input/waterbird')]
data_root=next((x for x in roots if (x/'metadata.csv').exists()),None)
if data_root is None:
    cs=glob.glob('/kaggle/input/**/metadata.csv',recursive=True)
    if not cs:raise FileNotFoundError('Waterbirds metadata.csv unavailable')
    data_root=Path(cs[0]).parent
meta=pd.read_csv(data_root/'metadata.csv')
meta['path']=meta.img_filename.map(lambda v:str(data_root/str(v)))
train=meta.loc[meta.split==0].set_index('path')
if not set(tr_paths).issubset(set(train.index)):
    raise RuntimeError('Cached TRAIN paths do not match Waterbirds metadata')
train=train.loc[tr_paths]
ytr=train.y.to_numpy(dtype=int);pltr=train.place.to_numpy(dtype=int)
if len(train)!=len(tr_s):raise RuntimeError('TRAIN rows mismatch')

frames={}
for model in MODELS:
    f=ROOT/model/'validation_features.csv'
    if not f.exists():raise FileNotFoundError(f'First run dual CIG pipeline: {f}')
    d=pd.read_csv(f)
    needed={'path','error','predicted_class','y','place','confidence','interaction_abs',
            'interaction_signed','cross_interaction','signed_cross_interaction'}
    if needed-set(d.columns):raise RuntimeError(f'{model} missing {needed-set(d.columns)}')
    if d.path.duplicated().any() or set(d.path.astype(str))!=set(va_paths):
        raise RuntimeError(f'{model} validation paths mismatch with CLIP cache')
    d=d.set_index('path').loc[va_paths].reset_index()
    if not (d.error.to_numpy(dtype=int)==(d.y!=d.predicted_class).to_numpy(dtype=int)).all():
        raise RuntimeError(f'{model}: incorrect error labels')
    frames[model]=d

Xtr=tr_s[:,:N];Xva=va_s[:,:N]
thresholds=np.quantile(Xtr,0.7,axis=0)
Ztr=(Xtr>=thresholds).astype(int)
Zva=(Xva>=thresholds).astype(int)

def lr(Cval=0.3):
    return make_pipeline(StandardScaler(),LogisticRegression(C=Cval,max_iter=1500,random_state=SEED))
def prob_obs(p,z):return np.where(z==1,p,1-p)
def fit_binary(X,y,w=None):
    if len(np.unique(y))!=2:raise RuntimeError('Degenerate CLIP concept threshold')
    p=lr();p.fit(X,y,logisticregression__sample_weight=w) if w is not None else p.fit(X,y)
    return p

a_pair=np.zeros((len(Xva),N,N),dtype=np.float32)
pmi_pair=np.zeros_like(a_pair)
for i in tqdm(range(N),desc='Fit contextual pair models (TRAIN only)'):
    for j in range(i+1,N):
        ctx=[k for k in range(N) if k not in (i,j)]
        ctx_tr=Xtr[:,ctx];ctx_va=Xva[:,ctx]
        fit_i=fit_binary(ctx_tr,Ztr[:,i]);fit_j=fit_binary(ctx_tr,Ztr[:,j])
        fit_cond=fit_binary(np.column_stack([ctx_tr,Ztr[:,i]]),Ztr[:,j])
        pi=prob_obs(fit_i.predict_proba(ctx_va)[:,1],Zva[:,i])
        pj=prob_obs(fit_j.predict_proba(ctx_va)[:,1],Zva[:,j])
        pj_cond=prob_obs(fit_cond.predict_proba(
            np.column_stack([ctx_va,Zva[:,i]]))[:,1],Zva[:,j])
        joint=np.clip(pi*pj_cond,EPS,1.0)
        surprisal=-np.log(joint)
        negative_pmi=np.log(np.clip(pi*pj,EPS,1)/joint)
        a_pair[:,i,j]=a_pair[:,j,i]=surprisal
        pmi_pair[:,i,j]=pmi_pair[:,j,i]=negative_pmi

groups=2*ytr+pltr
count=np.bincount(groups,minlength=4)
weights=len(groups)/(4*count[groups])
cls_model=fit_binary(Xtr,ytr,w=weights)
p_class=cls_model.predict_proba(Xva)[:,1]
node_cf=np.zeros((len(Xva),N),dtype=float)
for i in range(N):
    Xalt=Xva.copy();Xalt[:,i]=np.median(Xtr[:,i]);node_cf[:,i]=cls_model.predict_proba(Xalt)[:,1]

pi,pj=np.triu_indices(N,1)
AP=a_pair[:,pi,pj].astype(float)
PM=pmi_pair[:,pi,pj].astype(float)
np.savez_compressed(OUT/'semantic_pair_scores_validation.npz',paths=va_paths,
                    surprisal=AP,negative_conditional_pmi=PM,
                    concept_pair_indices=np.stack([pi,pj],axis=1))
active=(Zva[:,pi]*Zva[:,pj]).astype(bool)
sem=pd.DataFrame({'path':va_paths})
sem['cc_surprisal_mean']=AP.mean(axis=1)
sem['cc_surprisal_active']=(AP*active).sum(axis=1)/np.maximum(active.sum(axis=1),1)
sem['cc_surprisal_top3']=np.sort(AP,axis=1)[:,-3:].mean(axis=1)
sem['cc_negative_pmi_mean']=PM.mean(axis=1)
sem['cc_negative_pmi_positive']=np.maximum(PM,0).mean(axis=1)
sem['cc_negative_pmi_active']=(np.maximum(PM,0)*active).sum(axis=1)/np.maximum(active.sum(axis=1),1)
sem['cc_negative_pmi_top3']=np.sort(PM,axis=1)[:,-3:].mean(axis=1)
bg_pair=np.isin(pi,BIRD)&np.isin(pj,BG)
sem['cc_surprisal_bird_bg']=AP[:,bg_pair].mean(axis=1)
sem['cc_negative_pmi_bird_bg']=np.maximum(PM[:,bg_pair],0).mean(axis=1)
sem['clip_full_land']=va_s[:,N:N+2].mean(1)
sem['clip_full_water']=va_s[:,N+2:N+4].mean(1)
sem['clip_full_margin']=sem.clip_full_water-sem.clip_full_land
sem['semantic_bird_fraction']=Zva[:,BIRD].mean(axis=1)
sem['semantic_background_fraction']=Zva[:,BG].mean(axis=1)

base_sem=['cc_surprisal_mean','cc_surprisal_active','cc_surprisal_top3',
          'cc_negative_pmi_mean','cc_negative_pmi_positive',
          'cc_negative_pmi_active','cc_negative_pmi_top3',
          'cc_surprisal_bird_bg','cc_negative_pmi_bird_bg']
clip_controls=['clip_full_land','clip_full_water','clip_full_margin']

def conditional_oof(d,cols):
    ans=np.full(len(d),np.nan)
    y=d.error.to_numpy(dtype=int);pc=d.predicted_class.to_numpy(dtype=int)
    for cl in [0,1]:
        idx=np.flatnonzero(pc==cl); yy=y[idx]
        n=min(CV,int(np.bincount(yy,minlength=2).min()))
        if n<2:raise RuntimeError(f'Not enough positives and negatives in predicted class {cl}')
        folds=StratifiedKFold(n_splits=n,shuffle=True,random_state=SEED)
        XX=d.iloc[idx][cols].to_numpy(dtype=float)
        if not np.isfinite(XX).all():raise RuntimeError('Nonfinite features: '+str(cols))
        for tr,te in folds.split(XX,yy):
            mdl=lr(C);mdl.fit(XX[tr],yy[tr]);ans[idx[te]]=mdl.predict_proba(XX[te])[:,1]
    if not np.isfinite(ans).all():raise RuntimeError('OOF output incomplete')
    return ans

def met(y,s):
    return (float(roc_auc_score(y,s)),float(average_precision_score(y,s)))
def boot_delta(y,a,b,its=BOOT):
    rng=np.random.default_rng(SEED)
    y=np.asarray(y,dtype=int);a=np.asarray(a);b=np.asarray(b)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    z=[]
    for _ in range(its):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        z.append([roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx]),
                  average_precision_score(y[idx],a[idx])-average_precision_score(y[idx],b[idx])])
    z=np.array(z)
    return [float(v) for v in [*np.quantile(z[:,0],[.025,.975]),*np.quantile(z[:,1],[.025,.975])]]

rows=[];paired=[];strata=[]
for model in MODELS:
    d=frames[model].merge(sem,on='path',how='left',validate='one_to_one')
    if d[base_sem+clip_controls].isna().any().any():raise RuntimeError('Semantic merge failed')
    predicted=d.predicted_class.to_numpy(dtype=int)
    cls_p=np.clip(np.where(predicted==1,p_class,1-p_class),EPS,1)
    d['cn_class_surprisal']=-np.log(cls_p)
    pcf=np.clip(np.where(predicted[:,None]==1,node_cf,1-node_cf),EPS,1)
    nllcf=-np.log(pcf)
    nll=-np.log(cls_p)[:,None]
    cn_delta=nll-nllcf
    d['cn_ref_discrepancy_mean']=np.abs(cn_delta).mean(axis=1)
    d['cn_ref_discrepancy_max']=np.abs(cn_delta).max(axis=1)
    d['cn_ref_conflict']=np.maximum(cn_delta,0).mean(axis=1)
    importance=np.abs(cn_delta)
    d['cn_task_bg_excess']=importance[:,BG].mean(axis=1)-importance[:,BIRD].mean(axis=1)
    w=np.abs(d.interaction_abs.to_numpy(dtype=float))
    w_scale=np.quantile(w,.95) if np.quantile(w,.95)>0 else 1.
    w=np.log1p(w/w_scale)
    d['global_interaction_x_surprisal_PROXY']=w*d.cc_surprisal_active.to_numpy()
    d['global_interaction_x_pmi_PROXY']=w*d.cc_negative_pmi_active.to_numpy()
    d['global_interaction_x_cn_PROXY']=w*d.cn_task_bg_excess.to_numpy()
    d['clip_predicted_disagree']=(2*predicted-1)*(-d.clip_full_margin.to_numpy())
    d['clip_pred_mismatch']=(d.clip_full_margin.to_numpy()>0).astype(int)!=predicted
    d['clip_pred_mismatch']=d['clip_pred_mismatch'].astype(float)
    Bcols=sorted([c for c in d if c.startswith('B_') and c[2:].isdigit()],key=lambda x:int(x[2:]))
    Acols=sorted([c for c in d if c.startswith('activation_') and c.split('_')[-1].isdigit()],key=lambda x:int(x.split('_')[-1]))
    Icols=['interaction_abs','interaction_signed','cross_interaction','signed_cross_interaction']
    confidence=['confidence']
    clip=clip_controls+['clip_predicted_disagree','clip_pred_mismatch']
    cnn=confidence+Bcols+Acols
    graph=cnn+Icols
    cn=['cn_class_surprisal','cn_ref_discrepancy_mean','cn_ref_discrepancy_max',
        'cn_ref_conflict','cn_task_bg_excess']
    surprise=[x for x in base_sem if x.startswith('cc_surprisal')]
    pmi=[x for x in base_sem if x.startswith('cc_negative_pmi')]
    proxy=['global_interaction_x_surprisal_PROXY','global_interaction_x_pmi_PROXY',
           'global_interaction_x_cn_PROXY']
    specs={
      'confidence':confidence,
      'confidence_CLIP':confidence+clip,
      'cnn_contributions_CLIP':cnn+clip,
      'cnn_graph_CLIP':graph+clip,
      'cnn_graph_CLIP_surprisal':graph+clip+surprise,
      'cnn_graph_CLIP_PMI':graph+clip+pmi,
      'cnn_graph_CLIP_class_anomaly':graph+clip+cn,
      'cnn_graph_CLIP_interaction_weighted_PROXY':graph+clip+proxy,
      'cnn_graph_CLIP_all_anomaly_scores':graph+clip+surprise+pmi+cn+proxy,
    }
    d.to_csv(OUT/f'{model}_validation_anomaly_features.csv',index=False)
    out=pd.DataFrame({'path':d.path,'error':d.error,'predicted_class':d.predicted_class,
                      'true_y_EVAL_ONLY':d.y,'place_EVAL_ONLY':d.place})
    for name,cols in tqdm(specs.items(),desc=f'{model}: OOF anomaly-score ablations'):
        if len(cols)!=len(set(cols)):raise RuntimeError(f'Duplicate features: {name}')
        out[name]=conditional_oof(d,cols)
        auc,ap=met(d.error,out[name]);rows.append(dict(model=model,detector=name,n_val=len(d),
                                                        errors=int(d.error.sum()),AUROC=auc,AUPRC=ap))
        for cl in (0,1):
            ix=d.predicted_class==cl
            auc0,ap0=met(d.loc[ix,'error'],out.loc[ix,name])
            strata.append(dict(model=model,predicted_class=cl,detector=name,n=int(ix.sum()),
                               errors=int(d.loc[ix,'error'].sum()),AUROC=auc0,AUPRC=ap0))
    out.to_csv(OUT/f'{model}_validation_OOF_predictions.csv',index=False)
    contrasts=[('cnn_graph_CLIP','confidence_CLIP'),
               ('cnn_graph_CLIP','cnn_contributions_CLIP'),
               ('cnn_graph_CLIP_surprisal','cnn_graph_CLIP'),
               ('cnn_graph_CLIP_PMI','cnn_graph_CLIP'),
               ('cnn_graph_CLIP_class_anomaly','cnn_graph_CLIP'),
               ('cnn_graph_CLIP_interaction_weighted_PROXY','cnn_graph_CLIP'),
               ('cnn_graph_CLIP_all_anomaly_scores','cnn_graph_CLIP'),
               ('cnn_graph_CLIP_all_anomaly_scores','cnn_graph_CLIP_class_anomaly')]
    for upper,lower in contrasts:
        au,pr=met(d.error,out[upper]);ab,pb=met(d.error,out[lower]);ci=boot_delta(d.error,out[upper],out[lower])
        paired.append(dict(model=model,contrast=f'{upper} MINUS {lower}',delta_AUROC=au-ab,
                           delta_AUPRC=pr-pb,AUROC_CI_lo=ci[0],AUROC_CI_hi=ci[1],
                           AUPRC_CI_lo=ci[2],AUPRC_CI_hi=ci[3]))
        for cl in (0,1):
            ix=(d.predicted_class==cl).to_numpy()
            auc1,ap1=met(d.error.to_numpy()[ix],out.loc[ix,upper]);auc0,ap0=met(d.error.to_numpy()[ix],out.loc[ix,lower])
            strata.append(dict(model=model,predicted_class=cl,detector=f'DELTA: {upper} - {lower}',
                               n=int(ix.sum()),errors=int(d.error.to_numpy()[ix].sum()),
                               AUROC=auc1-auc0,AUPRC=ap1-ap0))
    print('\n==',model,'==')
    subset=pd.DataFrame(rows).query('model == @model')
    print(subset[['detector','AUROC','AUPRC']].sort_values('AUROC',ascending=False)
          .to_string(index=False,float_format=lambda x:f'{x:.4f}'))

pd.DataFrame(rows).to_csv(OUT/'oof_validation_results.csv',index=False)
pd.DataFrame(paired).to_csv(OUT/'paired_score_ablation_bootstrap.csv',index=False)
pd.DataFrame(strata).to_csv(OUT/'predicted_class_metrics_and_deltas.csv',index=False)
print('\n=== Paired exploratory comparisons ===')
print(pd.DataFrame(paired).to_string(index=False,float_format=lambda x:f'{x:+.4f}'))
with open(OUT/'METHOD_LIMITATIONS.json','w') as f:
    json.dump(dict(train_only_semantic_fit=True,validation_OOF_detectors=True,test_accessed=False,
      vocabulary=CONCEPTS,threshold='Train 70th percentile per CLIP concept',
      scores=['conditional joint surprisal','negative conditional pointwise mutual information',
              'class-NLL based discrepancy','interaction-weighted aggregate proxy'],
      nonclaims=['CLIP threshold does not establish verified concept presence',
                 'Pairwise semantic scores are defined on 10 prompt concepts, while CNN graph uses 16 clusters',
                 'Global interaction multiplication is not aligned edgewise interaction W_ij*A_cc_ij',
                 'Class models may learn spurious training correlations despite group weights',
                 'The validation set was previously reused for model development: exploratory only',
                 'Bootstrap treats the fitted detector scores as fixed: does not measure retraining uncertainty',
                 'Neither concept coherence nor causal semantics are established by this score ablation']),f,indent=2)
print('\nOutputs:',OUT)
print('IMPORTANT: This is a SCORE FAMILY ABLATION with unmatched semantic/CNN nodes.')
print('For literal h_ij edge alignment, re-extract per-pair CNN interactions for the SAME semantic concepts.')