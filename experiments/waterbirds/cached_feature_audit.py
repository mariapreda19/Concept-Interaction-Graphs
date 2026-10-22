
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 20261014
RNG = np.random.default_rng(SEED)
ROOT = Path("/kaggle/working/waterbirds_cig_dual_comparison")
OUT = ROOT / "final_controls_audit"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("old_spurious", "new_groupdro")
B = 700
C = 0.1
print("Audit folder:", OUT)

def load_features(model):
    folder = ROOT / model
    vfile = folder / "validation_features.csv"
    tfile = folder / "test_features.csv"
    if not vfile.exists() or not tfile.exists():
        raise FileNotFoundError(
            f"Missing graph feature CSVs for {model}. Re-run waterbirds_cig_dual_comparison.py "
            "first, or attach its Kaggle output directory to this notebook."
        )
    v, t = pd.read_csv(vfile), pd.read_csv(tfile)
    for key, d in [('validation', v), ('test', t)]:
        needed = {'path','y','place','predicted_class','error','confidence'}
        missing=needed-set(d.columns)
        if missing: raise ValueError(f"{model} {key} missing columns: {sorted(missing)}")
        assert not d.path.duplicated().any(), f"{model} {key} has duplicate paths"
        assert (d.error.astype(int) == (d.predicted_class.astype(int)!=d.y.astype(int))).all()
        assert np.isfinite(d.confidence.to_numpy(dtype=float)).all()
    assert set(v.path).isdisjoint(set(t.path)), "Train/evaluation paths overlap"
    return v.reset_index(drop=True), t.reset_index(drop=True)

def make_pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C,max_iter=3000,solver='lbfgs',random_state=SEED)
    )

def score_fit(v,t,columns,by_class=False):
    missing = set(columns)-set(v.columns)
    if missing: raise ValueError("Missing features: "+str(sorted(missing)))
    result = np.full(len(t), np.nan)
    if not by_class:
        if v.error.nunique()<2:
            result[:] = float(v.error.mean())
        else:
            p=make_pipe()
            p.fit(v[columns].to_numpy(dtype=float),v.error.to_numpy(dtype=int))
            result[:]=p.predict_proba(t[columns].to_numpy(dtype=float))[:,1]
    else:
        for cls in (0,1):
            vi=v[v.predicted_class==cls]
            idx=np.flatnonzero(t.predicted_class.to_numpy()==cls)
            if not len(idx): continue
            if not len(vi): raise ValueError(f"No validation rows with predicted class {cls}")
            if vi.error.nunique()<2:
                result[idx]=float(vi.error.mean())
            else:
                p=make_pipe()
                p.fit(vi[columns].to_numpy(dtype=float),vi.error.to_numpy(dtype=int))
                result[idx]=p.predict_proba(t.iloc[idx][columns].to_numpy(dtype=float))[:,1]
    if not np.isfinite(result).all(): raise ValueError("Non-finite detector predictions")
    return result

def safe_metrics(y,s):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    out = dict(n=len(y),errors=int(y.sum()),error_rate=float(y.mean()) if len(y) else np.nan)
    out['AUROC']=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else np.nan
    out['AUPRC']=float(average_precision_score(y,s)) if len(np.unique(y))==2 else np.nan
    return out

def differences(y,sa,sb,group=None,nboot=B):
    y=np.asarray(y,dtype=int); sa=np.asarray(sa);sb=np.asarray(sb)
    if group is None:
        idx_groups=[np.arange(len(y))]
    else:
        keys=np.asarray(group)
        idx_groups=[np.flatnonzero(keys==key) for key in np.unique(keys)]
    paired=[]
    for _ in range(nboot):
        inds=np.concatenate([RNG.choice(ix,size=len(ix),replace=True) for ix in idx_groups])
        if len(np.unique(y[inds]))<2: continue
        paired.append([
            roc_auc_score(y[inds],sa[inds])-roc_auc_score(y[inds],sb[inds]),
            average_precision_score(y[inds],sa[inds])-average_precision_score(y[inds],sb[inds])
        ])
    arr=np.asarray(paired)
    if not len(arr):return dict(auroc_lo=np.nan,auroc_hi=np.nan,auprc_lo=np.nan,auprc_hi=np.nan)
    return dict(auroc_lo=float(np.quantile(arr[:,0],.025)),
                auroc_hi=float(np.quantile(arr[:,0],.975)),
                auprc_lo=float(np.quantile(arr[:,1],.025)),
                auprc_hi=float(np.quantile(arr[:,1],.975)))

def subgroup_table(t,pred,model):
    result=[]
    for (bird,place), ids in t.groupby(['y','place']).groups.items():
        yy=t.loc[ids,'error'].to_numpy()
        for detector in pred.columns:
            result.append(dict(model=model,bird=int(bird),place=int(place),
                               detector=detector,**safe_metrics(yy,pred.loc[ids,detector].to_numpy())))
    return pd.DataFrame(result)

def subgroup_macro(sub,model):
    return (sub.groupby('detector',sort=False)
            .agg(macro_group_AUROC=('AUROC','mean'),
                 macro_group_AUPRC=('AUPRC','mean'),
                 min_group_AUROC=('AUROC','min'),
                 max_group_AUROC=('AUROC','max'))
            .reset_index().assign(model=model))

summaries=[]; subgroup_results=[]; macro_results=[]; all_differences=[]
files=[]
for model in MODELS:
    v,t=load_features(model)
    cols_b=sorted((c for c in v.columns if c.startswith('B_')),
                  key=lambda s:int(s.split('_')[1]))
    cols_a=sorted((c for c in v.columns if c.startswith('activation_')),
                  key=lambda s:int(s.rsplit('_',1)[1]))
    sem=['semantic_discrepancy','bg_support','bird_opposition','bird_support']
    inter=['interaction_abs','interaction_signed','cross_interaction','signed_cross_interaction']
    for c in cols_b+cols_a+sem+inter:
        if c not in t.columns:raise ValueError(f"{model} test missing {c}")
    v=v.copy(); t=t.copy()
    v['pred_class_onehot']=(v.predicted_class==1).astype(float)
    t['pred_class_onehot']=(t.predicted_class==1).astype(float)
    graph_no_inter=['confidence']+cols_b+cols_a+sem
    graph_with_inter=graph_no_inter+inter

    pred=pd.DataFrame(index=t.index)
    pred['confidence_raw']=t.confidence.to_numpy()
    pred['confidence_calibrated']=score_fit(v,t,['confidence'])
    pred['class_only']=score_fit(v,t,['pred_class_onehot'])
    pred['confidence_plus_class']=score_fit(v,t,['confidence','pred_class_onehot'])
    pred['confidence_conditional']=score_fit(v,t,['confidence'],by_class=True)
    pred['graph_no_interaction_pooled']=score_fit(v,t,graph_no_inter)
    pred['graph_with_interaction_pooled']=score_fit(v,t,graph_with_inter)
    pred['graph_no_interaction_plus_class']=score_fit(v,t,graph_no_inter+['pred_class_onehot'])
    pred['graph_interaction_plus_class']=score_fit(v,t,graph_with_inter+['pred_class_onehot'])
    pred['graph_no_interaction_conditional']=score_fit(v,t,graph_no_inter,by_class=True)
    pred['graph_interaction_conditional']=score_fit(v,t,graph_with_inter,by_class=True)

    joined=t[['path','y','place','predicted_class','error']].join(pred)
    joined.to_csv(OUT/f'{model}_audit_predictions.csv',index=False)
    overall=pd.DataFrame([dict(model=model,detector=c,**safe_metrics(t.error,pred[c]))
                          for c in pred.columns])
    overall.to_csv(OUT/f'{model}_overall.csv',index=False)
    summaries.append(overall)
    sub=subgroup_table(t,pred,model);sub.to_csv(OUT/f'{model}_subgroups.csv',index=False)
    subgroup_results.append(sub)
    macro=subgroup_macro(sub,model);macro.to_csv(OUT/f'{model}_macro.csv',index=False)
    macro_results.append(macro)

    comparisons=[
        ('graph_interaction_conditional','confidence_conditional'),
        ('graph_interaction_conditional','graph_no_interaction_conditional'),
        ('graph_interaction_plus_class','confidence_plus_class'),
        ('graph_interaction_plus_class','graph_no_interaction_plus_class'),
        ('graph_with_interaction_pooled','graph_no_interaction_pooled'),
        ('confidence_conditional','confidence_plus_class'),
    ]
    group_keys=(t.y.astype(str)+'_'+t.place.astype(str)).to_numpy()
    diffrows=[]
    for a,b in comparisons:
        ma=safe_metrics(t.error,pred[a]);mb=safe_metrics(t.error,pred[b])
        ci=differences(t.error,pred[a],pred[b],group=group_keys)
        diffrows.append(dict(model=model,comparison=f'{a} - {b}',
                             delta_AUROC=ma['AUROC']-mb['AUROC'],
                             delta_AUPRC=ma['AUPRC']-mb['AUPRC'],**ci))
    dif=pd.DataFrame(diffrows);dif.to_csv(OUT/f'{model}_paired_comparisons.csv',index=False)
    all_differences.append(dif)
    print('\n'+'='*75+'\n'+model+'\n'+'='*75)
    print('Test n:',len(t),'error count:',int(t.error.sum()),
          'error prevalence:',round(float(t.error.mean()),4))
    print('\nOVERALL METRICS')
    print(overall.sort_values('AUROC',ascending=False)[['detector','AUROC','AUPRC']].to_string(index=False,float_format=lambda a:f'{a:.4f}'))
    print('\nGROUP MACRO METRICS')
    print(macro[['detector','macro_group_AUROC','min_group_AUROC']].sort_values(
        'macro_group_AUROC',ascending=False).to_string(index=False,float_format=lambda a:f'{a:.4f}'))
    focus=sub[sub.detector.isin(['confidence_conditional','graph_no_interaction_conditional',
                                 'graph_interaction_conditional'])]
    print('\nSUBGROUP AUROC / AUPRC')
    print(focus[['bird','place','n','errors','detector','AUROC','AUPRC']].to_string(index=False,float_format=lambda a:f'{a:.4f}'))
    print('\nPAIRED BOOTSTRAP: group-stratified resampling')
    print(dif.to_string(index=False,float_format=lambda a:f'{a:+.4f}'))

pd.concat(summaries,ignore_index=True).to_csv(OUT/'overall_all_models.csv',index=False)
pd.concat(subgroup_results,ignore_index=True).to_csv(OUT/'subgroups_all_models.csv',index=False)
pd.concat(macro_results,ignore_index=True).to_csv(OUT/'macro_all_models.csv',index=False)
pd.concat(all_differences,ignore_index=True).to_csv(OUT/'comparisons_all_models.csv',index=False)

struct=[]
for model in MODELS:
    f=ROOT/model/'concept_semantics.csv'
    if not f.exists(): continue
    c=pd.read_csv(f)
    label_counts=c.clip_label.value_counts()
    struct.append(dict(model=model,n_concepts=len(c),
                       n_distinct_top_labels=len(label_counts),
                       max_label_fraction=float(label_counts.iloc[0]/len(c)),
                       mean_label_agreement=float(c.label_agreement.mean()),
                       median_label_agreement=float(c.label_agreement.median()),
                       min_label_agreement=float(c.label_agreement.min())))
if struct:
    sem_summary=pd.DataFrame(struct)
    sem_summary.to_csv(OUT/'concept_coherence_audit.csv',index=False)
    print('\nCONCEPT SEMANTICS AUDIT (top-CLIP-label agreement only; not a full coherence test)')
    print(sem_summary.to_string(index=False,float_format=lambda a:f'{a:.4f}'))

with open(OUT/'README_protocol.json','w') as f:
    json.dump({
        'seed':SEED,'bootstrap_iterations':B,'regularization_C':C,
        'detector_training':'validation only; never fit on test',
        'class_condition':'predicted_class only; true class not used in detector',
        'subgroup_reporting':'true class and background used only after scoring',
        'bootstrap':'paired, stratified by true Waterbirds subgroup; CNN fixed',
        'limitations':[
            'Repeated exploratory testing on Waterbirds test; results are NOT independent confirmation.',
            'Models predict different error targets; compare each detector to its own CNN baselines.',
            'CLIP labels are heuristic and may lack coherent concept meaning.',
            'Macro AUPRC is prevalence dependent and should not be interpreted as a prevalence-free ranking measure.',
            'Group-stratified bootstrap does not include variation across CNN training seeds.',
            'Group-specific AUROC can be noisy with very few positive/negative examples.',
            'Repeated detector fits on validation may select to validation; this is an audit not confirmatory generalization.'
        ]},f,indent=2)
print('\nSAVED:',OUT)
print('Best files: overall_all_models.csv, macro_all_models.csv, subgroups_all_models.csv, comparisons_all_models.csv')
print('NO MODEL RETRAINING, NO TEST-BASED HYPERPARAMETER SELECTION.')