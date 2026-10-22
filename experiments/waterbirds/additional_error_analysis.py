from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

SEED = 42
N_BOOTSTRAP = 1000
SOURCE = Path('/kaggle/working/waterbirds_real1_20epochs')
CACHE = SOURCE / 'real_background_invariance'
OUT = SOURCE / 'experiment1_final_audit'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

files = {}
for split in ('validation', 'test'):
    path = CACHE / (split + '_real_backgrounds.csv')
    if not path.exists():
        raise FileNotFoundError('Missing ' + str(path) + '. Restore the previous Kaggle output or rerun the real-background experiment.')
    files[split] = pd.read_csv(path)

validation = files['validation'].copy()
testing = files['test'].copy()
for split, df in files.items():
    required = ['img_filename', 'y', 'place', 'error', 'm', 'B_o', 'B_b', 'I_ob', 'real_flip_rate']
    missing = []
    for col in required:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(split + ': missing columns ' + str(missing))
    if df['img_filename'].duplicated().any():
        raise ValueError(split + ': duplicate filenames')
    if not df['error'].isin([0, 1]).all():
        raise ValueError(split + ': error must be binary')
    if not df['y'].isin([0, 1]).all() or not df['place'].isin([0, 1]).all():
        raise ValueError(split + ': y and place must be binary')
    df['confidence'] = df['m'].abs()
    if not np.isfinite(df[['confidence', 'B_o', 'B_b', 'I_ob', 'real_flip_rate']].to_numpy(dtype=float)).all():
        raise ValueError(split + ': nonfinite values')
    if ((df['m'] > 0).astype(int) != (df['y'].astype(int) != df['error'].astype(int)).astype(int)).any():
        raise ValueError(split + ': predictions inferred from margin disagree with error labels')

feature_sets = {
    'A_confidence': ['confidence'],
    'B_flip_only': ['real_flip_rate'],
    'C_confidence_flip': ['confidence', 'real_flip_rate'],
    'D_contributions_flip': ['confidence', 'B_o', 'B_b', 'real_flip_rate'],
    'E_graph_flip': ['confidence', 'B_o', 'B_b', 'I_ob', 'real_flip_rate'],
    'F_graph_without_flip': ['confidence', 'B_o', 'B_b', 'I_ob']
}

predictions = testing[['img_filename', 'y', 'place', 'error']].copy()
results = []
for name, cols in feature_sets.items():
    detector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED))
    detector.fit(validation[cols], validation['error'].astype(int))
    scores = detector.predict_proba(testing[cols])[:, 1]
    predictions['risk_' + name] = scores
    results.append({'detector': name, 'features': ', '.join(cols), 'n': len(testing), 'AUROC': roc_auc_score(testing['error'], scores), 'AUPRC': average_precision_score(testing['error'], scores)})
metrics = pd.DataFrame(results)
metrics.to_csv(OUT / 'final_detector_metrics.csv', index=False)
predictions.to_csv(OUT / 'final_detector_predictions.csv', index=False)
print('\nFINAL A-E ABLATION (SAME RETAINED TEST IMAGES)')
print(metrics.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

comparisons = [
    ('E_minus_D_interaction', 'E_graph_flip', 'D_contributions_flip'),
    ('E_minus_C_graph', 'E_graph_flip', 'C_confidence_flip'),
    ('E_minus_B_graph_confidence', 'E_graph_flip', 'B_flip_only'),
    ('E_minus_F_real_flip', 'E_graph_flip', 'F_graph_without_flip'),
    ('C_minus_A_invariance', 'C_confidence_flip', 'A_confidence')
]
y = predictions['error'].to_numpy(dtype=int)
indices = []
for iteration in range(N_BOOTSTRAP):
    sample = rng.integers(0, len(y), size=len(y))
    if len(np.unique(y[sample])) == 2:
        indices.append(sample)
boot_rows = []
for title, new, old in comparisons:
    a = predictions['risk_' + new].to_numpy(dtype=float)
    b = predictions['risk_' + old].to_numpy(dtype=float)
    for metric_name, metric_fn in (('AUROC', roc_auc_score), ('AUPRC', average_precision_score)):
        observed = metric_fn(y, a) - metric_fn(y, b)
        samples = []
        for sample in indices:
            samples.append(metric_fn(y[sample], a[sample]) - metric_fn(y[sample], b[sample]))
        low, high = np.percentile(samples, [2.5, 97.5])
        boot_rows.append({'comparison': title, 'metric': metric_name, 'difference': observed, 'CI_95_lower': low, 'CI_95_upper': high})
bootstrap = pd.DataFrame(boot_rows)
bootstrap.to_csv(OUT / 'final_paired_bootstrap.csv', index=False)
print('\nPAIRED BOOTSTRAP (FIXED CNN, EXPLORATORY TEST)')
print(bootstrap.to_string(index=False, float_format=lambda x: f'{x:+.6f}'))

subgroup_rows = []
for (bird, background), group in predictions.groupby(['y', 'place'], sort=True):
    for detector in feature_sets:
        record = {'bird': int(bird), 'background': int(background), 'detector': detector, 'n': len(group), 'n_errors': int(group['error'].sum()), 'error_rate': float(group['error'].mean())}
        if group['error'].nunique() == 2:
            record['AUROC'] = roc_auc_score(group['error'], group['risk_' + detector])
            record['AUPRC'] = average_precision_score(group['error'], group['risk_' + detector])
        else:
            record['AUROC'] = np.nan
            record['AUPRC'] = np.nan
        subgroup_rows.append(record)
subgroups = pd.DataFrame(subgroup_rows)
subgroups.to_csv(OUT / 'final_subgroup_metrics.csv', index=False)
print('\nSUBGROUP RESULTS')
print(subgroups.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
macro = subgroups.groupby('detector', as_index=False).agg(macro_AUROC=('AUROC', 'mean'), macro_AUPRC=('AUPRC', 'mean'), valid_groups=('AUROC', 'count'))
macro.to_csv(OUT / 'final_macro_metrics.csv', index=False)
print('\nMACRO SUBGROUP METRICS (DESCRIPTIVE; AUPRC DEPENDS ON SUBGROUP PREVALENCE)')
print(macro.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

feature_rows = []
for (bird, background, error), group in testing.groupby(['y', 'place', 'error'], sort=True):
    feature_rows.append({'bird': int(bird), 'background': int(background), 'error': int(error), 'n': len(group), 'flip_mean': group['real_flip_rate'].mean(), 'flip_median': group['real_flip_rate'].median(), 'interaction_mean': group['I_ob'].mean(), 'bird_contribution_mean': group['B_o'].mean(), 'background_contribution_mean': group['B_b'].mean()})
feature_stats = pd.DataFrame(feature_rows)
feature_stats.to_csv(OUT / 'final_feature_by_group_and_error.csv', index=False)
print('\nFEATURES BY GROUP AND CORRECTNESS')
print(feature_stats.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

root = Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird')
meta_path = root / 'metadata.csv'
if not meta_path.exists():
    candidates = []
    for path in Path('/kaggle/input').rglob('metadata.csv'):
        if 'waterbird' in str(path).lower():
            candidates.append(path)
    if candidates:
        meta_path = candidates[0]
if meta_path.exists():
    metadata = pd.read_csv(meta_path)
    old_path = SOURCE / 'test_interventions.csv'
    if old_path.exists():
        old = pd.read_csv(old_path)
        if 'error' in old.columns:
            all_test = old[['img_filename', 'error']].merge(metadata.loc[metadata['split'] == 2, ['img_filename', 'y', 'place']], on='img_filename', how='inner', validate='one_to_one')
            all_test['retained'] = all_test['img_filename'].isin(testing['img_filename'])
            retention = all_test.groupby(['y', 'place', 'retained'], as_index=False).agg(n=('img_filename', 'size'), errors=('error', 'sum'), error_rate=('error', 'mean'))
            retention.to_csv(OUT / 'final_retention_audit.csv', index=False)
            print('\nRETAINED VS EXCLUDED (ORIGINAL TEST)')
            print(retention.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            print('Original:', len(all_test), '| retained:', int(all_test['retained'].sum()), '| excluded:', int((~all_test['retained']).sum()))
        else:
            print('\nRETENTION AUDIT: original intervention CSV has no error column.')
    else:
        print('\nRETENTION AUDIT: missing', old_path)
else:
    print('\nRETENTION AUDIT: metadata unavailable.')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, key in zip(axes, ('AUROC', 'AUPRC')):
    ax.bar(metrics['detector'], metrics[key])
    ax.set_ylim(0, 1)
    ax.set_title('Error detection ' + key)
    ax.tick_params(axis='x', labelrotation=35)
fig.tight_layout()
fig.savefig(OUT / 'final_ablation.png', dpi=180)
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, detector in zip(axes, ('C_confidence_flip', 'E_graph_flip')):
    for bird in (0, 1):
        for background in (0, 1):
            group = predictions[(predictions['y'] == bird) & (predictions['place'] == background)]
            if group['error'].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(group['error'], group['risk_' + detector])
            ax.plot(fpr, tpr, label=f'{bird}/{background}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set_title(detector)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(title='Bird/background')
fig.tight_layout()
fig.savefig(OUT / 'final_subgroup_roc.png', dpi=180)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for bird in (0, 1):
    for background in (0, 1):
        group = testing[(testing['y'] == bird) & (testing['place'] == background)]
        ax.hist(group['real_flip_rate'], bins=np.linspace(0, 1, 8), alpha=0.45, density=True, label=f'{bird}/{background}')
ax.set_title('Real background flip rate by subgroup')
ax.set_xlabel('Flip rate')
ax.set_ylabel('Density')
ax.legend(title='Bird/background')
fig.tight_layout()
fig.savefig(OUT / 'final_flip_distribution_by_group.png', dpi=180)
plt.show()
plt.close(fig)

for split in ('validation', 'test'):
    path = CACHE / (split + '_swap_examples.png')
    if path.exists():
        print('VISUAL INTERVENTION EXAMPLES:', path)
    else:
        print('Visual examples not found for', split, '- no image interventions are recomputed.')

print('\nFINAL AUDIT COMPLETE. Outputs:', OUT)
print('IMPORTANT: This is an exploratory audit on a repeatedly examined test split, not a new independent confirmation.')
print('Approximate DeepLab masks and reconstructed donor backgrounds remain a limitation.')
print('Visual examples require human inspection; this script cannot certify that swaps are artifact-free.')