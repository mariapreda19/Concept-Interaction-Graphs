from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
N_BOOTSTRAP = 1000
EPS = 1e-6
SOURCE = Path('/kaggle/working/waterbirds_real1_20epochs')
OUT = SOURCE / 'semantic_consistency'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

required = ['y', 'place', 'error', 'm', 'm_without_background', 'delta_o', 'delta_b', 'B_o', 'B_b', 'I_ob']
frames = {}
for split in ('validation', 'test'):
    path = SOURCE / (split + '_interventions.csv')
    if not path.exists():
        raise FileNotFoundError('Missing ' + str(path) + '. Restore the saved outputs from the 20-epoch experiment.')
    frame = pd.read_csv(path)
    missing = []
    for column in required:
        if column not in frame.columns:
            missing.append(column)
    if missing:
        raise ValueError(str(path) + ' is missing columns: ' + str(missing))
    frame['confidence'] = frame['m'].abs()
    frame['D_flip'] = ((frame['m'] > 0) != (frame['m_without_background'] > 0)).astype(int)
    frame['D_bg'] = frame['delta_b'].abs()
    denom = frame['delta_o'].abs() + frame['delta_b'].abs() + EPS
    frame['D_rel'] = frame['delta_b'].abs() / denom
    frame['D_int'] = frame['I_ob'].abs() / denom
    frame['D_sem_old'] = np.maximum(0, frame['B_b']) + np.maximum(0, frame['B_b'] - frame['B_o'])
    for column in ('D_flip', 'D_bg', 'D_rel', 'D_int'):
        if not np.isfinite(frame[column]).all():
            raise ValueError('Non-finite values in ' + column)
    if frame['error'].nunique() != 2:
        raise ValueError('Both correct and incorrect predictions are required in ' + split)
    frames[split] = frame

validation = frames['validation']
testing = frames['test']
base = ['confidence', 'B_o', 'B_b', 'I_ob']
semantic = ['D_flip', 'D_bg', 'D_rel', 'D_int']
feature_sets = {
    'confidence': ['confidence'],
    'contributions': ['confidence', 'B_o', 'B_b'],
    'cnn_graph': base,
    'old_semantic': base + ['D_sem_old'],
    'semantic_consistency': ['confidence'] + semantic,
    'full_consistency': base + semantic,
    'plus_flip': base + ['D_flip'],
    'plus_bg': base + ['D_bg'],
    'plus_rel': base + ['D_rel'],
    'plus_int': base + ['D_int'],
    'without_flip': base + ['D_bg', 'D_rel', 'D_int'],
    'without_bg': base + ['D_flip', 'D_rel', 'D_int'],
    'without_rel': base + ['D_flip', 'D_bg', 'D_int'],
    'without_int': base + ['D_flip', 'D_bg', 'D_rel']
}

predictions = testing[['y', 'place', 'error']].copy()
if 'img_filename' in testing.columns:
    predictions.insert(0, 'img_filename', testing['img_filename'])
metrics = []
for name, columns in feature_sets.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED))
    model.fit(validation[columns], validation['error'])
    risk = model.predict_proba(testing[columns])[:, 1]
    predictions['risk_' + name] = risk
    metrics.append({'detector': name, 'features': ', '.join(columns), 'AUROC': roc_auc_score(testing['error'], risk), 'AUPRC': average_precision_score(testing['error'], risk)})
metrics = pd.DataFrame(metrics)
print('\nSEMANTIC CONSISTENCY ABLATION')
print(metrics.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
metrics.to_csv(OUT / 'semantic_consistency_metrics.csv', index=False)
predictions.to_csv(OUT / 'semantic_consistency_predictions.csv', index=False)

comparisons = [
    ('interaction_added', 'cnn_graph', 'contributions'),
    ('old_semantics_added', 'old_semantic', 'cnn_graph'),
    ('new_semantics_added', 'full_consistency', 'cnn_graph'),
    ('new_vs_old', 'full_consistency', 'old_semantic'),
    ('flip_added', 'plus_flip', 'cnn_graph'),
    ('bg_added', 'plus_bg', 'cnn_graph'),
    ('rel_added', 'plus_rel', 'cnn_graph'),
    ('int_added', 'plus_int', 'cnn_graph')
]
y = testing['error'].to_numpy(dtype=int)
bootstrap_indices = []
for iteration in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(y), len(y))
    if np.unique(y[idx]).size == 2:
        bootstrap_indices.append(idx)

comparison_rows = []
for label, added, baseline in comparisons:
    new_scores = predictions['risk_' + added].to_numpy()
    old_scores = predictions['risk_' + baseline].to_numpy()
    for metric_name, metric_fn in (('AUROC', roc_auc_score), ('AUPRC', average_precision_score)):
        difference = metric_fn(y, new_scores) - metric_fn(y, old_scores)
        boot = []
        for idx in bootstrap_indices:
            boot.append(metric_fn(y[idx], new_scores[idx]) - metric_fn(y[idx], old_scores[idx]))
        ci = np.percentile(boot, [2.5, 97.5])
        comparison_rows.append({'comparison': label, 'metric': metric_name, 'difference': difference, 'CI_95_lower': ci[0], 'CI_95_upper': ci[1]})
comparisons_df = pd.DataFrame(comparison_rows)
print('\nPAIRED BOOTSTRAP (FIXED CNN AND FIXED TEST SET)')
print(comparisons_df.to_string(index=False, float_format=lambda x: f'{x:+.6f}'))
comparisons_df.to_csv(OUT / 'semantic_consistency_bootstrap.csv', index=False)

rows = []
for name in semantic + ['D_sem_old', 'I_ob']:
    for error_value in (0, 1):
        subset = testing[testing['error'] == error_value][name]
        rows.append({'feature': name, 'error': error_value, 'n': len(subset), 'mean': subset.mean(), 'median': subset.median(), 'std': subset.std()})
summary = pd.DataFrame(rows)
print('\nFEATURE DISTRIBUTIONS: CORRECT (0) VS INCORRECT (1)')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
summary.to_csv(OUT / 'semantic_feature_distributions.csv', index=False)

subgroup_rows = []
for (bird, background), group in predictions.groupby(['y', 'place']):
    if group['error'].nunique() < 2:
        continue
    for name in ('confidence', 'cnn_graph', 'old_semantic', 'semantic_consistency', 'full_consistency'):
        risk = group['risk_' + name]
        subgroup_rows.append({'bird': int(bird), 'background': int(background), 'detector': name, 'n': len(group), 'error_rate': group['error'].mean(), 'AUROC': roc_auc_score(group['error'], risk), 'AUPRC': average_precision_score(group['error'], risk)})
subgroups = pd.DataFrame(subgroup_rows)
print('\nSUBGROUP RESULTS')
print(subgroups.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
subgroups.to_csv(OUT / 'semantic_consistency_subgroups.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
selected = metrics[metrics['detector'].isin(['confidence', 'contributions', 'cnn_graph', 'old_semantic', 'semantic_consistency', 'full_consistency'])]
for ax, metric in zip(axes, ('AUROC', 'AUPRC')):
    ax.bar(selected['detector'], selected[metric])
    ax.set_title('Error detection ' + metric)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelrotation=35)
fig.tight_layout()
fig.savefig(OUT / 'semantic_consistency_ablation.png', dpi=180)
plt.show()
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, feature in zip(axes.flat, semantic):
    correct = testing.loc[testing['error'] == 0, feature].to_numpy()
    incorrect = testing.loc[testing['error'] == 1, feature].to_numpy()
    ax.boxplot([correct, incorrect], tick_labels=['Correct', 'Incorrect'], showfliers=False)
    ax.set_title(feature)
fig.tight_layout()
fig.savefig(OUT / 'semantic_consistency_distributions.png', dpi=180)
plt.show()
plt.close(fig)

print('\nSaved to:', OUT)
print('IMPORTANT: This tests semantic invariance encoded using existing CNN interventions, not an independent NLP model.')
print('Existing interventions use approximate bird masks and a neutral reference background.')
print('Bootstrap intervals cover test-image sampling for one fixed CNN, not variation across CNN training seeds.')