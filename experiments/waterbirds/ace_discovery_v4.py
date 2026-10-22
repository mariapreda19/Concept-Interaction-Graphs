import os, glob, json, random, warnings, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

SEED = 42
N_TRAIN = 180
N_VAL = 80
IMAGE_SIZE = 224
SLIC_COUNTS = [15, 50, 80]
MAX_REGIONS_PER_SCALE = {15: 12, 50: 18, 80: 22}
MIN_REGION_AREA = 0.003
MAX_REGION_AREA = 0.65
FEATURE_BATCH = 40
K_PER_CLASS_SCALE = 10
TOP_PROTOTYPES = 25
MIN_TRAIN_IMAGES = 6
MIN_KEEP_REGIONS = 12
MAX_CANDIDATE_CONCEPTS = 20
MAX_VAL_IMAGES_FOR_GRAPH = 50
MAX_PAIRS_PER_IMAGE = 8
MAX_GRAPH_SAMPLES_PER_PAIR = 30
MAX_PROTOTYPE_PANELS = 20
USE_CLIP = True
CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'
CANDIDATE_LABELS = [
    'bird head', 'bird beak', 'bird wing', 'bird tail', 'bird feathers',
    'bird body', 'water surface', 'water waves', 'reeds', 'tree branches',
    'green vegetation', 'blue sky', 'clouds', 'sand', 'rocks',
    'grass', 'tree leaves', 'brown earth', 'bird silhouette'
]
LABEL_MIN_COSINE = 0.22
LABEL_MIN_GAP = 0.025
OUT = Path('/kaggle/working/waterbirds_ace_semantic_graph_v4')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda': torch.cuda.manual_seed_all(SEED)
print('DEVICE:', DEVICE, 'OUTPUT:', OUT)

roots = [Path('/kaggle/input/datasets/bahardibaie/waterbird/waterbird'),
         Path('/kaggle/input/waterbird/waterbird'), Path('/kaggle/input/waterbird')]
root = next((p for p in roots if (p/'metadata.csv').is_file()), None)
if root is None:
    found = [Path(x).parent for x in glob.glob('/kaggle/input/**/metadata.csv', recursive=True)
             if 'waterbird' in x.lower()]
    if not found: raise FileNotFoundError('Waterbirds metadata.csv was not found in /kaggle/input')
    root = found[0]
meta = pd.read_csv(root/'metadata.csv')
for field in ['img_filename', 'y', 'place', 'split']:
    if field not in meta: raise ValueError('Missing column: '+field)
def image_path(s):
    for p in [root/str(s), root/'dataset'/str(s)]:
        if p.is_file(): return str(p)
    raise FileNotFoundError('Image missing: '+str(s))
train = meta.loc[meta['split'] == 0].sample(frac=1, random_state=SEED).head(N_TRAIN).copy()
val = meta.loc[meta['split'] == 1].sample(frac=1, random_state=SEED).head(N_VAL).copy()
train['partition'] = 'train'; val['partition'] = 'validation'
subset = pd.concat([train,val], ignore_index=True)
subset['path'] = subset.img_filename.map(image_path)
subset['sample_id'] = np.arange(len(subset))
subset.to_csv(OUT/'selected_images.csv', index=False)
print('Selected:', len(train), 'TRAIN /', len(val), 'VALIDATION. TEST NOT USED.')

ckpts = [Path('/kaggle/working/waterbirds_human_concepts_v2/resnet18_waterbirds_best.pt'),
         Path('/kaggle/working/waterbirds_human_concepts/resnet18_waterbirds_best.pt')]
ckpts += [Path(p) for p in glob.glob('/kaggle/input/**/resnet18_waterbirds_best.pt',recursive=True)]
checkpoint = next((p for p in ckpts if p.is_file()), None)
if checkpoint is None:
    raise FileNotFoundError('Attach the previously trained resnet18_waterbirds_best.pt. No new CNN will be silently substituted.')
cnn = models.resnet18(weights=None)
cnn.fc = torch.nn.Linear(cnn.fc.in_features,2)
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
if isinstance(state,dict):
    for key in ['model_state_dict','state_dict','model']:
        if key in state and isinstance(state[key],dict):
            state = state[key]
            break
if not isinstance(state,dict): raise ValueError('Unexpected checkpoint format')
state = {k.removeprefix('module.'):v for k,v in state.items()}
cnn.load_state_dict(state,strict=True)
cnn.to(DEVICE).eval()
for p in cnn.parameters(): p.requires_grad_(False)
print('Fixed classifier:',checkpoint)

normalizer = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
preprocess = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),normalizer])
features_cache=[]
def hook_fn(module,inputs,output):
    features_cache.append(output.detach().flatten(1))
hook = cnn.avgpool.register_forward_hook(hook_fn)

def cnn_forward(pils,return_features=True):
    if not pils: return np.empty((0,2)),np.empty((0,512))
    logits_all=[]; vec_all=[]
    for i in range(0,len(pils),FEATURE_BATCH):
        batch = torch.stack([preprocess(p) for p in pils[i:i+FEATURE_BATCH]]).to(DEVICE)
        features_cache.clear()
        with torch.inference_mode(): logits = cnn(batch)
        logits_all.append(logits.float().cpu().numpy())
        if return_features: vec_all.append(features_cache[-1].float().cpu().numpy())
    arr = np.concatenate(logits_all)
    vec = np.concatenate(vec_all) if return_features else np.empty((len(pils),0))
    return arr,vec

def open_image(path):
    with Image.open(path) as x: return x.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE))

train_patches=[]; tr_rows=[]; train_region_masks={}
for row in tqdm(subset.iloc[:len(train)].itertuples(index=False), total=len(train),desc='Train SLIC regions'):
    pil=open_image(row.path)
    rgb=np.asarray(pil)
    for scale in SLIC_COUNTS:
        segments = slic(rgb, n_segments=scale, compactness=10, sigma=1, start_label=0, channel_axis=-1)
        possible=[]
        for idx in np.unique(segments):
            mask=segments==idx
            area=float(mask.mean())
            if not (MIN_REGION_AREA<=area<=MAX_REGION_AREA): continue
            ys,xs=np.where(mask)
            x0,x1=int(xs.min()),int(xs.max())+1
            y0,y1=int(ys.min()),int(ys.max())+1
            if (x1-x0)<5 or (y1-y0)<5:continue
            possible.append((idx,mask,area,(x0,y0,x1,y1)))
        generator=np.random.default_rng(SEED+int(row.sample_id)*101+scale)
        if len(possible)>MAX_REGIONS_PER_SCALE[scale]:
            indices=generator.choice(len(possible),size=MAX_REGIONS_PER_SCALE[scale],replace=False)
            possible=[possible[i] for i in sorted(indices)]
        for idx,mask,area,box in possible:
            x0,y0,x1,y1=box
            canvas=np.full(rgb.shape,117,dtype=np.uint8)
            canvas[mask]=rgb[mask]
            patch=Image.fromarray(canvas[y0:y1,x0:x1]).resize((IMAGE_SIZE,IMAGE_SIZE))
            rid=len(tr_rows)
            train_patches.append(patch)
            tr_rows.append(dict(region_id=rid,sample_id=int(row.sample_id),path=row.path,
                                y=int(row.y),place=int(row.place),scale=int(scale),
                                segment=int(idx),area=area,box=json.dumps(box)))
            train_region_masks[rid]=mask
print('Train candidate regions:',len(tr_rows))
if len(tr_rows)<100: raise RuntimeError('Too few valid training regions; inspect segmentation settings')

all_vectors=[]
for i in tqdm(range(0,len(train_patches),FEATURE_BATCH),desc='CNN descriptors'):
    _,vectors=cnn_forward(train_patches[i:i+FEATURE_BATCH]);all_vectors.append(vectors)
X=normalize(np.concatenate(all_vectors,axis=0).astype(np.float32))
tr_regions=pd.DataFrame(tr_rows)
np.save(OUT/'train_region_vectors.npy',X)
train_patches=[]; all_vectors=[]
if DEVICE=='cuda': torch.cuda.empty_cache()

cluster_records=[]; concept_info=[]; centroids=[]
for (y,scale),part in tr_regions.groupby(['y','scale']):
    ids=part.region_id.to_numpy(dtype=np.int64)
    if len(ids)<max(2*K_PER_CLASS_SCALE,40):continue
    k=min(K_PER_CLASS_SCALE,max(2,len(ids)//25))
    km=MiniBatchKMeans(n_clusters=k,random_state=SEED,batch_size=min(512,len(ids)),n_init=5,max_iter=200)
    labels=km.fit_predict(X[ids]);cent=normalize(km.cluster_centers_)
    for j in range(k):
        members=ids[labels==j]
        if len(members)<MIN_KEEP_REGIONS: continue
        distances=1-X[members]@cent[j]
        order=np.argsort(distances)
        ranked=members[order]
        prototypes=ranked[:TOP_PROTOTYPES]
        n_images=int(tr_regions.loc[members,'sample_id'].nunique())
        if n_images<MIN_TRAIN_IMAGES:continue
        compact=float(np.median(distances[order[:len(prototypes)]]))
        spread=float(np.median(distances))
        cid='C'+str(len(concept_info)).zfill(3)
        centroids.append(cent[j]);concept_info.append(dict(concept_id=cid,source_class=int(y),
                  scale=int(scale),member_regions=len(members),distinct_train_images=n_images,
                  prototype_compactness=compact,cluster_spread=spread,semantic_label='',
                  semantic_similarity=np.nan,semantic_margin=np.nan,semantic_validated=False))
        for rank,rid in enumerate(prototypes):
            cluster_records.append(dict(concept_id=cid,region_id=int(rid),rank=int(rank),
                                        cosine_distance=float(distances[order[rank]])))
concepts=pd.DataFrame(concept_info)
proto=pd.DataFrame(cluster_records)
if len(concepts)==0:raise RuntimeError('No coherent candidates survived. Decrease MIN_KEEP_REGIONS / MIN_TRAIN_IMAGES.')

concepts['ranking']=concepts.distinct_train_images/(0.1+concepts.cluster_spread)
concepts=concepts.sort_values('ranking',ascending=False).head(MAX_CANDIDATE_CONCEPTS).reset_index(drop=True)
chosen_ids=set(concepts.concept_id)
proto=proto[proto.concept_id.isin(chosen_ids)].copy()
center_lookup={info['concept_id']:centroids[int(info['concept_id'][1:])] for info in concept_info}
tr_regions.to_csv(OUT/'train_region_assignments.csv',index=False)
proto.to_csv(OUT/'concept_prototypes.csv',index=False)

def crop_masked(row):
    image=open_image(row.path)
    rgb=np.asarray(image)
    seg=slic(rgb,n_segments=int(row.scale),compactness=10,sigma=1,start_label=0,channel_axis=-1)
    mask=seg==int(row.segment)
    box=json.loads(row.box)
    x0,y0,x1,y1=box
    if not mask.any():return Image.fromarray(rgb[y0:y1,x0:x1])
    canvas=np.full_like(rgb,117)
    canvas[mask]=rgb[mask]
    return Image.fromarray(canvas[y0:y1,x0:x1])

for record in tqdm(concepts.itertuples(index=False),total=len(concepts),desc='Concept contact sheets'):
    pc=proto[proto.concept_id==record.concept_id].sort_values('rank').head(16)
    fig,axes=plt.subplots(4,4,figsize=(10,10))
    for ax in axes.ravel():ax.axis('off')
    for ax,pr in zip(axes.ravel(),pc.itertuples(index=False)):
        r=tr_regions.iloc[int(pr.region_id)]
        ax.imshow(crop_masked(r));ax.set_title('img '+str(int(r.sample_id)),fontsize=8)
    fig.suptitle(record.concept_id+' | class '+str(record.source_class)+' | scale '+str(record.scale))
    fig.tight_layout();fig.savefig(OUT/(record.concept_id+'_prototypes.png'),dpi=120)
    plt.close(fig)

clip_available=False
clip_reason=''
if USE_CLIP:
    try:
        import sys, importlib
        if 'transformers' not in sys.modules:
            hf=Path('/kaggle/working/hf_v3')
            if hf.is_dir() and str(hf) not in sys.path:sys.path.insert(0,str(hf))
        from transformers import CLIPModel, CLIPProcessor
        clipproc=CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        clipmodel=CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
        for p in clipmodel.parameters():p.requires_grad_(False)
        clip_available=True
    except Exception as exc:
        clip_reason=str(exc)
        print('CLIP labeling unavailable; manual labeling remains possible:',type(exc).__name__,clip_reason[:250])
if clip_available:
    descriptions=['a photograph of '+label for label in CANDIDATE_LABELS]
    t=clipproc(text=descriptions,return_tensors='pt',padding=True).to(DEVICE)
    with torch.inference_mode():
        text_embedding=clipmodel.get_text_features(**t)
    if hasattr(text_embedding,'pooler_output'):text_embedding=text_embedding.pooler_output
    text_embedding=torch.nn.functional.normalize(text_embedding,dim=-1)
    for i,record in tqdm(enumerate(concepts.itertuples(index=False)),total=len(concepts),desc='CLIP naming'):
        pc=proto[proto.concept_id==record.concept_id].sort_values('rank').head(10)
        images=[crop_masked(tr_regions.iloc[int(p.region_id)]).convert('RGB') for p in pc.itertuples(index=False)]
        if not images:continue
        batch=clipproc(images=images,return_tensors='pt').to(DEVICE)
        with torch.inference_mode():emb=clipmodel.get_image_features(**batch)
        if hasattr(emb,'pooler_output'):emb=emb.pooler_output
        emb=torch.nn.functional.normalize(emb,dim=-1)
        similarities=(emb@text_embedding.T).float().cpu().numpy().mean(axis=0)
        top=np.argsort(-similarities)
        best=float(similarities[top[0]])
        gap=float(best-similarities[top[1]])
        concepts.loc[i,'semantic_similarity']=best
        concepts.loc[i,'semantic_margin']=gap
        if best>=LABEL_MIN_COSINE and gap>=LABEL_MIN_GAP:
            concepts.loc[i,'semantic_label']=CANDIDATE_LABELS[top[0]]
    del clipmodel
    if DEVICE=='cuda':torch.cuda.empty_cache()
concepts['display_label']=concepts.semantic_label
concepts.loc[concepts.display_label=='','display_label']='unlabeled'
concepts.to_csv(OUT/'concept_summary.csv',index=False)
review=concepts[['concept_id','source_class','scale','distinct_train_images','semantic_label',
                 'semantic_similarity','semantic_margin']].copy()
review['human_label']='';review['coherent_0_to_2']='';review['meaningful_0_to_2']='';review['human_notes']=''
review.to_csv(OUT/'human_concept_review_to_fill.csv',index=False)
print('\nCANDIDATES (all labels provisional):')
print(concepts[['concept_id','source_class','scale','distinct_train_images','display_label','semantic_similarity','semantic_margin']].to_string(index=False))

chosen_centers=np.stack([center_lookup[c] for c in concepts.concept_id])
concept_scales=concepts.scale.to_numpy()
val_records=[]
val_masks=defaultdict(dict)
for row in tqdm(subset.iloc[len(train):].itertuples(index=False),total=len(val),desc='Validation concept localization'):
    pil=open_image(row.path);arr=np.asarray(pil)
    patches=[];mask_infos=[]
    for scale in SLIC_COUNTS:
        segmentation=slic(arr,n_segments=scale,compactness=10,sigma=1,start_label=0,channel_axis=-1)
        options=[]
        for sid in np.unique(segmentation):
            mask=segmentation==sid;area=float(mask.mean())
            if not (MIN_REGION_AREA<=area<=MAX_REGION_AREA):continue
            ys,xs=np.where(mask);x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
            if (x1-x0)<5 or (y1-y0)<5:continue
            options.append((int(sid),mask,area,(x0,y0,x1,y1)))
        generator=np.random.default_rng(SEED+int(row.sample_id)*101+scale)
        if len(options)>MAX_REGIONS_PER_SCALE[scale]:
            idx=generator.choice(len(options),size=MAX_REGIONS_PER_SCALE[scale],replace=False)
            options=[options[i] for i in sorted(idx)]
        for sid,mask,area,box in options:
            x0,y0,x1,y1=box
            canvas=np.full_like(arr,117);canvas[mask]=arr[mask]
            patches.append(Image.fromarray(canvas[y0:y1,x0:x1]).resize((IMAGE_SIZE,IMAGE_SIZE)))
            mask_infos.append((scale,mask,area,box))
    if not patches:continue
    _,emb=cnn_forward(patches)
    emb=normalize(emb)
    best={}
    for j,(scale,mask,area,box) in enumerate(mask_infos):
        candidates=np.flatnonzero(concept_scales==scale)
        if len(candidates)==0:continue
        similarity=chosen_centers[candidates]@emb[j]
        win=int(np.argmax(similarity)); idx=int(candidates[win]); score=float(similarity[win])
        cid=str(concepts.iloc[idx].concept_id)
        if cid not in best or score>best[cid][0]:best[cid]=(score,mask,area,box)
    for cid,(score,mask,area,box) in best.items():
        val_masks[int(row.sample_id)][cid]=mask
        val_records.append(dict(sample_id=int(row.sample_id),concept_id=cid,
                       y=int(row.y),place=int(row.place),similarity=score,area=area,
                       box=json.dumps(box),semantic_label=str(concepts.loc[concepts.concept_id==cid,'display_label'].iloc[0])))
    if best:
        ids=list(best)
        np.savez_compressed(OUT/('validation_masks_'+str(int(row.sample_id)).zfill(4)+'.npz'),
                            masks=np.stack([best[c][1] for c in ids]).astype('uint8'),
                            concept_ids=np.array(ids))
val_assign=pd.DataFrame(val_records)
val_assign.to_csv(OUT/'validation_concept_assignments.csv',index=False)
if len(val_assign)==0:raise RuntimeError('Validation region assignment was empty')

def relation_between(a,b):
    ya,xa=np.where(a);yb,xb=np.where(b)
    if len(xa)==0 or len(xb)==0:return 'unknown'
    ca=np.array([xa.mean(),ya.mean()]);cb=np.array([xb.mean(),yb.mean()]);delta=cb-ca
    diag=math.sqrt(a.shape[0]**2+a.shape[1]**2)
    d=float(np.linalg.norm(delta))/diag
    if d<0.12:return 'near'
    if abs(delta[1])>abs(delta[0])*1.2:return 'above' if delta[1]>0 else 'below'
    if abs(delta[0])>abs(delta[1])*1.2:return 'left_of' if delta[0]>0 else 'right_of'
    return 'near' if d<0.25 else 'separated'

def masked(pil,mask):
    a=np.asarray(pil).copy()
    a[mask]=np.array([123,116,103],dtype=np.uint8)
    return Image.fromarray(a)

def class_margin(logits):
    return float(logits[1]-logits[0])

pair_rows=[]
validation_ids=list(subset.iloc[len(train):].sample(frac=1,random_state=SEED).head(MAX_VAL_IMAGES_FOR_GRAPH).sample_id.astype(int))
for sid in tqdm(validation_ids,desc='Pair interventions'):
    masks=val_masks.get(sid,{})
    if len(masks)<2:continue
    pil=open_image(subset.iloc[sid].path)
    base=cnn_forward([pil],False)[0][0]
    ids=list(masks.keys())
    options=[]
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            a=masks[ids[i]];b=masks[ids[j]]
            iou=float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
            if iou>0.10:continue
            options.append((ids[i],ids[j],iou))
    rng=np.random.default_rng(SEED+sid)
    if len(options)>MAX_PAIRS_PER_IMAGE:
        indices=rng.choice(len(options),MAX_PAIRS_PER_IMAGE,replace=False)
        options=[options[i] for i in indices]
    if not options:continue
    single_ids=list(set(x for a,b,_ in options for x in [a,b]))
    images=[masked(pil,masks[c]) for c in single_ids]
    images += [masked(pil,np.logical_or(masks[a],masks[b])) for a,b,_ in options]
    outputs=cnn_forward(images,False)[0]
    margins={c:class_margin(logits) for c,logits in zip(single_ids,outputs[:len(single_ids)])}
    base_m=class_margin(base)
    for (a,b,iou),out in zip(options,outputs[len(single_ids):]):
        interaction=base_m-margins[a]-margins[b]+class_margin(out)
        pair_rows.append(dict(sample_id=sid,concept_a=a,concept_b=b,
                              label_a=str(concepts.loc[concepts.concept_id==a,'display_label'].iloc[0]),
                              label_b=str(concepts.loc[concepts.concept_id==b,'display_label'].iloc[0]),
                              spatial_relation=relation_between(masks[a],masks[b]),
                              mask_iou=iou,base_margin=base_m,
                              delta_a=base_m-margins[a],delta_b=base_m-margins[b],
                              interaction=float(interaction),abs_interaction=float(abs(interaction)),
                              y=int(subset.iloc[sid].y),place=int(subset.iloc[sid].place)))
pairs=pd.DataFrame(pair_rows)
pairs.to_csv(OUT/'pair_interactions_individual.csv',index=False)
if len(pairs):
    pairs['edge_id']=pairs.apply(lambda r:'|'.join(sorted([r.concept_a,r.concept_b])),axis=1)
    graph=pairs.groupby('edge_id',as_index=False).agg(n=('interaction','size'),
                mean_interaction=('interaction','mean'),mean_abs_interaction=('abs_interaction','mean'),
                std_interaction=('interaction','std'),mean_mask_iou=('mask_iou','mean'))
    graph=graph.sort_values('mean_abs_interaction',ascending=False)
    graph[['concept_a','concept_b']]=graph.edge_id.str.split('|',expand=True)
    graph['semantic_a']=graph.concept_a.map(concepts.set_index('concept_id').display_label)
    graph['semantic_b']=graph.concept_b.map(concepts.set_index('concept_id').display_label)
    graph.to_csv(OUT/'graph_edges.csv',index=False)
    spatial=pairs.groupby(['concept_a','concept_b','spatial_relation'],as_index=False).size()
    spatial.rename(columns={'size':'count'},inplace=True)
    spatial.to_csv(OUT/'spatial_relations.csv',index=False)
    print('\nTOP OBSERVED EDGES (hypotheses, not validated causal relationships):')
    print(graph[['concept_a','concept_b','n','mean_interaction','mean_abs_interaction']].head(15).to_string(index=False))
else:
    graph=pd.DataFrame(columns=['concept_a','concept_b','n','mean_interaction','mean_abs_interaction'])
    graph.to_csv(OUT/'graph_edges.csv',index=False)
    print('No eligible non-overlapping pairs; inspect masks before drawing conclusions.')

for sid in validation_ids[:8]:
    masks=val_masks.get(sid,{})
    if not masks:continue
    orig=np.asarray(open_image(subset.iloc[sid].path));n=min(6,len(masks))
    fig,axes=plt.subplots(1,n+1,figsize=(3*(n+1),3))
    axes[0].imshow(orig);axes[0].set_title('Original');axes[0].axis('off')
    for ax,(cid,mask) in zip(axes[1:],list(masks.items())[:n]):
        overlay=orig.copy();overlay[mask]=(0.55*overlay[mask]+0.45*np.array([255,50,20])).astype(np.uint8)
        ax.imshow(overlay)
        label=str(concepts.loc[concepts.concept_id==cid,'display_label'].iloc[0])
        ax.set_title(cid+' '+label,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/('validation_review_'+str(sid).zfill(4)+'.png'),dpi=110)
    plt.close(fig)

configuration=dict(seed=SEED,checkpoint=str(checkpoint),train_images=N_TRAIN,validation_images=N_VAL,
                   test_used=False,method='ACE-inspired SLIC ResNet18 descriptors + prototype filtering + optional CLIP labels',
                   n_candidate_concepts=int(len(concepts)),n_intervention_images=len(validation_ids),
                   n_interventions=int(len(pairs)),notes=[
                       'Not exact ACE: small class/scale KMeans, no TCAV implementation.',
                       'Provisional CLIP names are not validated; see human_concept_review_to_fill.csv.',
                       'Same-scale nearest-centroid matching has not been calibrated and can assign wrong concepts.',
                       'Spatial relations use centroid heuristics; not a verified NLP scene graph.',
                       'CNN interactions measure four-condition nonadditivity for color-masking perturbations.',
                       'No claims of semantic graph fidelity or improved error detection yet.',
                       'Test untouched; validation used to explore graph and must not be final unbiased evaluation.'
                   ])
(OUT/'config.json').write_text(json.dumps(configuration,indent=2))
hook.remove()
print('\nDONE:',OUT)
print('Send: concept_summary.csv, concept contact sheets, validation_review_*.png, graph_edges.csv, pair_interactions_individual.csv')
print('Complete human_concept_review_to_fill.csv before treating labels as verified.')
print('Save Version > Save & Run All to persist Kaggle outputs.')