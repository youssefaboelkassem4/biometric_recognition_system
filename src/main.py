import os
from preprocessing import load_face_dataset
from feature_extraction import EigenfaceExtractor, LBPExtractor, HOGExtractor
from build_features import build_gallery_features, build_probe_features
from matching import identify_subject, compute_all_scores, compute_euclidean, compute_cosine
from fusion import FusedExtractor
from evaluation import fuse_genuine_impostor, run_full_evaluation

# ── Step 1: Load data ──────────────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading data")
print("=" * 50)
gallery, probes = load_face_dataset()
print(f"Gallery subjects: {len(gallery)}")   # should be 50
print(f"Probe subjects:   {len(probes)}")    # should be 50

# ── Step 2: Fit & save PCA (Eigenfaces) ───────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Fitting Eigenfaces (PCA)")
print("=" * 50)

os.makedirs("models", exist_ok=True)
MODEL_PATH = 'models/pca_model.pkl'

if os.path.exists(MODEL_PATH):
    ef = EigenfaceExtractor.load(MODEL_PATH)
else:
    ef = EigenfaceExtractor(n_components=75)
    ef.fit(gallery)
    ef.save(MODEL_PATH)

# ── Step 3: Build individual feature sets ─────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Extracting features (all 3 methods)")
print("=" * 50)

lbp_ext = LBPExtractor()
hog_ext = HOGExtractor()

pca_gallery = build_gallery_features(gallery, ef.extract)
pca_probes  = build_probe_features(probes,   ef.extract)
print(f"PCA  vectors ready — shape: {pca_gallery['s01'].shape}")   # (75,)

lbp_gallery = build_gallery_features(gallery, lbp_ext.extract)
lbp_probes  = build_probe_features(probes,   lbp_ext.extract)
print(f"LBP  vectors ready — shape: {lbp_gallery['s01'].shape}")   # (1024,)

hog_gallery = build_gallery_features(gallery, hog_ext.extract)
hog_probes  = build_probe_features(probes,   hog_ext.extract)
print(f"HOG  vectors ready — shape: {hog_gallery['s01'].shape}")   # (8100,)

# ── Step 4: Build fused feature set ───────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Fused features (PCA + LBP + HOG)")
print("=" * 50)

FUSED_PATH = 'models/fused_model.pkl'

if os.path.exists(FUSED_PATH):
    fused_ext = FusedExtractor.load(FUSED_PATH)
else:
    fused_ext = FusedExtractor(
        EigenfaceExtractor(n_components=75),
        LBPExtractor(),
        HOGExtractor()
    )
    fused_ext.fit(gallery)
    fused_ext.save(FUSED_PATH)

fused_gallery = build_gallery_features(gallery, fused_ext.extract)
fused_probes  = build_probe_features(probes,   fused_ext.extract)
print(f"Fused vectors ready — shape: {fused_gallery['s01'].shape}")

# ── Step 5: Matching — all methods × both metrics ─────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Matching — Rank-1 Accuracy")
print("=" * 50)

methods = {
    'PCA (Eigenfaces)': (pca_gallery,   pca_probes),
    'LBP':              (lbp_gallery,   lbp_probes),
    'HOG':              (hog_gallery,   hog_probes),
    'Fused':            (fused_gallery, fused_probes),
}

all_scores = {}

print(f"\n{'Method':<20} {'Euclidean Rank-1':>17} {'Cosine Rank-1':>14}")
print("-" * 55)

for method_name, (gal, prb) in methods.items():
    total = len(prb)

    euc_correct = sum(
        1 for pid, pvec in prb.items()
        if identify_subject(pvec, gal, metric='euclidean')[0] == pid
    )
    cos_correct = sum(
        1 for pid, pvec in prb.items()
        if identify_subject(pvec, gal, metric='cosine')[0] == pid
    )
    euc_rank1 = euc_correct / total * 100
    cos_rank1 = cos_correct / total * 100

    print(f"{method_name:<20} {euc_rank1:>16.2f}% {cos_rank1:>13.2f}%")

    genuine, impostor = compute_all_scores(gal, prb, metric='euclidean')
    all_scores[method_name] = {
        'genuine':   genuine,
        'impostor':  impostor,
        'rank1_euc': euc_rank1,
        'rank1_cos': cos_rank1,
    }


print("\n" + "=" * 50)
print("STEP 6: Full Evaluation — Metrics, Figures, ROC")
print("=" * 50)

lbp_genuine  = all_scores['LBP']['genuine']
lbp_impostor = all_scores['LBP']['impostor']
lbp_rank1    = all_scores['LBP']['rank1_euc']

eig_genuine  = all_scores['PCA (Eigenfaces)']['genuine']
eig_impostor = all_scores['PCA (Eigenfaces)']['impostor']
eig_rank1    = all_scores['PCA (Eigenfaces)']['rank1_euc']

hog_genuine  = all_scores['HOG']['genuine']
hog_impostor = all_scores['HOG']['impostor']
hog_rank1    = all_scores['HOG']['rank1_euc']

fused_gen_arr, fused_imp_arr = fuse_genuine_impostor(
    genuine_lists  = [lbp_genuine, eig_genuine, hog_genuine],
    impostor_lists = [lbp_impostor, eig_impostor, hog_impostor],
)

fused_genuine  = fused_gen_arr.tolist()
fused_impostor = fused_imp_arr.tolist()

fused_rank1 = all_scores['Fused']['rank1_euc']

run_full_evaluation(
    lbp_genuine,   lbp_impostor,   lbp_rank1,
    eig_genuine,   eig_impostor,   eig_rank1,
    hog_genuine,   hog_impostor,   hog_rank1,
    fused_genuine, fused_impostor, fused_rank1,
)