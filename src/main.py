import os
from preprocessing import load_face_dataset
from feature_extraction import EigenfaceExtractor, LBPExtractor, HOGExtractor
from build_features import build_gallery_features, build_probe_features
from matching import identify_subject, compute_all_scores
from fusion import FusedExtractor

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

os.makedirs("models", exist_ok=True)   # create models/ folder if it doesn't exist
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
print(f"LBP  vectors ready — shape: {lbp_gallery['s01'].shape}")   # (4096,)

hog_gallery = build_gallery_features(gallery, hog_ext.extract)
hog_probes  = build_probe_features(probes,   hog_ext.extract)
print(f"HOG  vectors ready — shape: {hog_gallery['s01'].shape}")   # (8100,)

# ── Step 4: Build fused features ──────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Fused features (PCA + LBP + HOG)")
print("=" * 50)

FUSED_PATH = 'models/fused_model.pkl'

if os.path.exists(FUSED_PATH):
    fused = FusedExtractor.load(FUSED_PATH)
else:
    # Create fresh extractors for fused — PCA needs to be refitted inside fused
    pca_f = EigenfaceExtractor(n_components=75)
    lbp_f = LBPExtractor()
    hog_f = HOGExtractor()
    fused = FusedExtractor(pca_f, lbp_f, hog_f)
    fused.fit(gallery)
    fused.save(FUSED_PATH)

fused_gallery = build_gallery_features(gallery, fused.extract)
fused_probes  = build_probe_features(probes,   fused.extract)
print(f"Fused vectors ready — shape: {fused_gallery['s01'].shape}")  # (12271,)

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

all_scores = {}  # saved for Person 4's metrics and curves

print(f"\n{'Method':<20} {'Euclidean Rank-1':>17} {'Cosine Rank-1':>14}")
print("-" * 55)

for method_name, (gal, prb) in methods.items():
    total = len(prb)

    # Euclidean
    euc_correct = sum(
        1 for pid, pvec in prb.items()
        if identify_subject(pvec, gal, metric='euclidean')[0] == pid
    )
    euc_rank1 = euc_correct / total * 100

    # Cosine
    cos_correct = sum(
        1 for pid, pvec in prb.items()
        if identify_subject(pvec, gal, metric='cosine')[0] == pid
    )
    cos_rank1 = cos_correct / total * 100

    print(f"{method_name:<20} {euc_rank1:>16.2f}% {cos_rank1:>13.2f}%")

    # Scores for curves — use euclidean (Person 4 will use these)
    genuine, impostor = compute_all_scores(gal, prb, metric='euclidean')
    all_scores[method_name] = {
        'genuine':       genuine,
        'impostor':      impostor,
        'rank1_euc':     euc_rank1,
        'rank1_cos':     cos_rank1,
    }

print("\n" + "=" * 50)
print("DONE — all_scores ready for metrics and curves")
print("=" * 50)
print("Keys in all_scores:", list(all_scores.keys()))
print("Each entry contains: genuine scores, impostor scores, rank1_euc, rank1_cos")
