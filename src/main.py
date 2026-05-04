from preprocessing import *
from feature_extraction import *
from matching import *
from evaluation import *
import os
from preprocessing import load_face_dataset
from feature_extraction import EigenfaceExtractor
from build_features import build_gallery_features, build_probe_features

# ── Step 1: Load data ──────────────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading data")
print("=" * 50)
gallery, probes = load_face_dataset()
print(f"Gallery subjects: {len(gallery)}")   # should be ~50
print(f"Probe subjects:   {len(probes)}")    # should be ~50

# ── Step 2: Fit Eigenfaces ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Fitting Eigenfaces (PCA)")
print("=" * 50)

MODEL_PATH = 'pca_model.pkl'

if os.path.exists(MODEL_PATH):
    ef = EigenfaceExtractor.load(MODEL_PATH)
else:
    ef = EigenfaceExtractor(n_components=75)
    ef.fit(gallery)
    ef.save(MODEL_PATH)

# ── Step 3: Extract features ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Extracting features")
print("=" * 50)

gallery_features = build_gallery_features(gallery, ef.extract)
probe_features   = build_probe_features(probes,   ef.extract)

print(f"Gallery feature vectors: {len(gallery_features)}")  # should be ~50
print(f"Probe feature vectors:   {len(probe_features)}")    # should be ~50

sample_subject = sorted(gallery_features.keys())[0]
sample_vector  = gallery_features[sample_subject]
print(f"Feature vector shape: {sample_vector.shape}")   # should say (75,)
print(f"Sample values: {sample_vector[:5].round(3)}")

print("\nAll done. Ready for matching.")

