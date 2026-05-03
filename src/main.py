from preprocessing import load_face_dataset
from feature_extraction import extract_lbp, PCAExtractor, extract_hog, extract_fused

training, testing = load_face_dataset()

print(f"Training shape: {training.shape}")  # expect (600, 16384)
print(f"Testing shape:  {testing.shape}")   # expect (150, 16384)

assert training.shape == (600, 16384), "Wrong training size!"
assert testing.shape  == (150, 16384), "Wrong testing size!"