import numpy as np
from feature_extraction import EigenfaceExtractor, LBPExtractor, HOGExtractor
import pickle
import os


def l2_normalize(vec):
    #Scale a vector to length 1 so all methods contribute equally.
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def extract_fused(face_img, pca_ext, lbp_ext, hog_ext):
    """
    Takes one face image (flattened numpy array from preprocessing).
    Runs all three extractors, normalizes each, concatenates.
    Returns one combined feature vector.
    """
    pca_vec = l2_normalize(pca_ext.extract(face_img))   # (75,)
    lbp_vec = l2_normalize(lbp_ext.extract(face_img))   # (4096,)
    hog_vec = l2_normalize(hog_ext.extract(face_img))   # (8100,)
    return np.concatenate([pca_vec, lbp_vec, hog_vec])  # (12271,)


class FusedExtractor:
  
    def __init__(self, pca_ext, lbp_ext, hog_ext):
        self.pca_ext = pca_ext
        self.lbp_ext = lbp_ext
        self.hog_ext = hog_ext

    def fit(self, gallery):
        # Only PCA needs training 
        self.pca_ext.fit(gallery)
        print("FusedExtractor fitted.")

    def extract(self, face_img):
        return extract_fused(face_img, self.pca_ext,self.lbp_ext, self.hog_ext)
                             

    def save(self, path='models/fused_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Fused model saved to {path}")

    @staticmethod
    def load(path='models/fused_model.pkl'):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Fused model loaded from {path}")
        return model