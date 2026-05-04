import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

class EigenfaceExtractor:
    def __init__(self, n_components=75):
        self.n_components=n_components
        self.pca=PCA(n_components=n_components, whiten=True)
        self.fitted = False


    def fit(self, preprocessed_train_dir):
        images=[]
        for subject_id, face_list in gallery.items():
            for face in face_list:
                images.append(face.astype(np.float32).flatten())


        X=np.array(images) # shape: (600, 10000)
        self.pca.fit(X)
        self.fitted=True

        var_retained=self.pca.explained_variance_ratio_.cumsum()[-1]
        print(f"PCA fitted on {len(images)} images.")
        print(f"{self.n_components} components retain {var * 100:.1f}% of variance.")
        

    def extract(self, face_img):
            assert self.fitted,  "Call fit() before extract()."
            x =face_img.astype(np.float32).ravel().reshape(1,-1)
            return self.pca.transform(x).ravel() # shape: (75,)
        

    def save(self, path='pca_model.pkl'):
            with open(path, 'wb') as f:
                pickle.dump(self,f)
            print(f"PCA model saved to {path}")   

        
    @staticmethod
    def load(path='pca_model.pkl'):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"PCA model loaded from {path}")
            return model

