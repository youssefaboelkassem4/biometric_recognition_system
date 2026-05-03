import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp(images, img_size=(128, 128), neighbors=8, radius=1, n=256):
    features = []
    for i in images:
        img     = (i.reshape(img_size) * 255).astype(np.uint8)
        lbp     = local_binary_pattern(img, P=neighbors, R=radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n, range=(0, n))
        hist    = hist.astype(np.float32)
        hist   /= hist.sum() + 1e-6
        features.append(hist)
    return np.array(features, dtype=np.float32)