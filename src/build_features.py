import numpy as np


def build_gallery_features(gallery, extract_fn):
    
    gallery_features = {}
    for subject_id, face_list in gallery.items():
        vectors = [extract_fn(face) for face in face_list]
        # Average all training vectors → one robust template per person
        gallery_features[subject_id] = np.mean(vectors, axis=0)
    return gallery_features


def build_probe_features(probes, extract_fn):
    
    probe_features = {}
    for subject_id, face_list in probes.items():
        vectors = [extract_fn(face) for face in face_list]
        probe_features[subject_id] = np.mean(vectors, axis=0)
    return probe_features