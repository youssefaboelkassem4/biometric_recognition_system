import numpy as np

def compute_euclidean(v1, v2):
    return np.linalg.norm(v1 - v2)

def compute_cosine(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return 1.0 - (dot_product / (norm_v1 * norm_v2))

def identify_subject(probe_vector, gallery_features, metric='euclidean'):
    min_distance = float('inf')
    predicted_id = None
    
    for subject_id, gallery_vector in gallery_features.items():
        if metric == 'euclidean':
            distance = compute_euclidean(probe_vector, gallery_vector)
        elif metric == 'cosine':
            distance = compute_cosine(probe_vector, gallery_vector)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
            
        if distance < min_distance:
            min_distance = distance
            predicted_id = subject_id
            
    return predicted_id, min_distance

def compute_all_scores(gallery_features, probe_features, metric='euclidean'):
    genuine_scores = []
    imposter_scores = []
    
    for probe_id, probe_vector in probe_features.items():
        for gallery_id, gallery_vector in gallery_features.items():
            if metric == 'euclidean':
                distance = compute_euclidean(probe_vector, gallery_vector)
            elif metric == 'cosine':
                distance = compute_cosine(probe_vector, gallery_vector)
            
            if probe_id == gallery_id:
                genuine_scores.append(distance)
            else:
                imposter_scores.append(distance)
                
    return genuine_scores, imposter_scores
