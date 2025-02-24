import numpy as np
from numpy import dot 
from numpy.linalg import norm 


def cos_similarity(v1, v2):
    if (v1 is None) or (v2 is None):
        return None
    if (len(v1) == 0) or (len(v2) == 0):
        return None
    return dot(v1, v2) / (norm(v1) * norm(v2)) 


def l2_distance(v1, v2):
    return norm(v1 - v2)


def get_average_similarity(v0, v_list):
    if v_list:
        return np.average([ cos_similarity(v0, v) for v in v_list if v is not None])
    return None


def normalize_matrix(v, axis=1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2
        

def matrix_similarity(m1, m2):
    if (m1 is None) or (len(m1) == 0) or (m2 is None) or (len(m2) == 0):
        return None
    m1 = normalize_matrix(m1, axis=1)
    m2 = normalize_matrix(m2, axis=1)
    return (m1 @ m2.T).max()


def filter_cluster_elements(clusters, filter_function):
    filtered = []
    for cluster in clusters:
        r = [i for i in cluster if filter_function(i)]
        if r:
            filtered.append(r)
    return filtered
