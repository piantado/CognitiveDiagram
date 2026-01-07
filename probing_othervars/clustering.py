# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(vec_list, k, random_seed):
    
    vec_list = StandardScaler().fit_transform(vec_list)
    
    pca = PCA(n_components = 0.9)
    vec_list = pca.fit_transform(vec_list)
    
    kmeans = KMeans(n_clusters = k, n_init = 50, random_state = random_seed)
    cluster_pred = kmeans.fit_predict(vec_list)
    
    return cluster_pred

def run_DBSCAN(vec_list, eps):
    
    vec_list = StandardScaler().fit_transform(vec_list)
    
    pca = PCA(n_components = 0.9)
    vec_list = pca.fit_transform(vec_list)
    
    min_samples = max(vec_list.shape[1] + 1, 5)
    
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    cluster_pred = dbscan.fit_predict(vec_list)
    
    return cluster_pred
    
def calculate_wamb(state_list, pred_list):
    
    state_list = np.asarray(state_list)
    pred_list = np.asarray(pred_list)
    assert state_list.shape == pred_list.shape
    
    abs_HQ = state_list.shape[0]
    abs_Q = len(np.unique(state_list))
    K = np.unique(pred_list)
    
    state_idx = {state: idx for idx, state in enumerate(np.unique(state_list))}
    state_encoded = np.vectorize(state_idx.get)(state_list)
    
    wamb_k = 0.0
    for cluster in K:
        
        qk_mask = (pred_list == cluster)
        n_k = int(qk_mask.sum())
        if n_k == 0:
            continue

        p = np.bincount(state_encoded[qk_mask], minlength = len(np.unique(state_list))) / n_k
        p = p[p > 0]
        
        amb_k = -np.sum(p * (np.log(p) / np.log(abs_Q)))
        wamb_k += amb_k * n_k
    
    return wamb_k / abs_HQ
    
    