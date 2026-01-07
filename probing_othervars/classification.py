# -*- coding: utf-8 -*-

import numpy as np
import random
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def kfold(vec_list, state_list, class_model, random_seed, k, target_dim, dim_method, state_shuffle, tol):
    
    kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_seed)
    acc_list = []

    vec = np.array(vec_list)
    state = np.array(state_list)
    split_list = list(kf.split(vec, state))
    
    if state_shuffle:
        state_shuffle = state_list.copy()
        random.seed(random_seed)
        random.shuffle(state_shuffle)
        state = np.array(state_shuffle)
        
    for fold in tqdm(range(kf.get_n_splits()), desc = f"{class_model}", leave = False):
        
        train_idx, test_idx = split_list[fold]
   
        vec_train, vec_test = vec[train_idx], vec[test_idx]
        state_train, state_test = state[train_idx], state[test_idx]
        
        if (not target_dim == None) and (dim_method == "LDA"):
            LDA_model = LinearDiscriminantAnalysis(n_components = target_dim)
            vec_train = LDA_model.fit_transform(vec_train, state_train)
            vec_test = LDA_model.transform(vec_test)
        elif (not target_dim == None) and (dim_method == "PCA"):
            PCA_model = PCA(n_components = target_dim)
            vec_train = PCA_model.fit_transform(vec_train, state_train)
            vec_test = PCA_model.transform(vec_test)

        if class_model == "LDA":
            acc = run_LDA(vec_train, state_train, vec_test, state_test)
        elif class_model == "SVM_lin":
            acc = run_SVM(vec_train, state_train, vec_test, state_test, 'linear', tol)
        elif class_model == "SVM_rbf":
            acc = run_SVM(vec_train, state_train, vec_test, state_test, 'rbf', tol)
            
        acc_list.append(acc)

    return acc_list

def run_LDA(vec_train, state_train, vec_test, state_test):
    
    LDA_model = LinearDiscriminantAnalysis()
    LDA_model.fit(vec_train, state_train)
    state_pred = LDA_model.predict(vec_test)
    
    return accuracy_score(state_test, state_pred)

def run_SVM(vec_train, state_train, vec_test, state_test, kernel, tol):
    
    if not tol == None:
        SVM_model_null = SVC(kernel = kernel, tol = tol)
    else:
        SVM_model_null = SVC(kernel = kernel)

    param_grid = {
        'C': [0.1, 1.0, 10, 50],
        'gamma': ['scale', 'auto']
    }
    grid_svm = GridSearchCV(estimator = SVM_model_null, param_grid = param_grid, cv = 2, scoring = 'accuracy', n_jobs = -1)
    grid_svm.fit(vec_train, state_train)
    C_optim = grid_svm.best_params_['C']
    gamma_optim = grid_svm.best_params_['gamma']
    
    if not tol == None:
        SVM_model = SVC(kernel = kernel, C = C_optim, gamma = gamma_optim, tol = tol)
    else:
        SVM_model = SVC(kernel = kernel, C = C_optim, gamma = gamma_optim)
    SVM_model.fit(vec_train, state_train)
    state_pred = SVM_model.predict(vec_test)
    
    return accuracy_score(state_test, state_pred)