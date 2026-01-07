# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category = UserWarning)
import os
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
import building_cognitive_diagram as bcd
import sample_generation as sg
import model_training as mt
import classification as cla
import clustering as clu

# ==== Variables ====
task_type = "MaxNum"    # e.g. "WCST", "Nback", "MaxNum"

seq_len = 10    # the length of each stimuli sequence
num_train_samples = 10000
num_test_samples = 1000
random_seed = 19

hidden_layer_size = 32
batch_size = 4
learning_rate = 0.001
val_ratio = 0.2    # validation ratio

tol = 1e-2    # tolerance for SVM classification
num_str_vars = 2    # how many last stimuli to consider for str_vars

# ==== Directories ====
output_path = os.path.join('output', task_type)

os.makedirs(output_path, exist_ok = True)

if __name__ == "__main__":
    
    print("- NOTE: If you want to change the task variables, refer to building_cognitive_diagram.py...")
    
    # build a cognitive diagram
    if task_type == "Nback":
        task = bcd.build_diagram_Nback()
    elif task_type == "MaxNum":
        task = bcd.build_diagram_MaxNum()
    elif task_type == "WCST":
        task = bcd.build_diagram_WCST()
    
    # save the diagram
    with open(os.path.join(output_path, f"{task_type}_task.json"), "w") as f:
        json.dump(task.to_dict(), f, indent = 2)
        
    # generate samples
    print("- Generating samples...")
    
    train_samples, test_samples = sg.generate_samples(task, seq_len, num_train_samples, num_test_samples, random_seed)
    train_ans, _ = sg.generate_ans_state(task, train_samples)
    test_ans, test_state = sg.generate_ans_state_test(task, test_samples)
    
    stimuli_enc, ans_enc = sg.generate_enc(task)
    
    train_samples_encoded = sg.encode_samples(train_samples, stimuli_enc)
    train_ans_encoded = sg.encode_answers(train_ans, ans_enc)
    
    test_samples_encoded = sg.encode_samples(test_samples, stimuli_enc)
    test_ans_encoded = sg.encode_answers(test_ans, ans_enc)
    
    np.savez_compressed(os.path.join(output_path, f"{task_type}_test_data.npz"), 
                        test_samples = test_samples, test_ans = test_ans, test_state = test_state)

    # train an RNN model
    print("- Training an RNN model...")
    train_loader, val_loader = mt.generate_loader(train_samples_encoded, train_ans_encoded, val_ratio, batch_size, random_seed, if_train = True)
    test_loader = mt.generate_loader(test_samples_encoded, test_ans_encoded, val_ratio, batch_size, random_seed, if_train = False)
    
    model, log = mt.train_model(train_loader, val_loader, len(task.stimuli_type), hidden_layer_size, len(task.action_type), learning_rate)
    
    test_vectors = mt.extract_features(model, test_loader)

    torch.save(model.state_dict(), os.path.join(output_path, f"{task_type}_model_state.pt"))
    with open(os.path.join(output_path, f"{task_type}_train_log.csv"), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    
    # [analysis 1] Probing
    print("- Doing an analysis: Probing...")
    shuffle_acc = cla.kfold(test_vectors, test_state, "SVM_lin", random_seed, k = 10, 
                           target_dim = None, dim_method = None, state_shuffle = True, tol = tol)
    lin_acc = cla.kfold(test_vectors, test_state, "SVM_lin", random_seed, k = 10, 
                           target_dim = None, dim_method = None, state_shuffle = False, tol = tol)
    rbf_acc = cla.kfold(test_vectors, test_state, "SVM_rbf", random_seed, k = 10, 
                           target_dim = None, dim_method = None, state_shuffle = False, tol = tol)
    
    print(f"Summary: SVM + shuffle {np.mean(shuffle_acc):.4f} | SVM + linear kernel {np.mean(lin_acc):.4f} | SVM + rbf kernel {np.mean(rbf_acc):.4f}")
    
    probing_acc = {"shuffle_acc": shuffle_acc, "lin_acc": lin_acc, "rbf_acc": rbf_acc}
    with open(os.path.join(output_path, f"{task_type}_probing_acc.json"), "w") as f:
        json.dump(probing_acc, f, indent = 2)

    # [analysis 2] Dimension Reduction
    print("- Doing an analysis: Dimension Reduction...")
    max_dim = min(hidden_layer_size, len(np.unique(test_state)))
    
    dimension_acc = dict()
    
    for target_dim in tqdm(range(1, max_dim + 1), desc = f"Classification (max_dim: {max_dim})", leave = False):
        if target_dim == max_dim:
            target_dim = None
            
        lin_acc_lda = cla.kfold(test_vectors, test_state, "SVM_lin", random_seed, k = 10, 
                               target_dim = target_dim, dim_method = "LDA", state_shuffle = False, tol = tol)
        rbf_acc_lda = cla.kfold(test_vectors, test_state, "SVM_rbf", random_seed, k = 10, 
                               target_dim = target_dim, dim_method = "LDA", state_shuffle = False, tol = tol)
        
        lin_acc_pca = cla.kfold(test_vectors, test_state, "SVM_lin", random_seed, k = 10, 
                               target_dim = target_dim, dim_method = "PCA", state_shuffle = False, tol = tol)
        rbf_acc_pca = cla.kfold(test_vectors, test_state, "SVM_rbf", random_seed, k = 10, 
                               target_dim = target_dim, dim_method = "PCA", state_shuffle = False, tol = tol)
        
        dimension_acc[str(target_dim)] = {"lin_acc_lda": lin_acc_lda, "rbf_acc_lda": rbf_acc_lda, 
                                          "lin_acc_pca": lin_acc_pca, "rbf_acc_pca": rbf_acc_pca}
    
    with open(os.path.join(output_path, f"{task_type}_dimension_acc.json"), "w") as f:
        json.dump(dimension_acc, f, indent = 2)

    # [analysis 3] Probing Other Variables
    print("- Doing an analysis: Probing Other Variables...")
    vars_acc = {"func_vars": dict(), "str_vars": dict()}
    
    func_vars = task.generate_func_vars(test_samples)
    for var in tqdm(func_vars.keys(), desc = "func_vars", leave = False):
        lin_acc = cla.kfold(test_vectors, func_vars[var], "SVM_lin", random_seed, k = 10, 
                               target_dim = None, dim_method = None, state_shuffle = False, tol = tol)
        rbf_acc = cla.kfold(test_vectors, func_vars[var], "SVM_rbf", random_seed, k = 10, 
                               target_dim = None, dim_method = None, state_shuffle = False, tol = tol)
        vars_acc["func_vars"][var] = {"lin_acc": lin_acc, "rbf_acc": rbf_acc}
    
    for var in tqdm(range(num_str_vars, 0, -1), desc = "str_vars", leave = False):
        str_var = [sample[-var] for sample in test_samples]
        lin_acc = cla.kfold(test_vectors, str_var, "SVM_lin", random_seed, k = 10, 
                               target_dim = None, dim_method = None, state_shuffle = False, tol = tol)
        rbf_acc = cla.kfold(test_vectors, str_var, "SVM_rbf", random_seed, k = 10, 
                               target_dim = None, dim_method = None, state_shuffle = False, tol = tol)
        
        vars_acc["str_vars"][f"{var} positions back"] = {"lin_acc": lin_acc, "rbf_acc": rbf_acc}
        
    with open(os.path.join(output_path, f"{task_type}_vars_acc.json"), "w") as f:
        json.dump(vars_acc, f, indent = 2)

    # [analysis 4] Clustering
    print("- Doing an analysis: Clustering...")
    num_states = len(np.unique(test_state))
    k_list = [num_states - 1, num_states, num_states + 1, 2 * num_states, 4 * num_states, 6 * num_states, 8 * num_states]
    eps_list = [0.25, 0.5, 1, 1.5, 2]
    
    clustering_wamb = {"cog_state": dict(), "func_vars": dict(), "str_vars": dict(), "k_list": k_list, "eps_list": eps_list}
    
    # cog_state
    clustering_wamb["cog_state"] = {"kmeans": [], "DBSCAN": []}
    
    for k in k_list:
        cluster = clu.run_kmeans(test_vectors, k, random_seed)
        wamb = clu.calculate_wamb(test_state, cluster)
        clustering_wamb["cog_state"]["kmeans"].append(wamb)
        
    for eps in eps_list:
        cluster = clu.run_DBSCAN(test_vectors, eps)
        wamb = clu.calculate_wamb(test_state, cluster)
        clustering_wamb["cog_state"]["DBSCAN"].append(wamb)
        
    # func_vars
    for var in tqdm(func_vars.keys(), desc = "func_vars", leave = False):
        clustering_wamb["func_vars"][var] = {"kmeans": [], "DBSCAN": []}
        
        for k in k_list:
            cluster = clu.run_kmeans(test_vectors, k, random_seed)
            wamb = clu.calculate_wamb(func_vars[var], cluster)
            clustering_wamb["func_vars"][var]["kmeans"].append(wamb)
            
        for eps in eps_list:
            cluster = clu.run_DBSCAN(test_vectors, eps)
            wamb = clu.calculate_wamb(func_vars[var], cluster)
            clustering_wamb["func_vars"][var]["DBSCAN"].append(wamb)
    
    # str_vars
    for var in tqdm(range(num_str_vars, 0, -1), desc = "str_vars", leave = False):
        str_var = [sample[-var] for sample in test_samples]
        clustering_wamb["str_vars"][f"{var} positions back"] = {"kmeans": [], "DBSCAN": []}
        
        for k in k_list:
            cluster = clu.run_kmeans(test_vectors, k, random_seed)
            wamb = clu.calculate_wamb(str_var, cluster)
            clustering_wamb["str_vars"][f"{var} positions back"]["kmeans"].append(wamb)
        
        for eps in eps_list:
            cluster = clu.run_DBSCAN(test_vectors, eps)
            wamb = clu.calculate_wamb(str_var, cluster)
            clustering_wamb["str_vars"][f"{var} positions back"]["DBSCAN"].append(wamb)
    
    with open(os.path.join(output_path, f"{task_type}_clustering_wamb.json"), "w") as f:
        json.dump(clustering_wamb, f, indent = 2)
    
    # save the metadata
    metadata = { "task_type": task_type, "seq_len": seq_len, "num_train_samples": num_train_samples,
                "num_test_samples": num_test_samples, "random_seed": random_seed, "hidden_layer_size": hidden_layer_size,
                "batch_size": batch_size, "learning_rate": learning_rate, "val_ratio": val_ratio, "tol": tol,
                "num_str_vars": num_str_vars }
    with open(os.path.join(output_path, f"{task_type}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent = 2)
    
    print("- Code done successfully!")

