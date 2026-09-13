import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
import clustering as clu
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
import model_training as mt
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from tqdm import tqdm

# Analysis 1
def project_hidden(model, vectors, tol):

    W = model.fc.weight.detach().clone()
    W = W - W.mean(dim = 0, keepdim = True)
    
    U, S, Vh = torch.linalg.svd(W, full_matrices = False)

    rank = (S > tol * S.max()).sum().item()
    Q = Vh[:rank].T.numpy()

    proj_coords = vectors @ Q
    vectors_proj = proj_coords @ Q.T

    return proj_coords, vectors_proj


def readout_decodability(vectors, states, model, SVM_kernel, k, random_state, tol):

    readout_acc = {
		classifier: {vec_type: {} for vec_type in ["vectors_raw", "vectors_proj", "vectors_wo_proj"]}
		for classifier in ["LDA", "SVM", "PCA_LDA", "PCA_SVM"]
	    }
    
    vectors_center = vectors - np.mean(vectors, axis = 0, keepdims = True)
    proj_coords, vectors_proj = project_hidden(model, vectors_center, tol)

    for classifier in readout_acc.keys():
        for vec_type in readout_acc[classifier].keys():

            vectors_probe = vectors_center
            if vec_type == "vectors_proj":
                vectors_probe = vectors_proj
            elif vec_type == "vectors_wo_proj":
                vectors_probe = vectors_center - vectors_proj

            if classifier == "LDA":
                clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            elif classifier == "SVM":
                clf = make_pipeline(StandardScaler(), SVC(kernel = SVM_kernel))
            elif classifier == "PCA_LDA":
                clf = make_pipeline(StandardScaler(), PCA(n_components = 2), LinearDiscriminantAnalysis())
            else:
                clf = make_pipeline(StandardScaler(), PCA(n_components = 2), SVC(kernel = SVM_kernel))
        
            cv = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_state)
            scores_acc = cross_val_score(clf, vectors_probe, states, cv = cv, scoring = "accuracy")
            scores_bal_acc = cross_val_score(clf, vectors_probe, states, cv = cv, scoring = "balanced_accuracy")

            readout_acc[classifier][vec_type] = {"scores_acc": scores_acc, "scores_bal_acc": scores_bal_acc}

    return readout_acc


def calculate_separability(vectors, states, eps):

    unique_states = np.unique(states)
    states = np.array(states)
    centroids = []
    spreads = []

    preproc = make_pipeline(StandardScaler(), PCA(n_components = 0.9))
    vectors_preproc = preproc.fit_transform(vectors)

    for state in unique_states:

        vectors_state = vectors_preproc[states == state]

        mu = vectors_state.mean(axis = 0)
        centroids.append(mu)
        dist_to_mu = np.linalg.norm(vectors_state - mu, axis = 1)
        spreads.append(np.mean(dist_to_mu))

    centroids = np.vstack(centroids)
    spreads = np.asarray(spreads)

    D = cdist(centroids, centroids, metric = "euclidean")
    np.fill_diagonal(D, np.inf)

    nearest_dist = D.min(axis = 1)
    separability_index = nearest_dist / (spreads + eps)

    return separability_index


def readout_separability(model, vectors, states, tol):

    readout_sep = {vec_type: {} for vec_type in ["vectors_raw", "vectors_proj", "vectors_wo_proj"]}

    vectors_center = vectors - np.mean(vectors, axis = 0, keepdims = True)
    proj_coords, vectors_proj = project_hidden(model, vectors_center, tol)

    for vec_type in readout_sep.keys():

        vectors_probe = vectors_center
        if vec_type == "vectors_proj":
            vectors_probe = vectors_proj
        elif vec_type == "vectors_wo_proj":
            vectors_probe = vectors_center - vectors_proj
        
        readout_sep[vec_type] = calculate_separability(vectors_probe, states, tol)

    return readout_sep


def readout_cond_decodability(vectors, states, ans_list, model, SVM_kernel, k, random_state, tol):

    ans_list = np.array(ans_list)
    states = np.array(states)
    unique_ans = np.unique(ans_list)

    readout_cond_acc = {
        classifier: {vec_type: {} for vec_type in ["vectors_raw", "vectors_proj", "vectors_wo_proj"]}
		for classifier in ["LDA", "SVM", "PCA_LDA", "PCA_SVM"]
	    }

    vectors_center = vectors - np.mean(vectors, axis = 0, keepdims = True)
    proj_coords, vectors_proj = project_hidden(model, vectors_center, tol)
    
    for classifier in readout_cond_acc.keys():
        for vec_type in readout_cond_acc[classifier].keys():

            scores_acc = []
            scores_bal_acc = []

            vectors_probe = vectors_center
            if vec_type == "vectors_proj":
                vectors_probe = vectors_proj
            elif vec_type == "vectors_wo_proj":
                vectors_probe = vectors_center - vectors_proj

            for ans in unique_ans:

                state_probe = states[ans_list == ans]
                vectors_cond = vectors_probe[ans_list == ans]

                if len(np.unique(state_probe)) < 2:
                    continue

                if classifier == "LDA":
                    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
                elif classifier == "SVM":
                    clf = make_pipeline(StandardScaler(), SVC(kernel = SVM_kernel))
                elif classifier == "PCA_LDA":
                    clf = make_pipeline(StandardScaler(), PCA(n_components = 2), LinearDiscriminantAnalysis())
                else:
                    clf = make_pipeline(StandardScaler(), PCA(n_components = 2), SVC(kernel = SVM_kernel))
        
                cv = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_state)
                scores_acc.extend(cross_val_score(clf, vectors_cond, state_probe, cv = cv, scoring = "accuracy"))
                scores_bal_acc.extend(cross_val_score(clf, vectors_cond, state_probe, cv = cv, scoring = "balanced_accuracy"))

            readout_cond_acc[classifier][vec_type] = {"scores_acc": scores_acc, "scores_bal_acc": scores_bal_acc}

    return readout_cond_acc


def readout_cond_separability(vectors, states, ans_list, model, tol): 

    ans_list = np.array(ans_list)
    states = np.array(states)
    unique_ans = np.unique(ans_list) 

    readout_cond_sep = {vec_type: {} for vec_type in ["vectors_raw", "vectors_proj", "vectors_wo_proj"]}
    
    vectors_center = vectors - np.mean(vectors, axis = 0, keepdims = True) 
    proj_coords, vectors_proj = project_hidden(model, vectors_center, tol) 
    
    for vec_type in readout_cond_sep.keys(): 

        sep = [] 
        
        vectors_probe = vectors_center 
        if vec_type == "vectors_proj": 
            vectors_probe = vectors_proj 
        elif vec_type == "vectors_wo_proj": 
            vectors_probe = vectors_center - vectors_proj 
            
        for ans in unique_ans: 
            
            state_probe = states[ans_list == ans] 
            vectors_cond = vectors_probe[ans_list == ans] 
            
            if len(np.unique(state_probe)) < 2:
                continue

            sep.extend(calculate_separability(vectors_cond, state_probe, tol)) 
            
        readout_cond_sep[vec_type] = sep 
            
    return readout_cond_sep


def get_other_vars(samples, ans, states):

    other_vars = {key: [] for key in ["last_stim", "sec_last_stim", "last_two_stim", "curr_ans", "last_ans", "trial_idx"]}
    other_vars["cog_state"] = states

    for sample_idx in range(len(samples)):

        sample = samples[sample_idx]

        for stim_idx in range(len(sample)):
            
            last_stim = "0" if stim_idx == 0 else sample[stim_idx - 1]
            sec_last_stim = "0" if stim_idx <= 1 else sample[stim_idx - 2]
            last_two_stim = last_stim + sec_last_stim
            
            curr_ans = ans[sample_idx][stim_idx]
            last_ans = "0" if stim_idx == 0 else ans[sample_idx][stim_idx - 1]

            vals = {"sec_last_stim": sec_last_stim, "last_stim": last_stim, "last_two_stim": last_two_stim, 
                    "curr_ans": curr_ans, "last_ans": last_ans, "trial_idx": stim_idx}

            for key, val in vals.items():
                other_vars[key].append(val)

    return other_vars


def get_clustering(vectors, var_list, num_states, random_seed):

    k_list = [round(num_states / 4), round(num_states / 2), num_states, round(num_states * 2), round(num_states * 4), round(num_states * 6)]
    eps_list = [0.25, 0.5, 1, 1.5, 2]

    optim_wamb_kmeans, optim_ari_kmeans = np.inf, -np.inf
    optim_wamb_dbscan, optim_ari_dbscan = np.inf, -np.inf

    for k in k_list:

        cluster_kmeans = clu.run_kmeans(vectors, k, random_seed)
        wamb_kmeans = clu.calculate_wamb(var_list, cluster_kmeans)
        ari_kmeans = clu.calculate_ari(var_list, cluster_kmeans)
            
        if optim_wamb_kmeans > wamb_kmeans:
            optim_k_wamb = k
            optim_wamb_kmeans = wamb_kmeans
        if optim_ari_kmeans < ari_kmeans:
            optim_k_ari = k
            optim_ari_kmeans = ari_kmeans

    for eps in eps_list:
        cluster_dbscan = clu.run_DBSCAN(vectors, eps, num_states)
        wamb_dbscan = clu.calculate_wamb(var_list, cluster_dbscan)
        ari_dbscan = clu.calculate_ari(var_list, cluster_dbscan)

        if optim_wamb_dbscan > wamb_dbscan:
            optim_eps_wamb = eps
            optim_wamb_dbscan = wamb_dbscan
        if optim_ari_dbscan < ari_dbscan:
            optim_eps_ari = eps
            optim_ari_dbscan = ari_dbscan

    clustering_metrics = {
        "optim_k_wamb": optim_k_wamb, "optim_wamb_kmeans": optim_wamb_kmeans,
        "optim_k_ari": optim_k_ari, "optim_ari_kmeans": optim_ari_kmeans,
        "optim_eps_wamb": optim_eps_wamb, "optim_wamb_dbscan": optim_wamb_dbscan, 
        "optim_eps_ari": optim_eps_ari, "optim_ari_dbscan": optim_ari_dbscan
        }

    return clustering_metrics

def conf_samples(model, vectors, loader, states, percentage):

    all_samples, all_conf, all_probs = [], [], []

    model.eval()

    with torch.no_grad():
        for seq_batch, ans_batch in loader:

            logits = model(seq_batch)
            predicted = torch.argmax(logits, dim = 2)

            probs = F.softmax(logits, dim = 2)
            confidence = probs.max(dim = 2).values

            all_samples.append(seq_batch.reshape(-1, seq_batch.shape[-1]))
            all_conf.append(confidence.reshape(-1))
            all_probs.append(probs.reshape(-1, probs.shape[-1]))

    all_samples = torch.cat(all_samples, dim = 0)
    all_probs = torch.cat(all_probs, dim = 0)
    all_conf = torch.cat(all_conf, dim = 0)

    all_vecs = torch.tensor(vectors)
    all_states = np.array(states)

    idx = max(1, int(len(all_conf) * percentage))
    sorted_idx = torch.argsort(all_conf)

    least_idx = sorted_idx[:idx]
    top_idx = sorted_idx[-idx:]

    assert len(all_vecs) == len(all_states)

    conf_dict = {
        "least_samples": all_samples[least_idx], "top_samples": all_samples[top_idx],
        "least_probs": all_probs[least_idx], "top_probs": all_probs[top_idx],
        "least_conf": all_conf[least_idx], "top_conf": all_conf[top_idx],
        "least_vecs": all_vecs[least_idx], "top_vecs": all_vecs[top_idx],
        "least_states": all_states[least_idx.numpy()], "top_states": all_states[top_idx.numpy()]
        }

    return conf_dict


def decoder_conf(conf_dict, states_all, SVM_kernel, k, random_seed):

    conf_acc = {classifier: dict() for classifier in ["LDA", "SVM", "PCA_LDA", "PCA_SVM"]}

    for cond in ["least", "top"]:
        
        vectors = conf_dict[cond + "_vecs"]
        vectors = (vectors - vectors.mean(dim = 0, keepdim = True)).numpy()
        states = conf_dict[cond + "_states"]
        unique_states = np.unique(states_all)

        for classifier in conf_acc.keys():

            pred = np.empty(len(states), dtype = states.dtype)
            proba = np.zeros((len(states), len(unique_states)))

            if classifier == "LDA":
                clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            elif classifier == "SVM":
                clf = make_pipeline(StandardScaler(), SVC(kernel = SVM_kernel, probability = True))
            elif classifier == "PCA_LDA":
                clf = make_pipeline(StandardScaler(), PCA(n_components = 2), LinearDiscriminantAnalysis())
            else:
                clf = make_pipeline(StandardScaler(), PCA(n_components = 2), SVC(kernel = SVM_kernel, probability = True))
        
            cv = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_seed)
            
            for train_idx, test_idx in cv.split(vectors, states):

                vec_train, vec_test = vectors[train_idx], vectors[test_idx]
                state_train, _ = states[train_idx], states[test_idx]

                clf_fold = clone(clf)
                clf_fold.fit(vec_train, state_train)

                pred[test_idx] = clf_fold.predict(vec_test)
                fold_proba = clf_fold.predict_proba(vec_test)

                fold_classes = clf_fold.classes_
                for i, cls in enumerate(fold_classes):

                    global_idx = np.where(unique_states == cls)[0][0]
                    proba[test_idx, global_idx] = fold_proba[:, i]

            conf_acc[classifier][cond] = {"pred": pred, "proba": proba}

    return conf_acc


def perm_entropy_test(proba_a, proba_b, num_perms, tol, random_seed):

    rng = np.random.default_rng(random_seed)

    entropy_a = -np.sum(proba_a * np.log(proba_a + tol), axis = 1)
    entropy_b = -np.sum(proba_b * np.log(proba_b + tol), axis = 1)
    entropy_diff = np.mean(entropy_b) - np.mean(entropy_a)

    null_dist = []

    entropy = np.concatenate([entropy_a, entropy_b])

    na = len(entropy_a)
    nb = len(entropy_b)

    for _ in range(num_perms):

        perm = rng.permutation(na + nb)
        perm_a = entropy[perm[:na]]
        perm_b = entropy[perm[na:]]

        perm_diff = np.mean(perm_b) - np.mean(perm_a)
        null_dist.append(perm_diff)

    null_dist = np.array(null_dist)
    pval = (np.sum(np.abs(null_dist) >= np.abs(entropy_diff)) + 1) / (num_perms + 1)

    return entropy_diff, pval, null_dist


def perm_decoder_conf(conf_acc, num_perms, tol, random_seed):

    pval_dict = dict()

    for classifier in ["LDA", "SVM", "PCA_LDA", "PCA_SVM"]:

        proba_a = conf_acc[classifier]["least"]["proba"]
        proba_b = conf_acc[classifier]["top"]["proba"]

        entropy_diff, pval, null_dist = perm_entropy_test(proba_a, proba_b, num_perms, tol, random_seed)
        pval_dict[classifier] = {"entropy_diff": entropy_diff, "pval": pval, "null_dist": null_dist}

    return pval_dict


def epoch_decodability(samples_list, log, state_list, SVM_kernel, k, random_seed):

    epoch_acc = {classifier: dict() for classifier in ["SVM"]}

    for classifier in epoch_acc.keys():

        for epoch in log.keys():

            vec_list = log[epoch]["epoch_vectors"]
            _, vectors, _, stim_level_state, stim_level_trial = mt.stim_level_features(samples_list, vec_list, None, state_list)

            vectors = np.array(vectors)
            vectors = vectors - vectors.mean(axis = 0, keepdims = True)

            if classifier == "LDA":
                clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            elif classifier == "SVM":
                clf = make_pipeline(StandardScaler(), SVC(kernel = SVM_kernel))
            elif classifier == "PCA_LDA":
                clf = make_pipeline(StandardScaler(), PCA(n_components = 2), LinearDiscriminantAnalysis())
            else:
                clf = make_pipeline(StandardScaler(), PCA(n_components = 2), SVC(kernel = SVM_kernel))

            cv = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_seed)
            scores_acc = cross_val_score(clf, vectors, stim_level_state, cv = cv, scoring = "accuracy")
            scores_bal_acc = cross_val_score(clf, vectors, stim_level_state, cv = cv, scoring = "balanced_accuracy")

            epoch_acc[classifier][epoch] = {"scores_acc": scores_acc, "scores_bal_acc": scores_bal_acc}

    return epoch_acc


def compute_alignment(log, epoch_acc):

    epochs = sorted(epoch_acc[list(epoch_acc.keys())[0]].keys())

    train_loss = np.array([log[epoch]["train_loss"] for epoch in epochs])
    val_loss = np.array([log[epoch]["val_loss"] for epoch in epochs])

    alignment = {}

    for classifier in epoch_acc.keys():

        mean_acc = np.array([np.mean(epoch_acc[classifier][epoch]["scores_acc"]) for epoch in epochs])
        mean_bal_acc = np.array([np.mean(epoch_acc[classifier][epoch]["scores_bal_acc"]) for epoch in epochs])

        alignment[classifier] = {}

        for acc_name, acc_vals in [("acc", mean_acc), ("bal_acc", mean_bal_acc)]:

            pear_r, pear_p = pearsonr(train_loss, acc_vals)
            spear_r, spear_p = spearmanr(train_loss, acc_vals)

            delta_loss = np.diff(train_loss)
            delta_acc = np.diff(acc_vals)
            d_pear_r, d_pear_p = pearsonr(delta_loss, delta_acc)
            d_spear_r, d_spear_p = spearmanr(delta_loss, delta_acc)

            alignment[classifier][acc_name] = {
                "pearson": {"r": pear_r, "p": pear_p},
                "spearman": {"r": spear_r, "p": spear_p},
                "delta_pearson": {"r": d_pear_r, "p": d_pear_p},
                "delta_spearman": {"r": d_spear_r, "p": d_spear_p},
            }

    return alignment


def calculate_span(vectors, state):

    vectors = vectors.copy()
    vectors = np.asarray(vectors)
    state = np.asarray(state)

    lda = LinearDiscriminantAnalysis(solver = "svd")
    lda.fit(vectors, state)

    W = lda.scalings_[:, :len(np.unique(state)) - 1]
    Q, _ = np.linalg.qr(W)

    return Q


def remove_axis(vectors, Q, t):

    vectors = vectors.clone()
    dtype = vectors.dtype

    proj = vectors @ Q @ Q.T
    vectors_wo_state = (vectors - proj).to(dtype = dtype)

    return vectors_wo_state


def calculate_rand_span(vectors, num_state, random_seed):

    vectors = vectors.copy()
    vectors = np.asarray(vectors)

    rng = np.random.default_rng(random_seed)

    k = num_state - 1
    W = rng.standard_normal((vectors.shape[1], k))
    Q, _ = np.linalg.qr(W)

    return Q


def evaluate_removed_performance(model, loader, vectors, stim_test_state, num_permutations, random_seed):

    model.eval()

    acc_wo_state_list, acc_wo_rand_list = [], []

    Q_state = calculate_span(vectors, stim_test_state)
    num_state = len(np.unique(stim_test_state))
    random_seed_copy = random_seed

    with torch.no_grad():
        for seq_batch, ans_batch in loader:

            pred_wo_state = model.step_wise_rnn(seq_batch, Q_state, corrupt_fn = remove_axis, return_hidden = False)
            pred_wo_state = pred_wo_state.argmax(dim = -1)

            acc_wo_state = (pred_wo_state == ans_batch)
            acc_wo_state_list.extend(acc_wo_state)

        acc_wo_state_num = torch.stack(acc_wo_state_list).float().mean().item()

        for perm in tqdm(range(num_permutations), desc = "perm", leave = False):
        
            acc_wo_rand_perm = []
            Q_rand = calculate_rand_span(vectors, num_state, random_seed_copy)

            for seq_batch, ans_batch in loader:

                pred_wo_rand = model.step_wise_rnn(seq_batch, Q_rand, corrupt_fn = remove_axis, return_hidden = False)
                pred_wo_rand = pred_wo_rand.argmax(dim = -1)

                acc_wo_rand = (pred_wo_rand == ans_batch)
                acc_wo_rand_perm.extend(acc_wo_rand)
                
            random_seed_copy += 1
            acc_wo_rand_list.append(torch.stack(acc_wo_rand_perm).float().mean().item())

        acc_abl = {
            "acc_wo_state_num": acc_wo_state_num,
            "acc_wo_rand_list": acc_wo_rand_list
            }

    return acc_abl

    
def state_centroids(stim_test_vec, stim_test_state):

    unique_states = np.unique(stim_test_state)

    stim_test_vec = np.asarray(stim_test_vec)
    stim_test_state = np.asarray(stim_test_state)

    state_vec_dict = dict()
    state_num_dict = dict()

    for state in unique_states:

        mask = stim_test_state == state
        state_vec = stim_test_vec[mask]

        centroid = state_vec.mean(axis = 0)

        state_vec_dict[state] = centroid
        state_num_dict[state] = len(state_vec)

    return state_vec_dict, state_num_dict


def change_state_hidden(vector, states_args, t):

    state_origin, state_change = states_args
    conf_vector = vector - state_origin[:, t, :] + state_change[:, t, :]

    return conf_vector


def states_comb(unique_states, states_origin, myself):

    num_states = len(unique_states)
    change_batches = []

    states_origin = np.asarray(states_origin)
    unique_states = np.asarray(unique_states)

    if myself:
        shift_range = [0]
    else:
        shift_range = range(1, num_states)

    for shift in shift_range:

        mapping = dict()

        for i, state in enumerate(unique_states):
            changed_state = unique_states[(i + shift) % num_states]
            mapping[state] = changed_state

        states_change = np.vectorize(mapping.get)(states_origin)
        change_batches.append(states_change)

    return change_batches


def states_vec_comb(state_vec_dict, test_state_temp, changed_batch):
    
    states_origin, states_conf = [], []
    test_state_temp = np.asarray(test_state_temp)
    changed_batch = np.asarray(changed_batch)

    for seq_idx in range(test_state_temp.shape[0]):

        vec1, vec2 = [], []

        for stim_idx in range(test_state_temp.shape[1]):
       
            vec1_temp = torch.as_tensor(state_vec_dict[test_state_temp[seq_idx, stim_idx]], dtype = torch.float32)
            vec2_temp = torch.as_tensor(state_vec_dict[changed_batch[seq_idx, stim_idx]], dtype = torch.float32)

            vec1.append(vec1_temp)
            vec2.append(vec2_temp)

        states_origin.append(torch.stack(vec1, dim = 0))
        states_conf.append(torch.stack(vec2, dim = 0))

    states_origin = torch.stack(states_origin, dim = 0)
    states_conf = torch.stack(states_conf, dim = 0)

    return states_origin, states_conf


def get_correct_ans(task, stimuli_encoder, ans_encoder, seq_batch, changed_batch, ans_shape, ans_dtype):
    
    seq_batch = np.asarray(seq_batch)
    correct_ans = torch.empty(ans_shape, dtype = ans_dtype)

    for seq_idx in range(seq_batch.shape[0]):
        for stim_idx in range(seq_batch.shape[1]):
            
            state = changed_batch[seq_idx, stim_idx]
            stim = stimuli_encoder.inverse_transform(seq_batch[seq_idx, stim_idx].reshape(1, -1))
            stim = stim[0, 0]

            ans = task.action[state][stim]
            ans = ans_encoder.transform([ans])[0]

            correct_ans[seq_idx, stim_idx] = ans

    return correct_ans


def evaluate_conf_performance(task, model, stimuli_encoder, ans_encoder, state_vec_dict, test_loader, test_state, test_ans):

    model.eval()

    unique_states = np.unique(test_state)

    acc_original_ifchanges, acc_original_ifstays = [], []
    acc_confusion_ifchanges, acc_confusion_ifstays = [], []

    def evaluate_changed_batches(seq_batch, test_state_temp, test_ans_temp, ans_shape, ans_dtype, myself):

        acc_confused_list, acc_supposed_list = [], []
        changed_batches = states_comb(unique_states, test_state_temp, myself)

        for changed_batch in changed_batches:

            states_args = states_vec_comb(state_vec_dict, test_state_temp, changed_batch)
            correct_ans = get_correct_ans(task, stimuli_encoder, ans_encoder, seq_batch, changed_batch, ans_shape, ans_dtype)
            
            pred = model.step_wise_rnn(seq_batch, states_args, corrupt_fn = change_state_hidden, return_hidden = False)
            pred = pred.argmax(dim = -1)
            
            acc_confused = (pred == correct_ans)
            acc_supposed = (pred == test_ans_temp)

            acc_confused_list.append(acc_confused.reshape(-1))
            acc_supposed_list.append(acc_supposed.reshape(-1))

        return acc_confused_list, acc_supposed_list

    with torch.no_grad():

        state_idx = 0

        for seq_batch, ans_batch in tqdm(test_loader, desc = "axis_conf", leave = False):

            ans_dtype = ans_batch.dtype
            ans_shape = ans_batch.shape
            batch_size = len(seq_batch)

            test_state_temp = test_state[state_idx : state_idx + batch_size]
            test_ans_temp = np.array([ans_encoder.transform(ans) for ans in test_ans[state_idx : state_idx + batch_size]])
            test_ans_temp = torch.tensor(test_ans_temp, dtype = torch.long)
            
            state_idx += batch_size
            
            acc_list = evaluate_changed_batches(seq_batch, test_state_temp, test_ans_temp, ans_shape, ans_dtype, myself = False)
            
            acc_confusion_ifchanges += acc_list[0]
            acc_confusion_ifstays += acc_list[1]
            
            acc_list = evaluate_changed_batches(seq_batch, test_state_temp, test_ans_temp, ans_shape, ans_dtype, myself = True)
            
            acc_original_ifchanges += acc_list[0]
            acc_original_ifstays += acc_list[1]

    acc_conf = {
        "acc_confusion_ifchanges": torch.cat(acc_confusion_ifchanges).float().numpy(),
        "acc_confusion_ifstays": torch.cat(acc_confusion_ifstays).float().numpy(),
        "acc_original_ifchanges": torch.cat(acc_original_ifchanges).float().numpy(),
        "acc_original_ifstays": torch.cat(acc_original_ifstays).float().numpy()
    }

    return acc_conf