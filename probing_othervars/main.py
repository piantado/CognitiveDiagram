import warnings
warnings.filterwarnings("ignore", category = UserWarning)
import os
import json
import csv
import torch
import numpy as np
import joblib
import building_cognitive_diagram as bcd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sample_generation as sg
import model_training as mt
import analysis_functions as af
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

### Variables ###
task_type = "FiniteList"    # e.g. "WCST", "Nback", "MaxNum", "FiniteSet", "FiniteList", "FiniteStack"
if_minimal = False

seq_len = 10    # the length of each stimuli sequence
num_train_trials = 50000
num_test_trials = 10000
random_seed = 42

hidden_layer_size = [16, 32, 64]
L2_lambda = [0, 1e-5, 1e-4, 1e-3]
target_cond = None	# or {"size": 16, "l2": 1e-3}

batch_size = 4
learning_rate = 0.001
val_ratio = 0.2    # validation ratio

tol = 1e-3

### Directories ###
results_path = os.path.join('results', task_type + f"_random{random_seed}")
os.makedirs(results_path, exist_ok = True)

### Main Processing ###
if __name__ == "__main__":

	print("Note: If you want to change the task variables, refer to building_cognitive_diagram.py")
	print(f"task_type: {task_type}")

	# save specifications
	metadata = {
		"task_type": task_type, "seq_len": seq_len, "num_train_trials": num_train_trials, "num_test_trials": num_test_trials,
		"random_seed": random_seed, "hidden_layer_size": hidden_layer_size, "L2_lambda": L2_lambda, "batch_size": batch_size,
		"learning_rate": learning_rate, "val_ratio": val_ratio, "tol": tol, "if_minimal": if_minimal
		}
	with open(os.path.join(results_path, f"{task_type}_metadata.json"), "w") as f:
		json.dump(metadata, f, indent = 2)

	# build a cognitive diagram
	task = bcd.build_cognitive_diagram(task_type, if_minimal)
	with open(os.path.join(results_path, f"{task_type}_task.json"), "w") as f:
		json.dump(task.to_dict(), f, indent = 2)

	# generate samples
	print("Sample Generation")
	train_samples, test_samples, train_ans, test_ans, test_state, stimuli_encoder, ans_encoder, train_loader, val_loader, test_loader = sg.generate_loaders(task, task_type, seq_len, num_train_trials, num_test_trials, random_seed, val_ratio, batch_size)

	print(f"- train_samples.shape: {np.array(train_samples).shape} | train_ans.shape: {np.array(train_ans).shape}")
	print(f"- test_samples.shape: {np.array(test_samples).shape} | test_ans.shape: {np.array(test_ans).shape} | test_state: {np.array(test_state).shape}")
	np.savez_compressed(os.path.join(results_path, f"{task_type}_test_data.npz"), 
                        train_samples = np.array(train_samples),
						train_ans = np.array(train_ans),
						test_samples = np.array(test_samples),
						test_ans = np.array(test_ans),
						test_state = np.array(test_state)
						)

	joblib.dump({
				"stimuli_encoder": stimuli_encoder,
				"ans_encoder": ans_encoder
				}, os.path.join(results_path, f"{task_type}_encoders.joblib"))

	
	# model training & state decoding
	print("Model Training")
	vectors_all_conds = {size: dict() for size in hidden_layer_size}
	acc_all_conds = {size: dict() for size in hidden_layer_size}

	best_acc_conds = {
		"acc_lin": {
			"mean_acc": -np.inf, "size": None, "l2": None, "values": None},
		"acc_rbf":{
			"mean_acc": -np.inf, "size": None, "l2": None, "values": None}
		}

	stim_level_sample, _, stim_level_ans, stim_level_state, _ = mt.stim_level_features(test_samples, None, test_ans, test_state)

	clf_lin = make_pipeline(StandardScaler(), SVC(kernel = "linear", tol = tol))
	clf_rbf = make_pipeline(StandardScaler(), SVC(kernel = "rbf", tol = tol))
	cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = random_seed)

	for size in hidden_layer_size:
		for l2 in L2_lambda:

			print(f"- size: {size} | l2: {l2}")
			model, log = mt.train_model(train_loader, val_loader, test_loader, task, learning_rate, 
							   layer_size = size, regularization = l2, random_seed = random_seed, get_vectors = True)
			test_vectors = mt.extract_features(model, test_loader)

			_, stim_level_vec, _, _, _ = mt.stim_level_features(test_samples, test_vectors, None, None)
			
			vectors_all_conds[size][l2] = stim_level_vec

			acc_lin = cross_val_score(clf_lin, stim_level_vec, stim_level_state, cv = cv, scoring = "accuracy")
			acc_rbf = cross_val_score(clf_rbf, stim_level_vec, stim_level_state, cv = cv, scoring = "accuracy")

			acc_all_conds[size][l2] = {"acc_lin": acc_lin, "acc_rbf": acc_rbf}

			if (target_cond is None) or ((target_cond["size"] == size) and (target_cond["l2"] == l2)):
				if best_acc_conds["acc_lin"]["mean_acc"] < np.mean(acc_lin):
					best_acc_conds["acc_lin"] = {"mean_acc": np.mean(acc_lin), "size": size, "l2": l2, "values": acc_lin}
					model_lin = model
					log_lin = log
				if best_acc_conds["acc_rbf"]["mean_acc"] < np.mean(acc_rbf):
					best_acc_conds["acc_rbf"] = {"mean_acc": np.mean(acc_rbf), "size": size, "l2": l2, "values": acc_rbf}
					model_rbf = model
					log_rbf = log

	vectors_all_conds = {str(key): val for key, val in vectors_all_conds.items()}
	acc_all_conds = {str(key): val for key, val in acc_all_conds.items()}

	np.savez_compressed(os.path.join(results_path, f"{task_type}_vectors_all.npz"), **vectors_all_conds)
	np.savez_compressed(os.path.join(results_path, f"{task_type}_acc_all.npz"), **acc_all_conds)

	print(f"- acc_lin | mean_acc: {best_acc_conds['acc_lin']['mean_acc']:.4f} | cond: {best_acc_conds['acc_lin']['size']}, {best_acc_conds['acc_lin']['l2']}")
	print(f"- acc_rbf | mean_acc: {best_acc_conds['acc_rbf']['mean_acc']:.4f} | cond: {best_acc_conds['acc_rbf']['size']}, {best_acc_conds['acc_rbf']['l2']}")

	if best_acc_conds["acc_lin"]["mean_acc"] > best_acc_conds["acc_rbf"]["mean_acc"]:
		vectors = vectors_all_conds[str(best_acc_conds["acc_lin"]["size"])][best_acc_conds["acc_lin"]["l2"]]
		model, log = model_lin, log_lin
		size, l2 = best_acc_conds["acc_lin"]["size"], best_acc_conds["acc_lin"]["l2"]
		SVM_kernel = "linear"
	else:
		vectors = vectors_all_conds[str(best_acc_conds["acc_rbf"]["size"])][best_acc_conds["acc_rbf"]["l2"]]
		model, log = model_rbf, log_rbf
		size, l2 = best_acc_conds["acc_rbf"]["size"], best_acc_conds["acc_rbf"]["l2"]
		SVM_kernel = "rbf"
	print(f"- {SVM_kernel} selected")

	torch.save(model.state_dict(), os.path.join(results_path, f"{task_type}_model_layer{size}_decay{l2}.pt"))
	joblib.dump(log, os.path.join(results_path, f"{task_type}_train_layer{size}_decay{l2}_log.npz"))

	# Analysis 1
	print("Analysis 1")
	readout_acc = af.readout_decodability(vectors, stim_level_state, model, SVM_kernel, k = 10, random_state = random_seed, tol = 1e-12)
	readout_sep = af.readout_separability(model, vectors, stim_level_state, tol = 1e-12)

	np.savez_compressed(os.path.join(results_path, f"{task_type}_acc_readout.npz"), **readout_acc)
	np.savez_compressed(os.path.join(results_path, f"{task_type}_separability_readout.npz"), **readout_sep)

	# Analysis 2
	print("Analysis 2")
	other_vars = af.get_other_vars(test_samples, test_ans, stim_level_state)
	clustering_metrics = af.get_clustering(vectors, other_vars, len(np.unique(list(task.states()))), random_seed)

	np.savez_compressed(os.path.join(results_path, f"{task_type}_othervars.npz"), **other_vars)
	np.savez_compressed(os.path.join(results_path, f"{task_type}_clustering_metrics.npz"), **clustering_metrics)

	# Analysis 3
	print("Analysis 3")
	conf_dict = af.conf_samples(model, vectors, test_loader, stim_level_state, percentage = 0.3)
	conf_acc = af.decoder_conf(conf_dict, stim_level_state, SVM_kernel, k = 10, random_seed = random_seed)
	pval_dict = af.perm_decoder_conf(conf_acc, num_perms = 1000, tol = 1e-12, random_seed = random_seed)

	np.savez_compressed(os.path.join(results_path, f"{task_type}_conf_dict.npz"), **conf_dict)
	np.savez_compressed(os.path.join(results_path, f"{task_type}_conf_acc.npz"), **conf_acc)
	np.savez_compressed(os.path.join(results_path, f"{task_type}_pval_dict.npz"), **pval_dict)

	# Analysis 4
	print("Analysis 4")
	epoch_acc = af.epoch_decodability(test_samples, log, test_state, SVM_kernel, k = 10, random_seed = random_seed)
	alignment = af.compute_alignment(log, epoch_acc)

	np.savez_compressed(os.path.join(results_path, f"{task_type}_epoch_acc.npz"), **epoch_acc)
	np.savez_compressed(os.path.join(results_path, f"{task_type}_alignment.npz"), **alignment)

	print("Analyses finished successfully!")