# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def idx_to_stimuli(idx, stimuli_type, seq_len):
    
    stimuli = [0] * seq_len
    for pos in range(seq_len - 1, -1, -1):
        idx, remainder = divmod(idx, len(stimuli_type))
        stimuli[pos] = stimuli_type[remainder]
    
    return list(stimuli)

def generate_samples(task, seq_len, num_train_samples, num_test_samples, random_seed):
    
    total_num_samples = num_train_samples + num_test_samples
    num_possible_samples = len(task.stimuli_type) ** seq_len
    
    assert num_possible_samples >= total_num_samples
    
    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(num_possible_samples, size = total_num_samples, replace = False)
    train_idx = sample_idx[:num_train_samples]
    test_idx = sample_idx[num_train_samples:]
    
    train_samples = [idx_to_stimuli(int(idx), task.stimuli_type, seq_len) for idx in train_idx]
    test_samples = [idx_to_stimuli(int(idx), task.stimuli_type, seq_len) for idx in test_idx]
    
    return train_samples, test_samples

def generate_ans_state(task, samples):
    
    ans_list = []
    state_list = []
    
    for sample in samples:
        action, state = task.seqActions(sample)
        ans_list.append(action[-1])
        state_list.append(state)
    
    return ans_list, state_list

def generate_ans_state_test(task, samples):
    
    ans_list = []
    state_list = []
    
    for sample in samples:
        action, state = task.seqActions(sample)
        ans_list.append(action[-1])
        state_list.append(state)
    
    return ans_list, state_list
    
def generate_enc(task):

    stimuli_enc = OneHotEncoder(sparse_output = False)
    stimuli_enc.fit(np.array(task.stimuli_type).reshape(-1, 1))
    
    ans_enc = LabelEncoder()
    ans_enc.fit(task.action_type)

    return stimuli_enc, ans_enc

def encode_samples(samples, stimuli_enc):
    
    samples_encoded = [stimuli_enc.transform(np.array(sample).reshape(-1, 1)) for sample in samples]
    
    return samples_encoded

def encode_answers(ans_list, ans_enc):
    
    ans_encoded = ans_enc.transform(ans_list)
    
    return ans_encoded
