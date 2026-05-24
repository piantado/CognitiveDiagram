import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import model_training as mt

def idx_to_stimuli(idx, stimuli_type, seq_len):
    
    stimuli = [0] * seq_len
    for pos in range(seq_len - 1, -1, -1):
        idx, remainder = divmod(idx, len(stimuli_type))
        stimuli[pos] = stimuli_type[remainder]
    
    return list(stimuli)

def generate_samples(task, seq_len, num_train_trials, num_test_trials, random_seed):

    num_train_samples = math.ceil(num_train_trials / seq_len)
    num_test_samples = math.ceil(num_test_trials / seq_len)
    num_samples = num_train_samples + num_test_samples

    num_possible_samples = len(task.stimuli_type) ** seq_len
    
    assert num_possible_samples >= num_samples
    
    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(num_possible_samples, size = num_samples, replace = False)
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
        ans_list.append(action)
        state_list.append(state)    # both becomes a nested list
    
    return ans_list, state_list
    
def generate_encoder(task):

    stimuli_encoder = OneHotEncoder(sparse_output = False)
    stimuli_encoder.fit(np.array(task.stimuli_type).reshape(-1, 1))
    
    ans_encoder = LabelEncoder()
    ans_encoder.fit(task.action_type)

    return stimuli_encoder, ans_encoder

def encode_samples(samples, stimuli_encoder):
    
    samples_encoded = [stimuli_encoder.transform(np.array(sample).reshape(-1, 1)) for sample in samples]
    
    return samples_encoded

def encode_answers(ans_list, ans_encoder):
    
    ans_encoded = [ans_encoder.transform(np.array(ans)) for ans in ans_list]
    
    return ans_encoded

def generate_loaders(task, task_type, seq_len, num_train_trials, num_test_trials, random_seed, val_ratio, batch_size):
 
    train_samples, test_samples = generate_samples(task, seq_len, num_train_trials, num_test_trials, random_seed)   # nested list form
    train_ans, _ = generate_ans_state(task, train_samples)  # nested list
    test_ans, test_state = generate_ans_state(task, test_samples)   # nested list

    stimuli_encoder, ans_encoder = generate_encoder(task)

    train_samples_encoded = encode_samples(train_samples, stimuli_encoder)
    train_ans_encoded = encode_answers(train_ans, ans_encoder)
    
    test_samples_encoded = encode_samples(test_samples, stimuli_encoder)
    test_ans_encoded = encode_answers(test_ans, ans_encoder)
   
    train_loader, val_loader = mt.generate_loader(train_samples_encoded, train_ans_encoded, val_ratio, batch_size, random_seed, if_train = True)
    test_loader = mt.generate_loader(test_samples_encoded, test_ans_encoded, val_ratio, batch_size, random_seed, if_train = False)

    return train_samples, test_samples, train_ans, test_ans, test_state, stimuli_encoder, ans_encoder, train_loader, val_loader, test_loader