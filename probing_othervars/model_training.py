import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers = 1):
        super(RNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)  
        out, _ = self.rnn(x, h0) 
        # last_hidden = out[:, -1, :]
        # out = self.fc(last_hidden)
        out = self.fc(out)
        return out

    def extract_hidden(self, x):
        self.eval() 
        with torch.no_grad():
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
            out, _ = self.rnn(x, h0)
        #     last_hidden = out[:, -1, :]
        # return last_hidden
        return out

    
def generate_loader(samples_encoded, ans_encoded, val_ratio, batch_size, random_seed, if_train):
    
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    samples_encoded = np.array(samples_encoded, dtype = np.float32)
    ans_encoded = np.array(ans_encoded, dtype = np.int64)
    
    if len(samples_encoded.shape) == 2:
        samples_encoded = np.expand_dims(samples_encoded, axis = -1)    # the RNN model is set to handle 2D inputs, 3D loader
    
    if if_train:
        samples_train, samples_val, ans_train, ans_val = train_test_split(samples_encoded, ans_encoded,
                                                                          test_size = val_ratio, random_state = random_seed)
    
        train_data = torch.utils.data.TensorDataset(torch.tensor(samples_train), torch.tensor(ans_train))
        val_data = torch.utils.data.TensorDataset(torch.tensor(samples_val), torch.tensor(ans_val))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, generator = generator)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size)
        
        return train_loader, val_loader
    
    else:
        test_data = torch.utils.data.TensorDataset(torch.tensor(samples_encoded), torch.tensor(ans_encoded))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
        
        return test_loader
    

def train_model(train_loader, val_loader, test_loader, task, learning_rate, layer_size, regularization, random_seed, get_vectors):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    input_size = len(task.stimuli_type)
    output_size = len(task.action_type)

    model = RNN(input_size, layer_size, output_size)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = regularization)
    
    val_acc = 0
    epoch = 0
    log = dict()
    reach_threshold = 0
    
    while reach_threshold < 3:
        
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        epoch += 1

        for seq_batch, ans_batch in train_loader:
            optimizer.zero_grad()
            probs = model(seq_batch)
            loss = criterion(probs.reshape(-1, probs.shape[-1]), ans_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = torch.argmax(probs, dim = 2)
            correct_train += (predicted == ans_batch).sum().item()
            total_train += ans_batch.numel()
        
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for seq_batch, ans_batch in val_loader:
                probs = model(seq_batch)
                loss = criterion(probs.reshape(-1, probs.shape[-1]), ans_batch.reshape(-1))
                val_loss += loss.item()

                predicted = torch.argmax(probs, dim = 2)
                correct_val += (predicted == ans_batch).sum().item()
                total_val += ans_batch.numel()

        val_acc = correct_val / total_val
        train_acc = correct_train / total_train

        if val_acc >= 0.99:
            reach_threshold += 1
        else:
            reach_threshold = 0

        if get_vectors:
            epoch_vectors = extract_features(model, test_loader)
        else:
            epoch_vectors = None
        
        if epoch > 0 and epoch % 10 == 9:
            print(f"-- Epoch {epoch + 1} | "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc:.4f}")
        
        log[epoch] = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      "train_acc": train_acc, "val_acc": val_acc, "timepoint": time.time(),
                      "epoch_vectors": epoch_vectors, "get_vectors": get_vectors}

    print(f"-- Early Stopped for Validation Acc >= 0.99 at epoch {epoch}")
           
    return model, log


def extract_features(model, loader):
    
    vec_list = []

    model.eval()
    for samples_batch, ans_batch in loader:
        with torch.no_grad():
            feat_batch = model.extract_hidden(samples_batch)
        for feature in feat_batch: 
            vec_list.append(feature.tolist()) 
            
    return vec_list


def stim_level_features(sample_list, vec_list, ans_list, state_list):

    stim_level_sample = []
    stim_level_vec = []
    stim_level_ans = []
    stim_level_state = []
    stim_level_trial = []
    
    for sample_idx in range(len(sample_list)):

        sample = sample_list[sample_idx]

        for stim_idx in range(len(sample)):
            
            stim_level_sample.append(sample[stim_idx])
            
            if vec_list is not None:
                stim_level_vec.append(vec_list[sample_idx][stim_idx])
            if ans_list is not None:
                stim_level_ans.append(ans_list[sample_idx][stim_idx])
            if state_list is not None:
                stim_level_state.append(state_list[sample_idx][stim_idx])
            stim_level_trial.append(str(stim_idx))
            
    return stim_level_sample, stim_level_vec, stim_level_ans, stim_level_state, stim_level_trial