# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class NBackRNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers = 1):
        super(NBackRNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)  
        out, _ = self.rnn(x, h0) 
        last_hidden = out[:, -1, :]
        out = self.fc(last_hidden)
        return out

    def extract_hidden(self, x):
        self.eval() 
        with torch.no_grad():
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
            out, _ = self.rnn(x, h0)
            last_hidden = out[:, -1, :]
        return last_hidden
    
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
    
def train_model(train_loader, val_loader, input_size, hidden_layer_size, output_size, learning_rate):
    
    model = NBackRNN(input_size, hidden_layer_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    val_acc = 0
    epoch = 0
    log = []
    
    while val_acc < 0.999:
        
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        epoch += 1

        for seq_batch, ans_batch in train_loader:
            optimizer.zero_grad()
            probs = model(seq_batch)
            loss = criterion(probs, ans_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = torch.argmax(probs, dim = 1)
            correct_train += (predicted == ans_batch).sum().item()
            total_train += ans_batch.size(0)
        
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for seq_batch, ans_batch in val_loader:
                probs = model(seq_batch)
                loss = criterion(probs, ans_batch)
                val_loss += loss.item()

                predicted = torch.argmax(probs, dim = 1)
                correct_val += (predicted == ans_batch).sum().item()
                total_val += ans_batch.size(0)

        val_acc = correct_val / total_val
        train_acc = correct_train / total_train
        
        if epoch > 0 and epoch % 10 == 9:
            print(f"Epoch {epoch + 1} | "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc:.4f}")
        
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "train_acc": train_acc, "val_acc": val_acc, "timepoint": time.time()})

    print(f"Early Stopped for Validation Acc >= 0.999 at epoch {epoch}")
           
    return model, log


def extract_features(model, loader):
    
    vec_list = []
    
    for samples_batch, ans_batch in loader:
        with torch.no_grad():
            feat_batch = model.extract_hidden(samples_batch)  # shape: (batch_size, hidden_size)
        for feature in feat_batch:
            vec_list.append(feature.tolist())
            
    return vec_list