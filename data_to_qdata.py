import torch
from mi_quantum.quantum.pennylane_backend import QuantumLayer
from mi_quantum import data as Data
from mi_quantum.quantum.quanvolution import QuantumKernel, QuantumConv2D
import os
import numpy as np
from tqdm import tqdm

graphs = {'king_2_ancilla': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                     [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4],
                     [4, 9], [0, 9], [2, 9], [8, 9], [6, 9],
                     [1, 10], [5, 10], [7, 10], [3, 10], [4, 10]],
          'king_1_ancilla': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                     [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4],
                     [4, 9], [0, 9], [2, 9], [8, 9], [6, 9]]}

p = {
    'num_qubits': 9,
    'ancilla' : 1,
    'entangle': True,
    'trainBool': False,
    'connectivity': graphs['king_1_ancilla'],
    'patch_size': 3,
    'stride' : 1,
    'padding': 1,
    'channels_last' : False,
    'batch_size' : 64,
    'cat_original' : True
}

# Parameters
save_dir = "DERMA/kernel_3x3_derma_" + '1_ancilla_cat_original'
os.makedirs(save_dir, exist_ok=True)

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Quanvolution = QuantumConv2D(patch_size=p['patch_size'], stride=p['stride'], padding=p['padding'], channels_last = p['channels_last'],
                             graph = p['connectivity'], channels_out= [-1], ancilla = p['ancilla']).to(device)

train, val, test, shape = Data.get_medmnist_dataloaders(pixel = 28, data_flag = 'dermamnist', batch_size=p['batch_size'], num_workers=4, pin_memory=True)

def preprocess_and_save(dataloader, quanv, split_name):
    quanv.eval()
    all_data = []
    all_labels = []

    print(f"Processing {split_name}...")
    with torch.no_grad():
        for x, y, ind in tqdm(dataloader, desc=f"{split_name}"):
            x = x.to(device)
            y = y.to(device)
            q_out = torch.cat( [x, quanv(x)], dim = 1) if p['cat_original'] else quanv(x)
            all_data.append(q_out.cpu())
            all_labels.append(y.cpu())

    # Stack and save
    X = torch.cat(all_data, dim=0).numpy()
    Y = torch.cat(all_labels, dim=0).numpy()

    print(f"shape after quantum conv: {X.shape}")
    
    np.save(os.path.join(save_dir, f"q_{split_name}_images.npy"), X)
    np.save(os.path.join(save_dir, f"q_{split_name}_labels.npy"), Y)
    print(f"Saved {split_name} to {save_dir}")

# Run for each split
preprocess_and_save(train, Quanvolution, "train")
preprocess_and_save(val, Quanvolution,"val")
preprocess_and_save(test, Quanvolution,"test")



















