import torch
import torch.nn as nn
import mi_quantum as qpctorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantum_transformers
from PIL import Image
import shutil
from pathlib import Path

carpeta = "derma_qc"


B = 512
N = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

auc_list_test = []
acc_list_test = []
auc_list_val = []
acc_list_val = []
hyp_list = []

lrs = [0.001]
hidden_sizes = [100]
dropouts = [0.5]
num_heads = [4]
num_transformer_blocks = [1]
mlp_size = [400]
patch_size = [4]
wds = [1e-7]

columns = ['idx', 'learning_rate', 'hidden_size', 'dropout', 'num_head', 'num_transf', 'mlp_size', 'patch_size', 'weight_decay', 'test_auc', 'test_acc', 'val_auc', 'val_acc']

df = pd.DataFrame(columns=columns)
df.to_csv(f'derma_results/low_res/{carpeta}/results_grid_search_qc.csv', mode='a', header=True, index=False)

for idx in range(50):
    print(f"Point {idx}")
    for learning_rate in lrs:
        for hidden_size in hidden_sizes:
            for dropout in dropouts:
                for num_head in num_heads:
                    for num_transf in num_transformer_blocks:
                        for mlp in mlp_size:
                            for patch in patch_size:
                                for wd in wds:

                                    ## Crear directorio para almacenar los datos de los 30 modelos
                                    directorio = Path(f"derma_results/low_res/{carpeta}/grid_search{idx}")
                                    directorio.mkdir(parents=True, exist_ok=True) 
                
                                    ## Entrenar el modelo
                                    train_dataloader, valid_dataloader, test_dataloader = qpctorch.data.get_quantum_medmnist_dataloaders(batch_size=B, num_workers=4, pin_memory=True)

                                    model = qpctorch.classical.VisionTransformer(data_type="quantum_embedd", img_size=26, num_channels=27, num_classes=7, patch_size=patch, hidden_size=hidden_size, num_heads=num_head, num_transformer_blocks=num_transf, mlp_hidden_size=mlp, dropout=dropout, channels_last=True)
                
                                    test_auc, test_acc, val_auc, val_acc =qpctorch.training.train_and_evaluate(model, train_dataloader, valid_dataloader, test_dataloader, num_classes=7, learning_rate=learning_rate, num_epochs=N, device=device, mapping=False, res_folder=f"derma_results/low_res/{carpeta}/grid_search{idx}", hidden_size=hidden_size, dropout=dropout, num_heads=num_head, patch_size=patch,  num_transf=num_transf, mlp=mlp, wd=wd)
                
                                    hyp_list.append((B, learning_rate, hidden_size, dropout, num_head, num_transf, mlp, patch))
                                    auc_list_test.append(test_auc)
                                    acc_list_test.append(test_acc)
                                    auc_list_val.append(val_auc)
                                    acc_list_val.append(val_acc)
                                    
                                    row = {
                                        'idx': idx,
                                        'learning_rate': learning_rate,
                                        'hidden_size': hidden_size,
                                        'dropout': dropout,
                                        'num_head': num_head,
                                        'num_transf': num_transf,
                                        'mlp_size': mlp,
                                        'patch_size': patch,
                                        'weight_decay': wd,
                                        'test_auc': test_auc,
                                        'test_acc': test_acc,
                                        'val_auc': val_auc,
                                        'val_acc': val_acc
                                    }
                    
                                    df = pd.DataFrame([row], columns=columns)
                                    df.to_csv(f'derma_results/low_res/{carpeta}/results_grid_search_qc.csv', mode='a', header=False, index=False)
