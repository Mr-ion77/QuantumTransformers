import torch
import torch.nn as nn
import mi_quantum as qpctorch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from mi_quantum.data import split_dataset_by_label, relabel_dataset
from mi_quantum.quantum.double_step_classification_vit import DbStpClssViT
from torch.utils.data import ConcatDataset

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

B = 256
N1 = 100  # Number of epochs
N2 = 100  # Number of epochs for the second step

# Hyperparams
p1 = {
    'learning_rate': 0.01, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.3, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' = True, 'num_head': 4, 'num_transf': 1, 'mlp_size': 6, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'none', 'entangle': True,
    'connectivity': 'david_star', 'RD': 1, 'patience': -1, 'scheduler_factor': 0.999, 'q_stride': 1  # No early stopping
}

p2 = {
    'learning_rate': 0.01, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.3, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' = True, 'num_head': 4, 'num_transf': 2, 'mlp_size': 6, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter', 'RD': 1, 
    'paralel': 2, 'patience': -1, 'scheduler_factor': 1.0, 'q_stride': 1  # No early stopping
}

columns = [
    # 'idx', 'learning_rate', 'hidden_size', 'dropout', 'num_head', 'num_transf', 'mlp_size', 'patch_size',
    # 'weight_decay', 'attention_selection', 'entangle', 'penny_or_kipu', 'RD', 'convolutional', 'paralel', 
    'test_auc1', 'test_acc1',  'space', 'test_auc2', 'test_acc2',  'spacE', # 'val_auc1', 'val_acc1', 'val_auc2', 'val_acc2',
    'test_auc_last', 'test_acc_last', 'thres'
]

channels_last = False

df = pd.DataFrame(columns=columns)
df.to_csv('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv', mode='a', header=True, index=False)

# Grid search loop
for idx in range(50):
    print(f"\n\nPoint {idx}")

    save_path = Path(f"../QTransformer_Results_and_Datasets/autoenformer_results/current_results/grid_search{idx}")
    save_path.mkdir(parents=True, exist_ok=True)

    print('\n')

    # Load data
    train_dl, val_dl, test_dl, shape = qpctorch.data.get_medmnist_dataloaders(
        pixel=28, data_flag='dermamnist', batch_size=B, num_workers=4, pin_memory=True
    )

    num_channels = shape[-1] if channels_last else shape[0]
    img_size = shape[1]  # Assuming square images

    # Split datasets by label
# Model
    model1 = qpctorch.quantum.vit.AutoEnformer(
        img_size=img_size, num_channels=num_channels,   # set num_classes as needed
        patch_size=p1['patch_size'], hidden_size=p1['hidden_size'], num_heads=p1['num_head'],
        num_transformer_blocks=p1['num_transf'], attention_selection=p1['attention_selection'],
        mlp_hidden_size=p1['mlp_size'], dropout=p1['dropout'], channels_last=False
    )

    
    # Train
    test_mse, val_mse, val_auc1, val_acc1, params1 = qpctorch.training.train_and_evaluate(
        model1, train_dl, val_dl, test_dl, num_classes=7,
        learning_rate=p1['learning_rate'], num_epochs=N1, device=device, mapping=False,
        res_folder=str(save_path), hidden_size=p1['hidden_size'], dropout=p1['dropout'],
        num_heads=p1['num_head'], patch_size=p1['patch_size'], num_transf=p1['num_transf'],
        mlp=p1['mlp_size'], wd=p1['weight_decay'], patience= p1['patience'], scheduler_factor= p1['scheduler_factor'], autoencoder=True
    )

    print(f"\nAutoencoder training completed succesfully.\nTest MSE (first step): {test_mse:.2f}")

    # Create second model for the second step
    model2 = qpctorch.quantum.VisionTransformer(
        img_size=img_size, num_channels=num_channels, num_classes=7,
        patch_size= p2['patch_size'], hidden_size=p2['hidden_size'], num_heads=p2['num_head'],
        num_transformer_blocks=p2['num_transf'], attention_selection=p2['attention_selection'],
        mlp_hidden_size=p2['mlp_size'], dropout=p2['dropout'],
        channels_last=False, entangle=p2['entangle'], q_stride = p2['q_stride'], 
        RD=p2['RD'], paralel=p2['paralel']
    )

    # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
    DataLoaders = [train_dl, val_dl, test_dl]
    LatentDatasetsTensors = []
    QuantumLayer = qpctorch.quantum.pennylane_backend.QuantumLayer(num_qubits = p1['mlp_size'], entangle = p1['entangle'], graph = p1['connectivity']) if p1['quantum'] else torch.nn.Identity()
    for dl in DataLoaders:
        all_latents = []
        all_labels = []
        all_indices = []
        for images, labels, indices in dl:
            images = images.to(device)
            with torch.no_grad():
                latent_representations = QuantumLayer( model1.get_latent_representation(images) )
            latent_representations = latent_representations.cpu()

            all_latents.extend( latent_representations )
            all_labels.extend( labels )

        all_latents = torch.stack(all_latents)
        all_labels = torch.tensor(all_labels)


        LatentDatasetsTensors.append( zip(list(all_latents,all_labels) ) )

    latent_train_dl, latent_val_dl, latent_test_dl = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                        tensors = LatentDatasetTensors, transforms={'train': q_train_transforms, 'val': q_valid_transforms, 'test': q_valid_transforms}
                                        )

    print('\nTraining second model: classifier ViT on latent representations\n')

    # Train second model
    test_auc2, test_acc2, val_auc2, val_acc2, params2 = qpctorch.training.train_and_evaluate(
        model2, train_others_dl, val_others_dl, test_others_dl, num_classes=6,
        learning_rate=p2['learning_rate'], num_epochs=N, device=device, mapping=False,
        res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=p2['dropout'],
        num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
        mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor']
    )


    
    # Save results
    row = {
        'idx': idx,
            'test_mse': test_mse, 'val_mse': val_mse, 
            'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc
    }

    pd.DataFrame([row], columns=columns).to_csv(
        '../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv', mode='a', header=False, index=False
    )
