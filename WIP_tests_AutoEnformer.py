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
import matplotlib.pyplot as plt
import json
# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

B = 256
N1 = 100  # Number of epochs
N2 = 150  # Number of epochs for the second step

# Hyperparams
p1 = {
    'learning_rate': 0.001, 'hidden_size': 49*3, 'dropout': {'embedding_attn': 0.2, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' : True, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 1, 'mlp_size': 9, 'patch_size': 7, 'weight_decay': 1e-7, 'attention_selection': 'none', 'entangle': True,
    'connectivity': 'king', 'RD': 1, 'patience': -1, 'scheduler_factor': 0.999, 'q_stride': 1 , 'RBF_similarity': 'none'  # No early stopping
}

p2 = {
    'learning_rate': 0.001, 'hidden_size': 49*3, 'dropout': {'embedding_attn': 0.1, 'after_attn': 0.05, 'feedforward': 0.05, 'embedding_pos': 0.05},
    'quantum' : True, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 9, 'patch_size': 7, 'weight_decay': 1e-7, 'attention_selection': 'filter', 'RD': 1, 
    'paralel': 2, 'patience': -1, 'scheduler_factor': 0.9995, 'q_stride': 1, 'RBF_similarity': 'none'  # No early stopping
}

# Save dictionary with all the hyperparameters and results in a json file

with open('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/hyperparameters.json', 'w') as f:
    f.write('\nHyperparameters for Autoencoder\n')
    json.dump(p1, f, indent=4)
    f.write('\nHyperparameters for Classifier\n')  # Separator text between dictionaries
    json.dump(p2, f, indent=4)

columns = [
    # 'idx', 'learning_rate', 'hidden_size', 'dropout', 'num_head', 'num_transf', 'mlp_size', 'patch_size',
    # 'weight_decay', 'attention_selection', 'entangle', 'penny_or_kipu', 'RD', 'convolutional', 'paralel', 
    'q_layer', 'test_mse', 'val_mse', '#params1' , 'test_auc', 'test_acc', 'val_auc', 'val_acc', '#params2'
]

channels_last = False

df = pd.DataFrame(columns=columns)
df.to_csv('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv', mode='a', header=True, index=False)

# Grid search loop
for idx in range(50):
    print(f"\n\nPoint {idx}")

    for q_config in [True, False]:
        p1['quantum'] = q_config

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
            mlp_hidden_size=p1['mlp_size'], Attention_N = p1['Attention_N'], dropout=p1['dropout'], 
            RBF_similarity = p1['RBF_similarity'] ,channels_last=False
        )

        
        # Train
        test_mse, val_mse, params1 = qpctorch.training.train_and_evaluate(
            model1, train_dl, val_dl, test_dl, num_classes=7,
            learning_rate=p1['learning_rate'], num_epochs=N1, device=device, mapping=False,
            res_folder=str(save_path), hidden_size=p1['hidden_size'], dropout=p1['dropout'],
            num_heads=p1['num_head'], patch_size=p1['patch_size'], num_transf=p1['num_transf'],
            mlp=p1['mlp_size'], wd=p1['weight_decay'], patience= p1['patience'], scheduler_factor= p1['scheduler_factor'], autoencoder=True
        ) # type: ignore

        print(f"\nAutoencoder training completed succesfully.\nTest MSE (first step): {test_mse:.2f}")

        # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
        DataLoaders = [train_dl, val_dl, test_dl]
        LatentDatasetsTensors = []
        QuantumLayer = qpctorch.quantum.pennylane_backend.QuantumLayer(num_qubits = p1['mlp_size'], entangle = p1['entangle'], graph = p1['connectivity']) if p1['quantum'] else torch.nn.Identity() # type: ignore
        print(f"Current information about the Quantum Layer: {QuantumLayer}") # type: ignore                                                
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


            LatentDatasetsTensors.append( list(zip(all_latents,all_labels)) )

        latent_train_dl, latent_val_dl, latent_test_dl, shape2 = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                            tensors = LatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                            )

        # Create second model for the second step)

        model2 = qpctorch.quantum.vit.DeViT(num_classes=7, p = p2, shape = shape)

        print('\nTraining second model: classifier ViT on latent representations\n')

        # Train second model
        test_auc, test_acc, val_auc, val_acc, params2 = qpctorch.training.train_and_evaluate(
            model2, latent_train_dl, latent_val_dl, latent_test_dl, num_classes=7,
            learning_rate=p2['learning_rate'], num_epochs=N2, device=device, mapping=False,
            res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=p2['dropout'],
            num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
            mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor'], autoencoder=False
        ) # type: ignore

        
        # Save results
        row = {
            'idx': idx, 
                'q_layer' : p1['quantum'], 'test_mse': test_mse, 'val_mse': val_mse, '#params1': params1, 
                'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc, '#params2': params2,
                **p1, **p2
        }

        pd.DataFrame([row], columns=columns).to_csv(
            '../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv', mode='a', header=False, index=False
        )


        History_df = pd.read_csv('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv')

        plt.figure(figsize=(10, 6))
        plt.boxplot([History_df['val_auc'], History_df['test_auc']], labels=['Validation AUC', 'Test AUC']) # type: ignore
        plt.title('Validation and Test AUC Distribution')
        plt.ylabel('AUC')
        plt.grid(axis='y')
        plt.savefig('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/auc_boxplot.png')

