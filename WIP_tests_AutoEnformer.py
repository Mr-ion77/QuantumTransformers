import torch
import torch.nn as nn
import mi_quantum as qpctorch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import itertools
from mi_quantum.data import split_dataset_by_label, relabel_dataset
from mi_quantum.quantum.quanvolution import QuantumConv2D
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

B = 256
N1 = 150  # Number of epochs
N2 = 150  # Number of epochs for the second step

# Hyperparams
p1 = {
    'learning_rate': 5e-3, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'num_head': 4, 'Attention_N' : 2, 'num_transf': 1, 'mlp_size': 9, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'none', 'entangle': True,
    'paralel' : 2 ,'connectivity': 'king', 'RD': 1, 'patience': -1, 'scheduler_factor': 0.999, 'q_stride': 1 , 'RBF_similarity': 'none'  # No early stopping
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.1, 'after_attn': 0.05, 'feedforward': 0.05, 'embedding_pos': 0.05},
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 9, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'RD': 1, 'special_cls' : False, 'paralel': 2, 'patience': -1, 'scheduler_factor': 0.9995, 'q_stride': 1, 'RBF_similarity': 'none'  # No early stopping
}

# Save dictionary with all the hyperparameters and results in a json file

os.makedirs('../QTransformer_Results_and_Datasets/autoenformer_results/current_results', exist_ok = True)

with open('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/hyperparameters.json', 'w') as f:
    f.write('\nHyperparameters for Autoencoder\n')
    json.dump(p1, f, indent=4)
    f.write('\nHyperparameters for Classifier\n')  # Separator text between dictionaries
    json.dump(p2, f, indent=4)

columns = [
    # 'idx', 'learning_rate', 'hidden_size', 'dropout', 'num_head', 'num_transf', 'mlp_size', 'patch_size',
    # 'weight_decay', 'attention_selection', 'entangle', 'penny_or_kipu', 'RD', 'convolutional', 'paralel', 
    'idx', 'lr', 'q_config', 'test_mse', 'val_mse', '#params1' , 'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc',  '#params2'
]

channels_last = False           # Set to True if last dimension of datasets tensors match channels dimension
RepeatAutoencoder = False       # Set to True if you want to train the autoencoder each time for more variability. For a better performance and faster training set to False.

df = pd.DataFrame(columns=columns)
df.to_csv('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv', mode='a', header=True, index=False)

q_config = {'none', 'patchwise', 'quanvolution'}

# Grid search loop
for idx in range(20):
    for lr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
        print(f"\n\nPoint {idx}")
        save_path = Path(f"../QTransformer_Results_and_Datasets/autoenformer_results/current_results/grid_search{idx}")
        save_path.mkdir(parents=True, exist_ok=True)
        os.makedirs(save_path / 'autoencoder', exist_ok=True)

        # Determine if quantum processing is enabled based on q_config
        # q_config can be a string or a list/tuple; handle both
        NoneBool, PatchBool, QuanBool = 'none' in q_config, 'patchwise' in q_config, 'quanvolution' in q_config
        print(f"Current quantum configuration:\nNormal latent representations: {NoneBool}\nPatchwise Quantum latent representations: {PatchBool}\nQuanvolution latent representations: {QuanBool}")

        if (idx == 0) and (lr == 1e-5)or RepeatAutoencoder:
            print(f'\nTraining first model: Autoencoder\nOptiosn: Autoencoder with Quantum Layers: {list(q_config)} and Learning Rate: {p1["learning_rate"]}\n')

            # Load data
            train_dl, val_dl, test_dl, shape = qpctorch.data.get_medmnist_dataloaders(
                pixel=28, data_flag='dermamnist', batch_size=B, num_workers=4, pin_memory=True
            )

            # Obtain general settings regarding dataset shape
            num_channels = shape[-1] if channels_last else shape[0]
            img_size = shape[1]  # Assuming square images

            # Model
            model1 = qpctorch.quantum.vit.AutoEnformer(
                img_size=img_size, num_channels=num_channels,   # set num_classes as needed
                patch_size=p1['patch_size'], hidden_size=p1['hidden_size'], num_heads=p1['num_head'],
                num_transformer_blocks=p1['num_transf'], attention_selection=p1['attention_selection'],
                mlp_hidden_size=p1['mlp_size'], Attention_N = p1['Attention_N'], dropout=p1['dropout'], 
                paralel = p1['paralel'], RBF_similarity = p1['RBF_similarity'] ,channels_last=False
            )

            # Train
            test_mse, val_mse, params1 = qpctorch.training.train_and_evaluate(
                model1, train_dl, val_dl, test_dl, num_classes=7,
                learning_rate=p1['learning_rate'], num_epochs=N1, device=device, mapping=False,
                res_folder=str(save_path) + '/autoencoder', hidden_size=p1['hidden_size'], dropout=p1['dropout'],
                num_heads=p1['num_head'], patch_size=p1['patch_size'], num_transf=p1['num_transf'],
                mlp=p1['mlp_size'], wd=p1['weight_decay'], patience= p1['patience'], scheduler_factor= p1['scheduler_factor'], autoencoder=True,
                save_reconstructed_images = True if idx == 0 else False
            ) # type: ignore

            print(f"\nAutoencoder training completed succesfully.\nTest MSE (first step): {test_mse:.5f}")

            # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
            DataLoaders = [train_dl, val_dl, test_dl]
            if NoneBool:
                NorLatentDatasetsTensors = []
            if PatchBool:
                QuLatentDatasetsTensors = []
                QuantumLayer = qpctorch.quantum.pennylane_backend.QuantumLayer(num_qubits = p1['mlp_size'], entangle = p1['entangle'], graph = p1['connectivity'])
            if QuanBool:
                MoLatentDatasetsTensors = []
                Quanvolution = QuantumConv2D(patch_size=3, stride=1, padding=1, channels_out = [4], ancilla = 0, graph= 'king')

            print(f'Quantum configuration is {q_config} ')
            
        for dl in tqdm(DataLoaders, desc="Processing DataLoaders"):
            if NoneBool:
                all_latents_normal = []
            if PatchBool:
                all_latents_quantum = []
            if QuanBool:
                all_latents_quanv = []
            
            all_labels = []
            all_indices = []
            for images, labels, indices in dl:
                images = images.to(device)
                with torch.no_grad():

                    if NoneBool:
                        normal_outs = []
                    if PatchBool:
                        quantum_outs = []
                    if QuanBool:
                        quanv_outs = []                   
                    
                    latent_aux = model1.get_latent_representation(images)
                    for i in range(p1['paralel']):
                            
                        if PatchBool:
                            quantum_outs.append( QuantumLayer( latent_aux[i] ) )
                        if QuanBool:
                            quanv_outs.append( Quanvolution( latent_aux[i].unsqueeze(dim = 1) ) )

                all_labels.extend( labels )
                if NoneBool:
                    all_latents_normal.extend( latent_aux.view( (latent_aux.shape[1],latent_aux.shape[2], -1) ) )
                if PatchBool:
                    quantum_representations = torch.cat(quantum_outs, dim = -1).cpu()        
                    all_latents_quantum.extend( quantum_representations )
                if QuanvOut:
                    quanv_representations = torch.cat(quanv_outs, dim = -1).cpu()
                    all_latents_quanv.extend( quanv_representations )

            all_labels = torch.tensor(all_labels)
            if NoneBool:
                all_latents_normal = torch.stack(all_latents_normal)
                NorLatentDatasetsTensors.append( list(zip(all_latents_normal ,all_labels))  )
            if PatchBool:
                all_latents_quantum = torch.stack(all_latents_quantum)
                QuLatentDatasetsTensors.append ( list(zip(all_latents_quantum, all_labels)) )
            if QuanvBool:
                all_latents_quanv = torch.stack(all_latents_quanv)
                MoLatentDatasetsTensors.append ( list(zip(all_latents_quanv, all_labels)) )

                
        if NoneBool:
            NorLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                        tensors = NorLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                        )
        if PatchBool:
            PatchLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                    tensors = QuLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                    )
        if QuanvBool:
            QuanvLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                    tensors = MoLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                    )

        Latents = {k: v for k, v in zip(['none', 'patchwise', 'quanvolution'], [NorLatents, PatchLatents, QuanvLatents]) if k in q_config}

        if idx == 0:
            save_path_latent = Path(f"../QTransformer_Results_and_Datasets/autoenformer_results/current_results/latent_datasets")
            save_path_latent.mkdir(parents=True, exist_ok=True)
            if PatchBool:
                torch.save(QuLatentDatasetsTensors[0], save_path_latent / 'qlatent_train_dataset.pt')
                torch.save(QuLatentDatasetsTensors[1], save_path_latent / 'qlatent_val_dataset.pt')
                torch.save(QuLatentDatasetsTensors[2], save_path_latent / 'qlatent_test_dataset.pt')
            if NoneBool:
                torch.save(NorLatentDatasetsTensors[0], save_path_latent / 'nlatent_train_dataset.pt')
                torch.save(NorLatentDatasetsTensors[1], save_path_latent / 'nlatent_val_dataset.pt')
                torch.save(NorLatentDatasetsTensors[2], save_path_latent / 'nlatent_test_dataset.pt')
            if QuanvBool:
                torch.save(MoLatentDatasetsTensors[0], save_path_latent / 'mlatent_train_dataset.pt')
                torch.save(MoLatentDatasetsTensors[1], save_path_latent / 'mlatent_val_dataset.pt')
                torch.save(MoLatentDatasetsTensors[2], save_path_latent / 'mlatent_test_dataset.pt')
            

        # Create second model for the second step)

        for config in list(q_config):
            config_dataset = Latents[config]
            shape2 = config_dataset[-1]
            p2['learning_rate'] = lr
            model2 = qpctorch.quantum.vit.DeViT(num_classes=7, p = p2, shape = shape, dim_latent = shape2[-1]) # The shape needed is that of the original images, in this case [3, 28, 28]

            print('\nTraining second model: classifier ViT on latent representations\n')
            print(f'QUANTUM SETTING IS: {config} and current lr is: {p2["learning_rate"]}')
            # Train second model
            test_auc, test_acc, val_auc, val_acc, train_auc, params2 = qpctorch.training.train_and_evaluate(
                model2, config_dataset[0], config_dataset[1], config_dataset[2], num_classes=7,
                learning_rate=p2['learning_rate'], num_epochs=N2, device=device, mapping=False,
                res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=p2['dropout'],
                num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
                mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor'], autoencoder=False
            ) # type: ignore

            
            # Save results
            row = {
                'idx': idx, 
                    'q_config' : config, 'test_mse': test_mse, 'val_mse': val_mse, '#params1': params1, 
                    'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc, 'train_auc': train_auc,'#params2': params2,
                    **p1, **p2
            }

            pd.DataFrame([row], columns=columns).to_csv(
                '../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv', mode='a', header=False, index=False
            )

            History_df = pd.read_csv('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/results_grid_search.csv')

            plt.figure(figsize=(10, 6))
            plt.boxplot([History_df['val_auc'], History_df['test_auc']], tick_labels=['Validation AUC', 'Test AUC']) # type: ignore
            plt.title('Validation and Test AUC Distribution')
            plt.ylabel('AUC')
            plt.grid(axis='y')
            plt.savefig('../QTransformer_Results_and_Datasets/autoenformer_results/current_results/auc_boxplot.png')

