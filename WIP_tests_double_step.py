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
N = 100  # Number of epochs

# Hyperparams
p1 = {
    'learning_rate': 0.01, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.3, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' = True, 'num_head': 4, 'num_transf': 2, 'mlp_size': 3, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter', 'entangle': True,
    'penny_or_kipu': 'kipu', 'RD': 1, 'convolutional': True, 'paralel': 2, 'patience': -1, 'scheduler_factor': 0.999, 'q_stride': 8  # No early stopping
}

p2 = {
    'learning_rate': 0.01, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.3, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' = True, 'num_head': 4, 'num_transf': 3, 'mlp_size': 3, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter', 'entangle': True,
    'penny_or_kipu': 'kipu', 'RD': 1, 'convolutional': True, 'paralel': 2, 'patience': -1, 'scheduler_factor': 1.0, 'q_stride': 8  # No early stopping
}

columns = [
    # 'idx', 'learning_rate', 'hidden_size', 'dropout', 'num_head', 'num_transf', 'mlp_size', 'patch_size',
    # 'weight_decay', 'attention_selection', 'entangle', 'penny_or_kipu', 'RD', 'convolutional', 'paralel', 
    'test_auc1', 'test_acc1',  'space', 'test_auc2', 'test_acc2',  'spacE', # 'val_auc1', 'val_acc1', 'val_auc2', 'val_acc2',
    'test_auc_last', 'test_acc_last', 'thres'
]

df = pd.DataFrame(columns=columns)
df.to_csv('derma_results/current_results/results_grid_search.csv', mode='a', header=True, index=False)

# Grid search loop
for idx in range(50):
    print(f"\n\nPoint {idx}")

    save_path = Path(f"derma_results/current_results/grid_search{idx}")
    save_path.mkdir(parents=True, exist_ok=True)

    print('\n')

    # Load data
    train_dl, val_dl, test_dl = qpctorch.data.get_medmnist_dataloaders(
        pixel=28, data_flag='dermamnist', batch_size=B, num_workers=4, pin_memory=True
    )

    # Split datasets by label
    
    train_majority_ds, train_others_ds = split_dataset_by_label(train_dl.dataset.clone(), majority_label=5)
    val_majority_ds, val_others_ds = split_dataset_by_label(val_dl.dataset.clone(), majority_label=5)
    test_majority_ds, test_others_ds = split_dataset_by_label(test_dl.dataset.clone(), majority_label=5)

    # Relabel datasets

    # Create binary datasets for majority class
    # Relabel the majority class to 1 and others to 0
    train_binary_for_majority_ds = relabel_dataset(train_dl.dataset.clone(), majority_label=5, majority_or_others='majority')
    val_binary_for_majority_ds = relabel_dataset(val_dl.dataset.clone(), majority_label=5, majority_or_others='majority')
    test_binary_for_majority_ds = relabel_dataset(test_dl.dataset.clone(), majority_label=5, majority_or_others='majority')

    count_train = 0
    for image, label, ind in train_binary_for_majority_ds:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == 1:
            count_train += 1

    count_val = 0
    for image, label, ind in val_binary_for_majority_ds:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == 1:
            count_val += 1

    count_test = 0
    for image, label, ind in test_binary_for_majority_ds:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == 1:
            count_test += 1

    print(f"\nTRAIN SPLIT: Number of samples in majority class (label 1): {count_train}. Number of samples in others class (label 0): {len(train_binary_for_majority_ds) - count_train}")
    print(f"VAL SPLIT: Number of samples in majority class (label 1): {count_val}. Number of samples in others class (label 0): {len(val_binary_for_majority_ds) - count_val}")
    print(f"TEST SPLIT: Number of samples in majority class (label 1): {count_test}. Number of samples in others class (label 0): {len(test_binary_for_majority_ds) - count_test}\n")
    
    # Create binary datasets for others class
    # Relabel classes with label > majority_label to label-1 for consistency
    train_others_ds = relabel_dataset(train_others_ds, majority_label=5, majority_or_others='others')
    val_others_ds = relabel_dataset(val_others_ds, majority_label=5, majority_or_others='others')
    test_others_ds = relabel_dataset(test_others_ds, majority_label=5, majority_or_others='others')

    # Create DataLoaders
    train_majority_dl, val_majority_dl, test_majority_dl = qpctorch.data.datasets_to_dataloaders(
        train_binary_for_majority_ds, val_binary_for_majority_ds, test_binary_for_majority_ds,
        batch_size=B, num_workers=4, pin_memory=True)

    train_others_dl, val_others_dl, test_others_dl = qpctorch.data.datasets_to_dataloaders(
        train_others_ds, val_others_ds, test_others_ds,
        batch_size=B, num_workers=4, pin_memory=True)

    # from collections import Counter
    # labels_train = [label.item() if isinstance(label, torch.Tensor) else label for _, label, _ in train_majority_ds]
    # print("Train labels distribution:", Counter(labels_train))


    # Model
    model1 = qpctorch.quantum.VisionTransformer(
        data_type="original", img_size=28, num_channels=3, num_classes=2,
        patch_size= p1['patch_size'], hidden_size=p1['hidden_size'], num_heads=p1['num_head'],
        num_transformer_blocks=p1['num_transf'], attention_selection=p1['attention_selection'],
        mlp_hidden_size=p1['mlp_size'], dropout=p1['dropout'], channels_last=False,
        entangle=p1['entangle'], penny_or_kipu=p1['penny_or_kipu'], RD=p1['RD'], 
        convolutional = p1['convolutional'], paralel=p1['paralel']
    )
    print('\nTraining first model for majority class...')
    # Train
    test_auc1, test_acc1, val_auc1, val_acc1, params1 = qpctorch.training.train_and_evaluate(
        model1, train_majority_dl, val_majority_dl, test_majority_dl, num_classes=2,
        learning_rate=p1['learning_rate'], num_epochs=N, device=device, mapping=False,
        res_folder=str(save_path), hidden_size=p1['hidden_size'], dropout=p1['dropout'],
        num_heads=p1['num_head'], patch_size=p1['patch_size'], num_transf=p1['num_transf'],
        mlp=p1['mlp_size'], wd=p1['weight_decay'], patience= p1['patience'], scheduler_factor= p1['scheduler_factor']
    )

    print(f"\nMajority class training completed succesfully.\nTest AUC (first step): {test_auc1:.2f}, Test Accuracy (first step): {test_acc1:.2f}")

    # Create second model for the second step
    model2 = qpctorch.quantum.VisionTransformer(
        data_type="original", img_size=28, num_channels=3, num_classes=6,
        patch_size= p2['patch_size'], hidden_size=p2['hidden_size'], num_heads=p2['num_head'],
        num_transformer_blocks=p2['num_transf'], attention_selection=p2['attention_selection'],
        mlp_hidden_size=p2['mlp_size'], dropout=p2['dropout'],
        channels_last=False, entangle=p2['entangle'], penny_or_kipu=p2['penny_or_kipu'], 
        RD=p2['RD'], convolutional = p2['convolutional'], paralel=p2['paralel']
    )
    print('\nTraining second model for others class...')
    # Train second model
    test_auc2, test_acc2, val_auc2, val_acc2, params2 = qpctorch.training.train_and_evaluate(
        model2, train_others_dl, val_others_dl, test_others_dl, num_classes=6,
        learning_rate=p2['learning_rate'], num_epochs=N, device=device, mapping=False,
        res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=p2['dropout'],
        num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
        mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor']
    )

    print(f"\nOthers class training completed succesfully.\nTest AUC (second step): {test_auc2:.2f}, Test Accuracy (second step): {test_acc2:.2f}")

    num_classes = 7
    # Create double step classification model
    best_auc, best_acc, best_thres = 0.0, 0.0, 0.5

    print('\nInference on test set for double step classification...')

    train_double_step_model = True
    if train_double_step_model:
        double_step_model = DbStpClssViT(
            vit1=model1, threshold1=0.6, class1=5, vit2=model2, num_classes=num_classes
        )


        best_auc, best_acc, best_val_auc, best_val_acc, n_params = qpctorch.training.train_and_evaluate(
            double_step_model, train_dl, val_dl, test_dl, num_classes=num_classes,
            learning_rate=0.01, num_epochs=N, device=device, mapping=False,
            res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=p2['dropout'],
            num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
            mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor= 0.9875
        )

    else:

        for thres in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: # Threshold for the first step classification
    
            double_step_model = DbStpClssViT(
                vit1=model1, threshold1=thres, class1=5, vit2=model2, num_classes=num_classes
            )

            double_step_model.to(device)

            # Evaluate the double step model
            
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            double_step_model.eval()
            y_true, y_pred, images = [], [], []
            with torch.no_grad():
                for inputs, labels, ind in test_dl:

                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.float() if num_classes == 2 else labels.long()
                    labels = labels.squeeze(1) if labels.dim() > 1 else labels


                    outputs = double_step_model(inputs)
                    if num_classes == 2 and outputs.shape[1] == 2:
                        outputs = outputs[:, 1]
            
                    probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)
            
                    y_true.extend(labels.tolist())
                    y_pred.extend(probabilities.tolist())
                    images.extend(ind.tolist())
            
            # Cálculo de AUC y precisión
            test_auc_last = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')
            y_pred_array_last = [value >= thres for value in y_pred] if num_classes == 2 else np.argmax(y_pred, axis=1)
            test_acc_last = 100.0 * accuracy_score(y_true, y_pred_array_last)

            

            if test_auc_last > best_auc:
                best_auc = test_auc_last
                best_acc = test_acc_last
                best_thres = thres

        print(f"\nBest AUC: {best_auc:.2f}, Best Accuracy: {best_acc:.2f}, Best Threshold: {best_thres:.2f}")
        

    # Save results
    row = {
        'idx': idx,
            'test_auc1': test_auc1, 'test_acc1': test_acc1, 'space': '           ', #'val_auc1': val_auc1, 'val_acc1': val_acc1,
            'test_auc2': test_auc2, 'test_acc2': test_acc2, 'spacE': '           ', # 'val_auc2': val_auc2, 'val_acc2': val_acc1,
            'test_auc_last': best_auc, 'test_acc_last': best_acc, 'thres': best_thres,

    }

    pd.DataFrame([row], columns=columns).to_csv(
        'derma_results/current_results/results_grid_search.csv', mode='a', header=False, index=False
    )
