import time
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import shutil
import pandas as pd
from PIL import Image 
import matplotlib.pyplot as plt
import re

def evaluate_test(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, model_weights_path: str, num_classes: int, device: torch.device, res_folder: str='results'):

    param_names = ["learning_rate", "hidden_size", "dropout", "num_heads", "num_transf", "mlp", "patch_size", "wd"]
    # Expresión regular para capturar cada valor entre {}
    pattern = r"[-+]?\d*\.\d+|\d+"

    # Buscar todos los valores entre {}
    param_values = re.findall(pattern, model_weights_path)

    # Crear un diccionario para almacenar cada parámetro con su valor
    params = {name: float(value) if '.' in value else int(value) for name, value in zip(param_names, param_values)}

    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)

    # Evaluación en el conjunto de prueba
    model.eval()
    y_true, y_pred, images = [], [], []
    with torch.no_grad():
        for inputs, labels, ind in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float() if num_classes == 2 else labels.long()
            labels = labels.squeeze(1)
            
            outputs = model(inputs)
            if num_classes == 2 and outputs.shape[1] == 2:
                outputs = outputs[:, 1]
    
            probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)
    
            y_true.extend(labels.tolist())
            y_pred.extend(probabilities.tolist())
            images.extend(ind.tolist())
    
    # Cálculo de AUC y precisión
    test_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')
    y_pred_array = [value >= 0.5 for value in y_pred] if num_classes == 2 else np.argmax(y_pred, axis=1)
    test_acc = 100.0 * accuracy_score(y_true, y_pred_array)
    
    # Guardar resultados
    df = pd.DataFrame({
        'Images': images,
        'True Labels': y_true,
        'Predictions': y_pred_array,
        'Predicted Probabilities': y_pred
    })

    lr=params['learning_rate']
    hs=params['hidden_size']
    dr=params['dropout']
    nh=params['num_heads']
    nt=params['num_transf']
    mlp=params['mlp']
    ps=params['patch_size']
    wd=params['wd']
    
    df.to_csv(f'{res_folder}/Val_predictions_test_{lr}_{hs}_{dr}_{nh}_{nt}_{mlp}_{ps}_{wd}.csv', index=False)
    
    print(f"TEST AUC: {test_auc:.2f}%, TEST ACC: {test_acc:.2f}%")

    return test_auc, test_acc
