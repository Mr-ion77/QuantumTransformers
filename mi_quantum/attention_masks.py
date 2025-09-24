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
import numpy as np
import matplotlib.pyplot as plt


def save_attention(output,image,dir,patch_size=14):
    attentions = output.attentions[-1] # we are only interested in the attention maps of the last layer
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    threshold = 0.6
    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size #model.config.patch_size

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
    attentions = attentions.detach().numpy()

    # show attentions heatmaps
    image = np.transpose(image[0].cpu(), (1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min())
    
    for j in range(nh):
        # Crear una figura
        fig, ax = plt.subplots()

        # Mostrar la imagen de fondo
        ax.imshow(image, cmap='gray')

        # Superponer la imagen con transparencia alpha
        attentions_normalized = (attentions[j] - attentions[j].min())/ (attentions[j].max()-attentions[j].min())
        # Overlay the attention image with transparency
        im = ax.imshow(attentions_normalized, alpha=0.2, cmap='coolwarm')

        # Add a colorbar for the attention map using the imshow object
        cbar = plt.colorbar(im, ax=ax)  # Use the im object as the mappable
        cbar.set_label('Attention Values')  # Label for the colorbar

        # Hide the axes
        ax.axis('off')

        # Guardar la imagen resultante
        plt.savefig(dir+"_attn-head_"+str(j)+".jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return


def attention_masks(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, res_folder: str = "results_cc", patch_size: int=6, num_classes: int=7, train_att: bool = 'False', valid_att: bool = 'False', test_att: bool = 'True'):

    model = model.to(device)
    
    if train_att:
        try:
            os.makedirs(f'{res_folder}/train_masks')
            for i in range(num_classes):
                os.makedirs(f'{res_folder}/train_masks/class{i}')
            os.makedirs(f'{res_folder}/train_masks_misclassified')
            for i in range():
                os.makedirs(f'{res_folder}/train_masks_misclassified/train_masks/class{i}')        
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        train_dataloader_single = torch.utils.data.DataLoader(train_dataloader.dataset, batch_size=1)
        for images, labels, names in train_dataloader_single:
            images, labels = images.to(device), labels.to(device)
            dir = f'{res_folder}/train/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            dir_misclassified_train = f'{train_misclassified_dir}/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            with torch.no_grad():
                pred = model(images)  # Get both outputs and attentions
                if res_folder == "results_qc" or res_folder == "results_qq":
                    images = images.permute(0, 3, 1, 2)
                    images = images[:, 0:1, :, :]
                save_attention(pred, images, dir,patch_size)  # Pass attentions
                # Find misclassified samples
                pred2 = pred[:, 1]
                probs = torch.sigmoid(pred2) if num_classes == 2 else torch.softmax(pred2, dim=1)
                pred2 = [value >= 0.5 for value in probs] if num_classes == 2 else np.argmax(probs, axis=1)
                misclassified = pred2[0] != labels.squeeze().bool()
                if misclassified:
                    save_attention(pred, images, dir_misclassified_train,patch_size)  # Use attention for misclassified
        

    if valid_att:
        try:
            os.makedirs(f'{res_folder}/valid_masks')
            for i in range(num_classes):
                os.makedirs(f'{res_folder}/valid_masks/class{i}')
            os.makedirs(f'{res_folder}/valid_masks_misclassified')
            for i in range():
                os.makedirs(f'{res_folder}/valid_masks_misclassified/valid_masks/class{i}')        
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        valid_dataloader_single = torch.utils.data.DataLoader(valid_dataloader.dataset, batch_size=1)
        for images, labels, names in valid_dataloader_single:
            images, labels = images.to(device), labels.to(device)
            dir = f'{res_folder}/valid/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            dir_misclassified_valid = f'{valid_misclassified_dir}/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            with torch.no_grad():
                pred = model(images)  # Get both outputs and attentions
                if res_folder == "results_qc" or res_folder == "results_qq":
                    images = images.permute(0, 3, 1, 2)
                    images = images[:, 0:1, :, :]
                save_attention(pred, images, dir,patch_size)  # Pass attentions instead of pred
                # Find misclassified samples
                pred2 = pred[:, 1]
                probs = torch.sigmoid(pred2) if num_classes == 2 else torch.softmax(pred2, dim=1)
                pred2 = [value >= 0.5 for value in probs] if num_classes == 2 else np.argmax(probs, axis=1)
                misclassified = pred2[0] != labels.squeeze().bool()
                if misclassified:
                    save_attention(pred, images, dir_misclassified_test,patch_size)  # Use attention for misclassified


    if test_att:
        try:
            os.makedirs(f'{res_folder}/test_masks')
            for i in range(num_classes):
                os.makedirs(f'{res_folder}/test_masks/class{i}')
            os.makedirs(f'{res_folder}/test_masks_misclassified')
            for i in range():
                os.makedirs(f'{res_folder}/test_masks_misclassified/test_masks/class{i}')        
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        test_dataloader_single = torch.utils.data.DataLoader(test_dataloader.dataset, batch_size=1)
        for images, labels, names in valid_dataloader_single:
            images, labels = images.to(device), labels.to(device)
            dir = f'{res_folder}/test/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            dir_misclassified_test = f'{test_misclassified_dir}/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            with torch.no_grad():
                pred = model(images)  # Get both outputs and attentions
                if res_folder == "results_qc" or res_folder == "results_qq":
                    images = images.permute(0, 3, 1, 2)
                    images = images[:, 0:1, :, :]
                save_attention(pred, images, dir,patch_size)  # Pass attentions instead of pred
                # Find misclassified samples
                pred2 = pred[:, 1]
                probs = torch.sigmoid(pred2) if num_classes == 2 else torch.softmax(pred2, dim=1)
                pred2 = [value >= 0.5 for value in probs] if num_classes == 2 else np.argmax(probs, axis=1)
                misclassified = pred2[0] != labels.squeeze().bool()
                if misclassified:
                    save_attention(pred, images, dir_misclassified_test,patch_size)  # Use attention for misclassified
        
        
        