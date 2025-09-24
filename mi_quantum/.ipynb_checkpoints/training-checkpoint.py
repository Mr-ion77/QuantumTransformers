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

                        
def save_attention(output,image,dir,patch_size):
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
        im = ax.imshow(attentions_normalized, alpha=0.4, cmap='coolwarm')

        # Add a colorbar for the attention map using the imshow object
        cbar = plt.colorbar(im, ax=ax)  # Use the im object as the mappable
        cbar.set_label('Attention Values')  # Label for the colorbar

        # Hide the axes
        ax.axis('off')

        # Guardar la imagen resultante
        plt.savefig(dir+"_attn-head_"+str(j)+".jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return


def train_and_evaluate(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, num_classes: int, num_epochs: int, device: torch.device, mapping: bool = False, learning_rate: float = 1e-4, res_folder: str = "results_cc", hidden_size: int = 12, patch_size: int = 14, num_heads: int = 2, dropout: float = 0.2, num_transf: int = 1, mlp: int = 1, wd: float = 0.1, verbose: bool = False) -> None:
    """Trains the given model on the given dataloaders for the given parameters"""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    model = model.to(device)
    model.trainlosslist = []
    model.vallosslist = []
    model.auclist = []
    model.acclist = []
    best_y_true, best_y_pred, best_y_pred_prob = [], [], []

    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_val_auc, best_val_acc, best_epoch = 0.0, 0, 0
    
    start_time = time.time()
    for epoch in range(num_epochs):
        step = 0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}') as progress_bar:
            model.train()
            train_loss = 0.0
            y_trueTr, y_predTr, imagesTr = [], [], []
            for inputs, labels, ind in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)                
                if num_classes == 2:
                    labels = labels.float()
                else:
                    labels = labels.long()
                labels = labels.squeeze(1) 
                
                operation_start_time = time.time()

                optimizer.zero_grad()

                if verbose:
                    print(f" Zero grad ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()
                
                outputs = model(inputs)

                if num_classes == 2 and outputs.shape[1] == 2:
                    outputs = outputs[:, 1]

                if verbose:
                    print(f" Forward ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                loss = criterion(outputs, labels)

                if verbose:
                    print(f" Loss ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                loss.backward()

                if verbose:
                    print(f" Backward ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                optimizer.step()

                if verbose:
                    print(f" Optimizer step ({time.time()-operation_start_time:.2f}s)")

                step += 1
                progress_bar.update(1)
                #if verbose:
                    #print(f"Epoch {epoch+1}/{num_epochs} ({time.time()-start_time:.2f}s): Step {step}, Loss = {loss.item():.4f}")

                train_loss += loss.item()       ##
                probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)

                y_trueTr.extend(labels.tolist())
                y_predTr.extend(probabilities.tolist())
                imagesTr.extend(ind.tolist())
                
            
            train_loss /= len(train_dataloader) ##
            tr_auc = 100.0 * roc_auc_score(y_trueTr, y_predTr, multi_class='ovr')
            y_predTr_array = [value >= 0.5 for value in y_predTr] if num_classes == 2 else np.argmax(y_predTr, axis=1)

            tr_acc = 100.0 * accuracy_score(y_trueTr, y_predTr_array)

            ## Validation
            model.eval()
            y_true, y_pred, images = [], [], []
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels, ind in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if num_classes == 2:
                        labels = labels.float()
                    else:
                        labels = labels.long()
                    labels = labels.squeeze(1) 
                    
                    outputs = model(inputs)  
                    if num_classes == 2 and outputs.shape[1] == 2:
                        outputs = outputs[:, 1]

                    val_loss += criterion(outputs, labels).item()

                    probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)

                    y_true.extend(labels.tolist())
                    y_pred.extend(probabilities.tolist())
                    images.extend(ind.tolist())

            val_loss /= len(valid_dataloader)
            val_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')

            y_pred_array = [value >= 0.5 for value in y_pred] if num_classes == 2 else np.argmax(y_pred, axis=1)

                        
            val_acc = 100.0 * accuracy_score(y_true, y_pred_array)

            progress_bar.set_postfix_str(f"Loss={val_loss:.3f}, AUC={val_auc:.2f}%")

            model.trainlosslist.append(train_loss) 
            model.trauclist.append(tr_auc)
            model.tracclist.append(tr_acc)
            model.vallosslist.append(val_loss) 
            model.auclist.append(val_auc) 
            model.acclist.append(val_acc) 
            
            if val_auc > best_val_auc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_epoch = epoch + 1
                best_y_pred = y_pred_array
                best_y_pred_prob = y_pred
                best_y_true = y_true
                best_y_ind = images
                state_dict = model.state_dict()  # get the original state_dict
                torch.save(state_dict, f'{res_folder}/model_weights_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.pth')
    
    # Save predictions to CSV
    df = pd.DataFrame({
        'trainlosslist': model.trainlosslist,
        'trauclist': model.trauclist,
        'tracclist': model.tracclist,
        'vallosslist': model.vallosslist,
        'auclist': model.auclist,
        'acclist': model.acclist
    })
    df.to_csv(f'{res_folder}/Training_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.csv', index=False)

    print(f"TOTAL TIME = {time.time()-start_time:.2f}s")
    print(f"BEST AUC = {best_val_auc:.2f}% AT EPOCH {best_epoch}")
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    model.load_state_dict(torch.load(f'{res_folder}/model_weights_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.pth'))
    # Predictions in test
    model.eval()
    y_true, y_pred, images = [], [], []
    with torch.no_grad():
        for inputs, labels, ind in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if num_classes == 2:
                labels = labels.float()
            else:
                labels = labels.long()
            labels = labels.squeeze(1) 
               
            outputs = model(inputs)  
            if num_classes == 2 and outputs.shape[1] == 2:
                outputs = outputs[:, 1]

            probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(probabilities.tolist())
            images.extend(ind.tolist())

    test_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')
    y_pred_array = [value >= 0.5 for value in y_pred] if num_classes == 2 else np.argmax(y_pred, axis=1)
    test_acc = 100.0 * accuracy_score(y_true, y_pred_array)

    df = pd.DataFrame({
        'Images': images,
        'True Labels': y_true,
        'Predictions': y_pred_array,
        'Predicted Probabilities': y_pred
    })
    df.to_csv(f'{res_folder}/predictions_test_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.csv', index=False)

    print(f"TEST AUC: {test_auc:.2f}%, TEST ACC: {test_acc:.2f}%")

    ## Saving attention maps   
    if mapping:
        # Create directories for misclassified images
        train_misclassified_dir = f'{res_folder}/train_misclassified'
        test_misclassified_dir = f'{res_folder}/test_misclassified'
        
        try:
            os.makedirs(f'{res_folder}/train')
            os.makedirs(f'{res_folder}/test')
            for i in range(num_classes):
                os.makedirs(f'{res_folder}/train/class{i}')
                os.makedirs(f'{res_folder}/test/class{i}')

            os.makedirs(train_misclassified_dir)
            os.makedirs(test_misclassified_dir)
            for i in range(num_classes):
                os.makedirs(f'{train_misclassified_dir}/class{i}')
                os.makedirs(f'{test_misclassified_dir}/class{i}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        valid_dataloader_single = torch.utils.data.DataLoader(valid_dataloader.dataset, batch_size=1)
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
                if (num_classes == 2 and pred.shape[1]==2):
                    pred2 = pred[:,1]

                else:
                    pred2 = pred
                probs = torch.sigmoid(pred2) if num_classes == 2 else torch.softmax(pred2, dim=1)
                pred2 = [value >= 0.5 for value in probs] if num_classes == 2 else torch.argmax(probs, dim=1)
                misclassified = pred2[0] != labels.squeeze().bool()
                if misclassified:
                    save_attention(pred, images, dir_misclassified_test,patch_size)  # Use attention for misclassified

        #train_dataloader_single = torch.utils.data.DataLoader(train_dataloader.dataset, batch_size=1)
        #for images, labels, names in train_dataloader_single:
        #    images, labels = images.to(device), labels.to(device)
        #    dir = f'{res_folder}/train/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
        #    dir_misclassified_train = f'{train_misclassified_dir}/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
        #    with torch.no_grad():
        #        pred = model(images)  # Get both outputs and attentions
        #        if res_folder == "results_qc" or res_folder == "results_qq":
        #            images = images.permute(0, 3, 1, 2)
        #            images = images[:, 0:1, :, :]
        #        save_attention(pred, images, dir,patch_size)  # Pass attentions
        #        # Find misclassified samples
        #        if (num_classes == 2 and pred.shape[1]==2):
        #            pred2 = pred[:, 1]
        #        else:
        #            pred2 = pred
        #        probs = torch.sigmoid(pred2) if num_classes == 2 else torch.softmax(pred2, dim=1)
        #        pred2 = [value >= 0.5 for value in probs] if num_classes == 2 else torch.argmax(probs, dim=1)
        #        misclassified = pred2[0] != labels.squeeze().bool()
        #        if misclassified:
        #            save_attention(pred, images, dir_misclassified_train,patch_size)  # Use attention for misclassified
    return test_auc, test_acc, best_val_auc, best_val_acc