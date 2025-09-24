import torch
import torch.nn as nn
import mi_quantum as qpctorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N=150
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
columns = ["Testing params", "Learning rate", "Best AUC val", "Best epoch val", "Elapsed time ms"]
SAVE_PATH = 'tabla_pruebas.csv'
df = pd.read_csv(SAVE_PATH)

train_dataloader, valid_dataloader = qpctorch.data.get_medmnist_dataloaders(pixel=224, data_flag="dermamnist", batch_size=64, num_workers=4, pin_memory=True)

model = qpctorch.quantum.VisionTransformer(data_type="original", img_size=224, num_channels=3, num_classes=7, 
                                           patch_size=14, hidden_size=6, num_heads=2, num_transformer_blocks=4, 
                                           mlp_hidden_size=3)

plt.figure(figsize=(6*2,6))
plt.suptitle(model.prueba)
qpctorch.training.train_and_evaluate(model, train_dataloader, valid_dataloader, num_classes=7, 
                                     learning_rate=0.0003, num_epochs=N, mapping=True, device=device)

plt.subplot(1,2,1)                                  
plt.plot(np.arange(1,N+1), model.trainlosslist) 
plt.plot(np.arange(1,N+1), model.vallosslist) 
plt.xlabel('Epoch')                               
plt.legend(['Train','Validation'])                  
plt.title('Train vs Validation Loss over time')      

plt.subplot(1,2,2)                                 
plt.plot(np.arange(1,N+1), model.auclist)   
plt.plot(np.arange(1,N+1), model.acclist) 
plt.xlabel('Epoch')
plt.legend(['AUC-ROC','Accuracy'])                 
plt.title('Validation AUC-ROC and accuracy over time')   

plt.savefig(f"./derma_results/prueba{len(df)}.png")
plt.show()

lists = pd.DataFrame({'trainloss': model.trainlosslist, 
                      'valloss': model.vallosslist,
                      'valauc': model.auclist,
                      'valacc': model.acclist})
lists.index = np.arange(1, len(lists)+1)
lists.to_csv(f'./derma_results/prueba{len(df)}.csv', index=False)

new_row = pd.DataFrame([model.prueba], columns=columns)
df = pd.concat([df, new_row], ignore_index=True)

with open(SAVE_PATH, 'a') as f:
    new_row.to_csv(f, header=False, index=False)

#################
df = pd.read_csv(SAVE_PATH)

train_dataloader, valid_dataloader = qpctorch.data.get_medmnist_dataloaders(pixel=224, data_flag="dermamnist", batch_size=64, num_workers=4, pin_memory=True)
model = qpctorch.classical.VisionTransformer(data_type="original", img_size=224, num_channels=3, num_classes=7, 
                                             patch_size=14, hidden_size=6, num_heads=2, num_transformer_blocks=4, 
                                             mlp_hidden_size=3)

plt.figure(figsize=(6*2,6))
plt.suptitle(model.prueba)
qpctorch.training.train_and_evaluate(model, train_dataloader, valid_dataloader, num_classes=7, 
                                     learning_rate=0.0003, num_epochs=N, mapping=True, device=device)

plt.subplot(1,2,1)                                  
plt.plot(np.arange(1,N+1), model.trainlosslist) 
plt.plot(np.arange(1,N+1), model.vallosslist) 
plt.xlabel('Epoch')                               
plt.legend(['Train','Validation'])                  
plt.title('Train vs Validation Loss over time')      

plt.subplot(1,2,2)                                 
plt.plot(np.arange(1,N+1), model.auclist)   
plt.plot(np.arange(1,N+1), model.acclist) 
plt.xlabel('Epoch')
plt.legend(['AUC-ROC','Accuracy'])                 
plt.title('Validation AUC-ROC and accuracy over time')   

plt.savefig(f"./derma_results/prueba{len(df)}.png")
plt.show()

lists = pd.DataFrame({'trainloss': model.trainlosslist, 
                      'valloss': model.vallosslist,
                      'valauc': model.auclist,
                      'valacc': model.acclist})
lists.index = np.arange(1, len(lists)+1)
lists.to_csv(f'./derma_results/prueba{len(df)}.csv', index=False)

new_row = pd.DataFrame([model.prueba], columns=columns)
df = pd.concat([df, new_row], ignore_index=True)

with open(SAVE_PATH, 'a') as f:
    new_row.to_csv(f, header=False, index=False)