import os

import torch
import torch.utils.data
import torchvision
import numpy as np

import medmnist

class IndexedTensorDataset(torch.utils.data.TensorDataset):
    def __getitem__(self, idx):
        # Obtener los datos (imágenes y etiquetas)
        data = super().__getitem__(idx)
        # Devolver los datos junto con el índice real
        return (*data, idx)

def datasets_to_dataloaders(train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the given datasets"""
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dataloader_kwargs)
    return train_dataloader, valid_dataloader

def get_mnist_dataloaders(root: str = '~/data', **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the MNIST digits dataset (computer vision, 10-class classification)"""
    root = os.path.expanduser(root)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)

def get_medmnist_dataloaders(pixel: int = 28, data_flag: str = 'breastmnist', **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the MedMNIST dataset"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    info = medmnist.INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    class IndexedMedMNIST(DataClass):
        def __getitem__(self, idx):
            # Obtener la imagen y la etiqueta original
            img, label = super().__getitem__(idx)
            # Devolver la imagen, etiqueta y el índice real
            return img, label, idx
        
    train_dataset = IndexedMedMNIST(split='train', transform=transform, download=True, size=pixel)
    valid_dataset = IndexedMedMNIST(split='val', transform=transform, download=True, size=pixel)
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)

def get_quantum_medmnist_dataloaders(data_flag: str = 'breastmnist', imgs_dir='BREAST/kernel_2x2_4_circuits_all_qubits/', **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the quantum's embedding MedMNIST dataset"""
    info = medmnist.INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    original_train_dataset = DataClass(split='train', download=True)
    original_valid_dataset = DataClass(split='val', download=True)
    
    q_train_images = np.load(imgs_dir + "q_train_images.npy")
    q_valid_images = np.load(imgs_dir + "q_val_images.npy")

    train_dataset = IndexedTensorDataset(torch.tensor(q_train_images, dtype=torch.float32), 
                                         torch.tensor(original_train_dataset.labels, dtype=torch.float32))
    valid_dataset = IndexedTensorDataset(torch.tensor(q_valid_images, dtype=torch.float32), 
                                         torch.tensor(original_valid_dataset.labels, dtype=torch.float32))
            
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)