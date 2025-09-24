import os
import torch
import numpy as np
import torchvision.transforms as transforms
import medmnist
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset




num_channels = 3

class MinMaxScalePerChannel:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        # Assume tensor shape: (C, H, W)
        c, h, w = tensor.shape
        scaled = torch.empty_like(tensor)
        
        for ch in range(c):
            ch_min = tensor[ch].min()
            ch_max = tensor[ch].max()

            if ch_max == ch_min:
                # Avoid division by zero; fill with min_val
                scaled[ch] = torch.full((h, w), self.min_val, dtype=tensor.dtype, device=tensor.device)
            else:
                ch_scaled = (tensor[ch] - ch_min) / (ch_max - ch_min)
                scaled[ch] = ch_scaled * (self.max_val - self.min_val) + self.min_val

        return scaled

train_transforms = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

valid_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

class SubsetAsDataset(Dataset):
    def __init__(self, subset: Subset):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def split_dataset_by_label(dataset, majority_label=5):
    majority_indices = []
    others_indices = []

    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == majority_label:
            majority_indices.append(i)
        else:
            others_indices.append(i)

    majority_subset = SubsetAsDataset( Subset(dataset, majority_indices) )
    others_subset = SubsetAsDataset( Subset(dataset, others_indices) )

    return majority_subset, others_subset

def relabel_dataset(dataset, majority_label, majority_or_others='majority'):
    """Relabels the dataset and returns a new IndexedTensorDataset."""
    new_images = []
    new_labels = []
    new_indices = []

    for i in range(len(dataset)):
        image, old_label, index = dataset[i]
        if isinstance(old_label, torch.Tensor):
            old_label = old_label.item()
        if majority_or_others == 'majority':
            new_label = 1 if old_label == majority_label else 0
        elif majority_or_others == 'others':
            new_label = old_label if old_label < majority_label else old_label - 1
            new_label = new_label[0]
        else:
            raise ValueError("majority_or_others must be 'majority' or 'others'")
        
        new_images.append(image)
        new_labels.append(new_label)
        new_indices.append(index)

    # Convertir a tensores
    image_tensor = torch.stack(new_images)
    label_tensor = torch.tensor(new_labels)
    index_tensor = torch.tensor(new_indices)

    print(f"Relabeled dataset: {len(new_images)} images, {len(new_labels)} labels, {len(new_indices)} indices. Also, {set(new_labels)} unique labels.")

    return TensorDataset(image_tensor, label_tensor, index_tensor)


def datasets_to_dataloaders(train_dataset, valid_dataset, test_dataset, **dataloader_kwargs):
    """Returns dataloaders for the given datasets"""
    shape = train_dataset[0][0].shape  # Assuming all datasets have the same shape
    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)        # shuffle=True en train
    valid_dataloader = DataLoader(valid_dataset, **dataloader_kwargs)      #shuffle=False en los otros
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    return train_dataloader, valid_dataloader, test_dataloader, shape


def get_medmnist_dataloaders(pixel: int = 28, data_flag: str = 'breastmnist', **dataloader_kwargs) -> tuple:
    """Returns dataloaders for the MedMNIST dataset"""
    # Transformaciones
    n_channels = 3

    valid_transform = valid_transforms
    
    train_transform = train_transforms
    
    info = medmnist.INFO[data_flag]                            # Estas dos líneas permiten reutilizar el código más fácilmente
    DataClass = getattr(medmnist, info['python_class'])        # para distintos datasets

    class IndexedMedMNIST(DataClass):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            return img, label, idx

        def clone(self):
            # Reconstruct the dataset with the same parameters
            new_instance = IndexedMedMNIST(
                split=self.split,
                transform=self.transform,
                target_transform=self.target_transform,
                download=False  # prevent re-downloading
            )

            # Manually copy relevant attributes
            if hasattr(self, 'data'):
                try:
                    new_instance.data = self.data.clone()
                except AttributeError:
                    new_instance.data = self.data.copy()  # For NumPy arrays

            if hasattr(self, 'labels'):
                try:
                    new_instance.labels = self.labels.clone()
                except AttributeError:
                    new_instance.labels = self.labels.copy()  # For NumPy arrays

            return new_instance


    train_dataset = IndexedMedMNIST(split='train', transform=train_transform, download=True)    # Construye el dataset a partir de la clase obtenida
    valid_dataset = IndexedMedMNIST(split='val', transform=valid_transform, download=True)
    test_dataset = IndexedMedMNIST(split='test', transform=valid_transform, download=True)


    return datasets_to_dataloaders(train_dataset, valid_dataset, test_dataset, **dataloader_kwargs)

# Function for: Quantum preprocessed datasets

q_train_transforms = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #MinMaxScalePerChannel(0, np.pi)
    ])

q_valid_transforms = transforms.Compose([
        #MinMaxScalePerChannel(0, np.pi)
    ])

class TransformedTensorDataset(torch.utils.data.Dataset):
    """Custom dataset that applies a transform to each item."""
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        image, label =  self.tensors[index] 
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        return len(self.tensors)

def load_numpy_data(data_dir, channels_last = True):
    # Load the .npy files
    train_images = torch.from_numpy( np.load(os.path.join(data_dir, 'q_train_images.npy')) ).float()
    val_images = torch.from_numpy( np.load(os.path.join(data_dir, 'q_val_images.npy')) ).float()
    test_images = torch.from_numpy( np.load(os.path.join(data_dir, 'q_test_images.npy')) ).float()

    if channels_last: # Change shape to [N, C, H, W]
        train_images = train_images.permute(0, 3, 1, 2)
        val_images = val_images.permute(0, 3, 1, 2)
        test_images = test_images.permute(0, 3, 1, 2)

    train_labels = np.load(os.path.join(data_dir, 'q_train_labels.npy'))
    val_labels = np.load(os.path.join(data_dir, 'q_val_labels.npy'))
    test_labels = np.load(os.path.join(data_dir, 'q_test_labels.npy'))

    # Create list of (image, label) pairs
    train_tensor = list(zip(train_images, torch.from_numpy(train_labels).long()))
    val_tensor = list(zip(val_images, torch.from_numpy(val_labels).long()))
    test_tensor = list(zip(test_images, torch.from_numpy(test_labels).long()))

    return train_tensor, val_tensor, test_tensor


def create_dataloaders(data_dir, batch_size=32, channels_last = True, shuffle=True, tensors = None, transforms={'train': q_train_transforms, 'val': q_valid_transforms, 'test': q_valid_transforms}):
    if tensors is not None:
        train_tensor, val_tensor, test_tensor = tensors
    else:
        train_tensor, val_tensor, test_tensor = load_numpy_data(data_dir, channels_last= channels_last)

    # Wrap with transformed datasets
    train_dataset = TransformedTensorDataset(train_tensor, transform=transforms['train'])
    val_dataset = TransformedTensorDataset(val_tensor, transform=transforms['val'])
    test_dataset = TransformedTensorDataset(test_tensor, transform=transforms['test'])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_tensor[0][0].shape  # Return the shape of the images






