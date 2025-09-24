import torch
from torch import nn
from mi_quantum.quantum.vit import VisionTransformer
import numpy as np

# See:
# - https://nlp.seas.harvard.edu/annotated-transformer/
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py


def DbStpClssViT(vit1, threshold1, class1 ,vit2, num_classes): 
    
    """
    Create a Vision Transformer model for double step classification.

    """
    # Create a new model that combines the two Vision Transformers
    class DoubleStepClassificationViT(nn.Module):
        def __init__(self, model1, model2, threshold1, class1, num_classes):
            super().__init__()
            self.model1 = model1
            self.model2 = model2
            self.threshold1 = threshold1
            self.class1 = class1
            self.num_classes = num_classes
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)

            self.trainlosslist = []
            self.trauclist = []
            self.tracclist = []
            self.vallosslist = []
            self.auclist = []
            self.acclist = []

            if num_classes < 2:
                raise ValueError("num_classes must be greater than 2 for double step classification.")

        def forward(self, x):
            batch_size = x.size(0)
            device = x.device

            out1_raw = self.model1(x)[0]                     # (B, C)
            out1 = self.sigmoid(out1_raw).to(device)         # (B, C)
            out1_class1 = out1[:, 1]               # (B,)
            out1_raw1 = out1_raw[:, 1]             # (B,)

            output = torch.zeros((batch_size, self.num_classes), device=device)
            mask = out1_class1 > self.threshold1

            if mask.any():
                n = mask.sum().item()
                out1_raw1_selected = out1_raw1[mask]  # (n,)

                # Compute the fill values for each row
                fill_values = (1 - out1_raw1_selected) / (self.num_classes ** 2)  # (n,)

                # Create the special_out tensor with the computed values
                special_out = fill_values.unsqueeze(1).repeat(1, self.num_classes)  # (n, num_classes)

                # Replace the class1 column with the original raw output
                special_out[:, self.class1] = out1_raw1_selected

                # Assign to output
                output[mask] = special_out

            if (~mask).any():
                out2 = self.model2(x[~mask])[0]  # (n, C-1)
                n = out2.size(0)
                blend = torch.zeros((n, self.num_classes), device=device)
                class_indices = [i for i in range(self.num_classes) if i != self.class1]
                blend[:, class_indices] = out2
                blend[:, self.class1] = torch.zeros((n,), device=device)  # Class1 is zeroed out
                output[~mask] = blend

            return self.softmax(output)  # optional second softmax


    return DoubleStepClassificationViT(vit1, vit2, threshold1=0.6, class1=5, num_classes=7)