import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_quantum.quantum.pennylane_backend import QuantumLayer


class QuantumKernel(nn.Module):
    def __init__(self, circuit, channels_out = [-1], ancilla = 1):
        super().__init__()
        self.circuit = circuit
        self.channels_out = channels_out if ancilla == 0 else list(range(-1, -ancilla - 1, -1))
        self.ancilla = ancilla        

    def forward(self,x):
        assert x.shape[-1] + self.ancilla == self.circuit.num_qubits, f"patch_size**2 should match number of qubits in the kernel"
        
        if self.ancilla > 0:
            ancilla_tensor = torch.zeros(*x.shape[:-1], self.ancilla, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ancilla_tensor], dim=-1)


        circuit_out = self.circuit(x)

        return circuit_out[...,  self.channels_out]
    

class QuantumConv2D(nn.Module):
    def __init__(self, patch_size=3, stride=1, padding=0, channels_out = [4], channels_last = False, graph= 'chain', ancilla = 1, trainBool = False):
        super().__init__()

        if ancilla and channels_out != [-1]:
            print(f'Please be ware that when ancilla is set to True channels_out must be [-1], but got {channels_out}. Automatically setting channels_out to [-1]')

        self.channels_out = channels_out if not ancilla else [-1]
        self.kernel = QuantumKernel(
            circuit = QuantumLayer(num_qubits = patch_size**2 + ancilla, entangle = True, graph = graph, trainBool = False),
            channels_out = channels_out, ancilla = ancilla
        )
        self.trainBool = trainBool
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.ancilla = ancilla
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.stride, padding=self.padding) # Unfold the input to get sliding local blocks
        self.channels_last = channels_last

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        x = x if self.channels_last == False else x.permute(0, 3, 1, 2)

        B, C, H, W = x.shape 
        H_out = (H + 2*self.padding - self.patch_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.patch_size) // self.stride + 1

        outputs_by_channel = []

        for channel in range(C):

            y = x[:,channel,:,:].unsqueeze(1) # shape (B, 1, H, W)

            patches = self.unfold(y)  # (B, patch_size^2, L), L is number of patches
            patches = patches.transpose(1, 2)  # (B, L, patch_size^2)

            # Flatten to (B * L, patch_vector)
            patch_vectors = patches.reshape(-1, patches.shape[-1])  # (B*L, patch_dim)

            if patch_vectors.shape[1]+ self.ancilla != self.kernel.circuit.num_qubits :
                raise ValueError(f"Expected input dim {self.kernel.circuit.num_qubits- + self.ancilla}, got {patch_vectors.shape[1]}")

            # Process with the Kernel — ideally batched
            with torch.set_grad_enabled(self.kernel.circuit.trainBool):
                kernel_out = self.kernel(patch_vectors)  # expected output: (B*L, out_dim)

            # Reshape back to (B, L, out_dim)
            kernel_out = kernel_out.view(B, -1, kernel_out.shape[-1])  # (B, L, D)

            # Reshape to (B, D, H_out, W_out)
    
            output = kernel_out.transpose(1, 2).view(B, -1, H_out, W_out)
            outputs_by_channel.append(output)

        return torch.cat(outputs_by_channel, dim=1)  # (B, C×D, H_out, W_out)
    







