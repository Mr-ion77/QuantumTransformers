import torch
import pennylane as qml

class QuantumLayer(torch.nn.Module):
    def __init__(self, num_qubits, num_qlayers=2):
        super().__init__()

        # Define the quantum device with the specified number of qubits and GPU device
        dev = qml.device('default.qubit', wires=num_qubits, torch_device="cuda:0") # before was default.qubit.torch
        
        # Define the quantum node (QNode)
        @qml.qnode(dev, interface='torch', diff_method="backprop")
        def circuit(inputs, weights):
            # Apply RY rotation followed by Hadamard 
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
            for idx in range(num_qubits):
                qml.Hadamard(wires=idx)
            
            # Apply entanglement using the specified weights
           # qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
            qml.templates.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))

            # Measure PauliZ expectation values for each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        
        # Integrate the QNode with PyTorch using TorchLayer
        qlayer = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        self.linear = qml.qnn.TorchLayer(qlayer, {"weights": (num_qlayers, num_qubits,3)})
    
    def forward(self, inputs):
        return self.linear(inputs)






