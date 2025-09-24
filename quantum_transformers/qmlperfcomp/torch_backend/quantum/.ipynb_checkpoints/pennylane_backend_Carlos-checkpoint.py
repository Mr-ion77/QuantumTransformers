import torch

import pennylane as qml


class QuantumLayer(torch.nn.Module):
    def __init__(self, num_qubits, num_qlayers=1): #qdevice='default.qubit', diff_method="best"):
        super().__init__()
        
        dev = qml.device('default.qubit.torch', wires=num_qubits, torch_device="cuda:4")
        @qml.qnode(dev, interface='torch', diff_method="backprop")
        def circuit(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
                qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

        qlayer = qml.QNode(circuit, dev, interface="torch", diff_method="backprop") # diff_method=diff_method)
        self.linear = qml.qnn.TorchLayer(qlayer, {"weights": (num_qlayers, num_qubits)})
    
    def forward(self, inputs):
        return self.linear(inputs)



class QuantumLayer_2(torch.nn.Module):
    def __init__(self, num_qubits, num_qlayers=1): #qdevice='default.qubit', diff_method="best"):
        super().__init__()
        
        dev = qml.device('default.qubit.torch', wires=num_qubits, torch_device="cuda:4")
        @qml.qnode(dev, interface='torch', diff_method="backprop")
        def circuit(inputs, weights):
            # DIGITAL ENCODING OF THE PIXELS
            for idx in range(n_qubits):
                qml.RY(inputs, wires=range(num_qubits))
                qml.Hadamard(wires=range(num_qubits))
            # ENTANGLEMENT
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

        qlayer = qml.QNode(circuit, dev, interface="torch", diff_method="backprop") # diff_method=diff_method)
        self.linear = qml.qnn.TorchLayer(qlayer, {"weights": (num_qlayers, num_qubits)})
    
    def forward(self, inputs):
        return self.linear(inputs)







