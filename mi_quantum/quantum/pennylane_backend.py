import torch
import pennylane as qml
import numpy as np


class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        num_qubits,
        entangle=True,
        trainBool=False,
        graph='chain'
    ):
        super().__init__()
        self.trainBool = trainBool
        self.num_qubits = num_qubits
        self.entangle = entangle
        self.graph_list = {

                    2: {
                        'chain': [[0, 1], [1, 0]],
                        'star' : [[0, 1], [1, 0]]
                    },
                    3: {
                        'chain': [[0, 1], [1, 2], [2, 0]],
                        'star' : [[0, 1], [1, 2], [2, 0]]
                    },
                    4: {
                        'chain': [[0, 1], [1, 2], [2, 3], [3, 0]],
                        'star' : [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
                    },
                    5: {
                        'chain': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
                        'star' : [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2], [1, 3], [2, 4], [3, 0], [4, 1]]
                    },
                    6: {
                        'chain':        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
                        'david_star':   [[0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1] ],
                        'entangled_triangle': [[0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1], [0, 3]],
                        'star' :        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
                                         [0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 2]]
                    },
                    7: {
                        'chain': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]],
                        'star' : [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0],
                                  [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 0], [6, 1]]
                    },
                    8: {
                        'chain': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]],
                        'star' : [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0],
                                  [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 0], [7, 1]]
                    },
                    9: {
                        'chain' : [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0]],
                        'star'  : [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
                                   [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 0], [8, 1]],
                        'king'  : [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                                   [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4]],
                        'center': [[0, 4], [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4]]
                    },
                    10: {
                        'chain': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]],
                        'star' : [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0],
                                  [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 0], [9, 1]]
                    }
                }

        self.graph = self.graph_list[num_qubits][graph] if isinstance(graph,str) else graph if isinstance(graph, list) else None
        if self.graph == None:
            raise ValueError(f'Graph must be a string or a list containing the edges of the graph, but got {graph}. Please enter a valid list or a string such as:\n "chain", "star"... ')
        
        dev = qml.device('default.qubit', wires=num_qubits)

        # Quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit_(inputs, weights):

            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')

            # if self.trainBool:
            #     for qubit in range(num_qubits):
            #         theta, phi, rho = weights[qubit*3 : (qubit+1)*3]
            #         qml.U3(theta, phi, rho, wires=qubit)

            #if self.entangle:
            #    for i, pair in enumerate(self.graph):
            #            qml.CRX(np.pi/3 if not self.trainBool else weights[num_qubits*3 + i], wires=[pair[0], pair[1]]) 

            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits), ranges = [2])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        #qlayer = qml.QNode(circuit_, dev, interface="torch", diff_method="backprop")
        #weight_shape = (num_qubits * 3 + entangle *  len(self.graph),) if trainBool else (1,)
        weight_shape = (1, num_qubits, 3)
        self.magic = qml.qnn.TorchLayer(circuit_, {"weights": weight_shape})

        # Freeze weights (non-trainable)
        if not self.trainBool:
            for param in self.magic.qnode_weights.values():
                param.requires_grad = False


    def forward(self, inputs):
        
        # Vectorized map over time and batch dimensions
        if inputs.ndim == 3:
            # Assume shape is (batch_size, time_steps, num_qubits)
            return torch.vmap(torch.vmap(self.magic))(inputs)
        elif inputs.ndim == 2:
            # Assume shape is (batch_size, num_qubits)
            return torch.vmap(self.magic)(inputs)
        elif inputs.ndim == 1:
            return self.magic(inputs)
        else:
            raise ValueError("Input tensor must be 1D, 2D or 3D")
