"""
Quantum Neural Network (QNN) Reference Implementation
Based on literature analysis from systematic review
"""

import pennylane as qml
import numpy as np

class SimpleQNN:
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize random parameters
        self.params = np.random.normal(0, np.pi, (n_layers, n_qubits, 3))
    
    @qml.qnode(dev)
    def quantum_circuit(self, inputs, params):
        """Variational quantum circuit for QNN"""
        # Encode input data
        for i in range(self.n_qubits):
            qml.RY(inputs[i % len(inputs)], wires=i)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Single qubit rotations
            for i in range(self.n_qubits):
                qml.Rot(*params[layer, i], wires=i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward pass through quantum circuit"""
        return np.array(self.quantum_circuit(x, self.params))
    
    def __call__(self, x):
        return self.forward(x)

if __name__ == "__main__":
    print("Quantum Neural Network Demonstration")
    print("=" * 50)
    
    model = SimpleQNN(n_qubits=4, n_layers=2)
    x_test = np.random.normal(0, 1, 4)
    output = model(x_test)
    
    print(f"Input: {x_test}")
    print(f"QNN output: {output}")
    print(f"Output shape: {output.shape}")