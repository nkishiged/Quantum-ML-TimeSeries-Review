"""
Quantum Reservoir Computing (QRC) Reference Implementation
Based on systematic review of QML for time series
"""

import numpy as np

class SimpleQRC:
    def __init__(self, reservoir_size=5, spectral_radius=0.9):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.W_out = None
        
        # Initialize reservoir weights
        self.W_res = np.random.normal(0, 1, (reservoir_size, reservoir_size))
        # Scale to desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= spectral_radius / radius
    
    def step(self, reservoir_state, input_val):
        """One step of reservoir dynamics"""
        new_state = np.tanh(self.W_res @ reservoir_state + input_val)
        return new_state
    
    def fit(self, X, y, washout=100):
        """Train the readout layer"""
        states = []
        current_state = np.zeros(self.reservoir_size)
        
        # Collect reservoir states (washout initial steps)
        for i in range(len(X)):
            current_state = self.step(current_state, X[i])
            if i >= washout:
                states.append(current_state.copy())
        
        states = np.array(states)
        # Ridge regression for readout
        I = 0.01 * np.eye(self.reservoir_size)
        self.W_out = np.linalg.pinv(states.T @ states + I) @ states.T @ y[washout:]
    
    def predict(self, X, initial_state=None):
        """Generate predictions"""
        if initial_state is None:
            current_state = np.zeros(self.reservoir_size)
        else:
            current_state = initial_state.copy()
            
        predictions = []
        for x in X:
            current_state = self.step(current_state, x)
            predictions.append(self.W_out @ current_state)
        
        return np.array(predictions)

if __name__ == "__main__":
    print("Quantum Reservoir Computing Demonstration")
    print("=" * 50)
    
    # Generate simple synthetic data
    t = np.linspace(0, 20, 2000)
    X = np.sin(t) + 0.1 * np.random.normal(size=2000)
    y = np.sin(t + 0.1)  # Next-step prediction
    
    model = SimpleQRC(reservoir_size=5)
    model.fit(X, y, washout=100)
    
    preds = model.predict(X[:100])
    print(f"QRC prediction shape: {preds.shape}")
    print(f"First 5 predictions: {preds[:5].flatten()}")