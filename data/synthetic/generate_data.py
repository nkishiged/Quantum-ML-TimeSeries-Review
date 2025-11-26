"""
Synthetic Time Series Data Generation
For demonstration and testing of QML models
"""

import numpy as np
import pandas as pd

def generate_lorenz(n_steps=10000, dt=0.01, sigma=10, beta=8/3, rho=28):
    """Generate Lorenz system time series"""
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    
    # Initial conditions
    x[0], y[0], z[0] = 1, 1, 1
    
    for i in range(1, n_steps):
        dx = sigma * (y[i-1] - x[i-1]) * dt
        dy = (x[i-1] * (rho - z[i-1]) - y[i-1]) * dt
        dz = (x[i-1] * y[i-1] - beta * z[i-1]) * dt
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        z[i] = z[i-1] + dz
    
    return x, y, z

def generate_mackey_glass(n_steps=10000, tau=17, n=10, beta=0.2, gamma=0.1, dt=1.0):
    """Generate Mackey-Glass time series"""
    history = np.ones(n_steps + tau) * 0.5
    for i in range(tau, n_steps + tau):
        history[i] = history[i-1] + dt * (beta * history[i-tau] / (1 + history[i-tau]**n) - gamma * history[i-1])
    return history[tau:]

def generate_synthetic_financial(n_steps=5000, volatility=0.01, drift=0.001):
    """Generate synthetic financial time series"""
    returns = np.random.normal(drift, volatility, n_steps)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices

if __name__ == "__main__":
    print("Generating Synthetic Time Series Data")
    print("=" * 50)
    
    # Generate Lorenz system
    x, y, z = generate_lorenz(5000)
    lorenz_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    lorenz_df.to_csv('data/synthetic/lorenz_series.csv', index=False)
    print("✓ Lorenz system data generated")
    
    # Generate Mackey-Glass
    mg_series = generate_mackey_glass(5000)
    mg_df = pd.DataFrame({'value': mg_series})
    mg_df.to_csv('data/synthetic/mackey_glass.csv', index=False)
    print("✓ Mackey-Glass data generated")
    
    # Generate financial data
    financial_series = generate_synthetic_financial(2000)
    financial_df = pd.DataFrame({'price': financial_series})
    financial_df.to_csv('data/synthetic/financial_series.csv', index=False)
    print("✓ Synthetic financial data generated")
    
    print("\nAll synthetic data saved to data/synthetic/")