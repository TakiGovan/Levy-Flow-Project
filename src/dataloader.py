import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_sp500_data(data_path='data/sp500_returns.csv', train_ratio=0.8):
    # Read saved data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = df['^GSPC'].values
    
    # Standardize
    mean = returns.mean()
    std = returns.std()
    returns_standardized = (returns - mean) / std
    
    # Split
    split_idx = int(train_ratio * len(returns_standardized))
    train_data = returns_standardized[:split_idx]
    test_data = returns_standardized[split_idx:]
    
    return train_data, test_data, mean, std

def visualise_data(train_data=None, test_data=None):
    plt.figure(figsize=(10, 4))
    plt.hist(train_data, bins=199, density=True)
    plt.title(f'SnP 500 Returns (Train: {len(train_data)}, Test: {len(test_data)})')
    plt.savefig('../results/SnP_returns_histogram.png')
    plt.show()

    print(f"Mean: {train_data.mean()}, Std: {train_data.std()}")