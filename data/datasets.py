"""Data generation utilities for flow matching experiments."""

import numpy as np
import torch


def create_mixture_of_gaussians(n_samples: int = 1000) -> np.ndarray:
    """Create a 2D mixture of Gaussians target distribution."""
    mean1, cov1 = np.array([2.0, 2.0]), np.array([[0.3, 0.1], [0.1, 0.5]])
    mean2, cov2 = np.array([-1.0, 1.0]), np.array([[0.5, -0.2], [-0.2, 0.3]])
    mean3, cov3 = np.array([0.0, -2.0]), np.array([[0.2, 0.0], [0.0, 0.8]])
    
    n1, n2 = int(0.7 * n_samples), int(0.2 * n_samples)
    n3 = n_samples - n1 - n2
    
    samples = np.vstack([
        np.random.multivariate_normal(mean1, cov1, n1),
        np.random.multivariate_normal(mean2, cov2, n2),
        np.random.multivariate_normal(mean3, cov3, n3)
    ])
    np.random.shuffle(samples)
    return samples


def get_data(n_samples: int = 2000, seed: int = 42):
    """Generate training data: X_0 (noise) and X_1 (target)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X_1_samples = create_mixture_of_gaussians(n_samples)
    X_1_torch = torch.tensor(X_1_samples, dtype=torch.float32)
    X_0_torch = torch.randn(n_samples, 2)
    
    return X_0_torch, X_1_torch, X_1_samples
