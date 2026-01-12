"""Utility functions for flow matching."""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_wasserstein2(samples1: np.ndarray, samples2: np.ndarray, n_subsample: int = 500) -> float:
    """Compute Wasserstein-2 distance between two sample sets."""
    n = min(len(samples1), len(samples2), n_subsample)
    s1 = samples1[np.random.choice(len(samples1), n, replace=False)]
    s2 = samples2[np.random.choice(len(samples2), n, replace=False)]
    cost = cdist(s1, s2, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind] ** 2))


def compute_hessian_frobenius_norm(log_prob: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius norm of Hessian of log probability."""
    batch_size, dim = x.shape
    hessian_norm_sq = torch.zeros(batch_size, device=x.device)

    grads = torch.autograd.grad(log_prob, x, torch.ones_like(log_prob),
                                 create_graph=True, retain_graph=True)[0]

    for i in range(dim):
        for j in range(dim):
            grad_outputs = torch.zeros_like(grads)
            grad_outputs[:, i] = 1.0
            second_derivs = torch.autograd.grad(grads, x, grad_outputs,
                                                  create_graph=True, retain_graph=True)[0][:, j]
            hessian_norm_sq += second_derivs ** 2

    return hessian_norm_sq


def hutchinson_trace_estimator(log_probs: torch.Tensor, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """Estimate tr(∇²log p) using Hutchinson's trace estimator.

    Uses the identity: tr(H) = E[v^T H v] where v is Gaussian random vector.
    This computes v^T ∇(∇log p · v) which equals v^T H v.

    Args:
        log_probs: Log probabilities, shape (batch_size,)
        x: Input points, shape (batch_size, dim)
        n_samples: Number of random vectors for estimation (default 1)

    Returns:
        Estimated trace of Hessian, shape (batch_size,)
    """
    batch_size, dim = x.shape
    trace_est = torch.zeros(batch_size, device=x.device)

    # Compute gradient ∇log p
    grad_log_p = torch.autograd.grad(log_probs.sum(), x, create_graph=True)[0]

    for _ in range(n_samples):
        # Gaussian random vector: v ~ N(0, I)
        v = torch.randn(batch_size, dim, device=x.device)

        # Compute v^T ∇log p (dot product per sample)
        grad_v = (grad_log_p * v).sum(dim=1)

        # Compute ∇(v^T ∇log p) w.r.t. x, then dot with v to get v^T H v
        hvp = torch.autograd.grad(grad_v.sum(), x, create_graph=True)[0]
        trace_est += (hvp * v).sum(dim=1)

    return trace_est / n_samples


def evaluate_samples(samples: np.ndarray, target: np.ndarray, name: str) -> dict:
    """Evaluate quality of generated samples against target."""
    w2 = compute_wasserstein2(samples, target)
    mean_dist = np.linalg.norm(samples.mean(axis=0) - target.mean(axis=0))
    std_ratio = np.std(samples) / np.std(target)
    return {'name': name, 'W2': w2, 'mean_dist': mean_dist, 'std_ratio': std_ratio}
