"""
Train only the interpolant and visualize the destruction process and Wasserstein-2 evolution.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.networks import FlowInterpolant
from data.datasets import get_data
from training import train_interpolant
from evaluation import visualize_destruction_from_data, visualize_wasserstein_comparison
import os

FIGURES_DIR = 'outputs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    # Load data (returns torch tensors)
    X_0_torch, X_1_torch, X_1_samples = get_data(n_samples=10000)

    # Initialize interpolant
    flow_interpolant = FlowInterpolant(dim=X_0_torch.shape[1])

    # ===== CONFIGURE LOSS WEIGHTS HERE =====
    # Options:
    #   lambda_dt_logp: (∂_t log p_t)² - time derivative of log prob
    #   lambda_grad_logp: ||∇ log p_t||² - gradient norm
    #   lambda_hessian_trace: (tr(∇² log p_t))² - Hessian trace via Hutchinson
    #   lambda_hessian_frob: ||∇² log p_t||_F² - Hessian Frobenius norm
    #   lambda_flow_reg: ||F||² - flow output regularization

    loss_config = {
        'lambda_dt_logp': 1.0,        # Time derivative loss
        'lambda_grad_logp': 0.0,       # Gradient norm regularization
        'lambda_hessian_trace': 0.0,   # Hutchinson trace estimator
        'lambda_hessian_frob': 0.0,    # Hessian Frobenius norm
        'lambda_flow_reg': 1.0,        # Flow regularization
        'n_hutchinson_samples': 1,     # Samples for Hutchinson estimator
    }

    # Train interpolant
    loss_history, flow_norms = train_interpolant(
        flow_interpolant, X_0_torch, X_1_torch,
        num_epochs=400, lr=3e-4, **loss_config
    )

    # Plot interpolant training losses
    active_losses = [(k, v) for k, v in loss_history.items()
                     if k not in ['total', 'flow_norm'] and max(v) > 0]

    n_plots = len(active_losses) + 1  # +1 for flow norm
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Plot each active loss
    loss_labels = {
        'dt_logp': ('(∂_t log p)²', 'Time Derivative Loss'),
        'grad_logp': ('||∇log p||²', 'Gradient Norm'),
        'hessian_trace': ('tr(H)²', 'Hessian Trace (Hutchinson)'),
        'hessian_frob': ('||H||_F²', 'Hessian Frobenius Norm'),
    }

    for i, (key, values) in enumerate(active_losses):
        ylabel, title = loss_labels.get(key, (key, key))
        axes[i].plot(values, 'b-', linewidth=2)
        axes[i].set_xlabel('Epoch', fontsize=11)
        axes[i].set_ylabel(ylabel, fontsize=11)
        axes[i].set_title(f'Interpolant Training: {title}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    # Plot flow norm
    axes[-1].plot(flow_norms, 'r-', linewidth=2)
    axes[-1].set_xlabel('Epoch', fontsize=11)
    axes[-1].set_ylabel('||F||² Regularization', fontsize=11)
    axes[-1].set_title('Interpolant Training: Flow Norm', fontsize=12, fontweight='bold')
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/interpolant_training_losses.png', dpi=150, bbox_inches='tight')
    print(f"Saved '{FIGURES_DIR}/interpolant_training_losses.png'")
    plt.close()

    # Visualize destruction process (interpolation only)
    visualize_destruction_from_data(flow_interpolant, None, X_1_samples, n_samples=500, t_steps=5)

    # Visualize Wasserstein-2 evolution (interpolation only)
    visualize_wasserstein_comparison(flow_interpolant, None, X_0_torch, X_1_torch, X_1_samples, n_time_steps=50)

    print("\nDestruction process and Wasserstein-2 visualizations saved!")


if __name__ == "__main__":
    main()
