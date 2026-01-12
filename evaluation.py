def visualize_destruction_from_data(flow_interpolant, gaussian_model, X_1_samples, n_samples=500, t_steps=5):
    """Plot X_t at 5 different time points using only the interpolation (no ODE)."""
    import matplotlib.cm as cm

    # Use exactly 5 time points
    times = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Sample data
    idx = np.random.choice(len(X_1_samples), n_samples, replace=False)
    X_1 = torch.tensor(X_1_samples[idx], dtype=torch.float32)
    X_0 = torch.randn(n_samples, 2)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Learned interpolant destruction process (if provided)
    if flow_interpolant is not None:
        for i, t in enumerate(times):
            with torch.no_grad():
                X_t = flow_interpolant.sample_X_t(X_0, X_1, t)
            X_t_np = X_t.detach().cpu().numpy()

            axes[0, i].scatter(X_t_np[:, 0], X_t_np[:, 1], alpha=0.5, s=8, c='green')
            axes[0, i].set_title(f'Learned: t={t:.2f}', fontsize=12, fontweight='bold')
            axes[0, i].set_xlim(-4, 4)
            axes[0, i].set_ylim(-4, 4)
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True, alpha=0.3)
    else:
        for i in range(5):
            axes[0, i].text(0.5, 0.5, 'No model', ha='center', va='center', transform=axes[0, i].transAxes)

    # Gaussian interpolation (baseline)
    for i, t in enumerate(times):
        with torch.no_grad():
            X_t = t * X_1 + (1 - t) * X_0
        X_t_np = X_t.detach().cpu().numpy()

        axes[1, i].scatter(X_t_np[:, 0], X_t_np[:, 1], alpha=0.5, s=8, c='blue')
        axes[1, i].set_title(f'Gaussian: t={t:.2f}', fontsize=12, fontweight='bold')
        axes[1, i].set_xlim(-4, 4)
        axes[1, i].set_ylim(-4, 4)
        axes[1, i].set_aspect('equal')
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle('Destruction Process X_t (Interpolation Only)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/destruction_process_interpolation.png', dpi=150, bbox_inches='tight')
    print(f"Saved '{FIGURES_DIR}/destruction_process_interpolation.png'")
    plt.close()
"""Evaluation and visualization functions."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.networks import FlowMatchingModel, GaussianFlowMatching, FlowInterpolant
from utils import evaluate_samples

FIGURES_DIR = 'outputs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def visualize_samples(samples_learned: np.ndarray, samples_gaussian: np.ndarray,
                      target: np.ndarray, metrics_learned: dict, metrics_gaussian: dict):
    """Visualize generated samples comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(target[:, 0], target[:, 1], alpha=0.5, s=10, c='red', label='Target X₁')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].set_title('Target Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(samples_learned[:, 0], samples_learned[:, 1], alpha=0.5, s=10, c='green')
    axes[1].scatter(target[:300, 0], target[:300, 1], alpha=0.2, s=5, c='red')
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Learned Interpolant FM\nW₂={metrics_learned["W2"]:.4f}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(samples_gaussian[:, 0], samples_gaussian[:, 1], alpha=0.5, s=10, c='blue')
    axes[2].scatter(target[:300, 0], target[:300, 1], alpha=0.2, s=5, c='red')
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal')
    axes[2].set_title(f'Gaussian FM\nW₂={metrics_gaussian["W2"]:.4f}', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/flow_matching_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved '{FIGURES_DIR}/flow_matching_comparison.png'")
    plt.close()


def visualize_trajectories(flow_model: FlowMatchingModel, gaussian_model: GaussianFlowMatching,
                           target: np.ndarray, n_traj: int = 20):
    """Visualize ODE trajectories for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Learned interpolant trajectories
    with torch.no_grad():
        x = flow_model.interpolant.sample_initial(n_traj)
    trajectories_learned = [x.numpy().copy()]
    num_steps = 50
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.full((n_traj,), t)
        with torch.no_grad():
            v = flow_model.velocity_net(x, t_tensor)
        x = x + dt * v
        if step % 5 == 0:
            trajectories_learned.append(x.numpy().copy())

    axes[0].scatter(target[:, 0], target[:, 1], alpha=0.2, s=5, c='red')
    for traj_points in zip(*trajectories_learned):
        traj = np.array(traj_points)
        axes[0].plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.5, linewidth=0.8)
    axes[0].scatter(trajectories_learned[0][:, 0], trajectories_learned[0][:, 1], s=30, c='blue', marker='o', zorder=5)
    axes[0].scatter(trajectories_learned[-1][:, 0], trajectories_learned[-1][:, 1], s=30, c='green', marker='x', zorder=5)
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].set_title('Learned Interpolant FM Trajectories', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Gaussian flow matching trajectories
    x = torch.randn(n_traj, 2)
    trajectories_gaussian = [x.numpy().copy()]

    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.full((n_traj,), t)
        with torch.no_grad():
            v = gaussian_model.velocity_net(x, t_tensor)
        x = x + dt * v
        if step % 5 == 0:
            trajectories_gaussian.append(x.numpy().copy())

    axes[1].scatter(target[:, 0], target[:, 1], alpha=0.2, s=5, c='red')
    for traj_points in zip(*trajectories_gaussian):
        traj = np.array(traj_points)
        axes[1].plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=0.8)
    axes[1].scatter(trajectories_gaussian[0][:, 0], trajectories_gaussian[0][:, 1], s=30, c='blue', marker='o', zorder=5)
    axes[1].scatter(trajectories_gaussian[-1][:, 0], trajectories_gaussian[-1][:, 1], s=30, c='green', marker='x', zorder=5)
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    axes[1].set_title('Gaussian FM Trajectories', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/trajectories_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved '{FIGURES_DIR}/trajectories_comparison.png'")
    plt.close()


def visualize_training_curves(loss_history: dict, fm_losses_learned: list,
                               fm_losses_gaussian: list, flow_norms: list):
    """Plot training curves for all stages.

    Args:
        loss_history: Dict with keys like 'dt_logp', 'grad_logp', 'hessian_trace', etc.
        fm_losses_learned: Flow matching losses for learned interpolant
        fm_losses_gaussian: Flow matching losses for Gaussian baseline
        flow_norms: ||F||² regularization values
    """
    # Determine which interpolant losses are active
    loss_labels = {
        'dt_logp': ('(∂_t log p)²', 'Time Derivative'),
        'grad_logp': ('||∇log p||²', 'Gradient Norm'),
        'hessian_trace': ('tr(H)²', 'Hessian Trace'),
        'hessian_frob': ('||H||_F²', 'Hessian Frobenius'),
    }

    active_losses = [(k, v) for k, v in loss_history.items()
                     if k in loss_labels and max(v) > 0]

    n_interp_plots = max(1, len(active_losses))
    fig, axes = plt.subplots(1, n_interp_plots + 2, figsize=(5 * (n_interp_plots + 2), 5))

    # Plot interpolant losses
    if active_losses:
        for i, (key, values) in enumerate(active_losses):
            ylabel, title = loss_labels[key]
            axes[i].plot(values, 'b-', linewidth=1.5)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f'Stage 1: {title}')
            if min(values) > 0:
                axes[i].set_yscale('log')
            axes[i].grid(True, alpha=0.3)
    else:
        # Default: plot total loss
        axes[0].plot(loss_history.get('total', [0]), 'b-', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Stage 1: Interpolant Training')
        axes[0].grid(True, alpha=0.3)

    # Plot flow norm
    axes[-2].plot(flow_norms, 'r-', linewidth=1.5)
    axes[-2].set_xlabel('Epoch')
    axes[-2].set_ylabel('||F_t^φ||²')
    axes[-2].set_title('Flow Norm Regularization')
    axes[-2].grid(True, alpha=0.3)

    # Plot flow matching losses
    axes[-1].plot(fm_losses_learned, label='Learned Interp.', linewidth=1.5)
    axes[-1].plot(fm_losses_gaussian, label='Gaussian', linewidth=1.5)
    axes[-1].set_xlabel('Epoch')
    axes[-1].set_ylabel('Flow Matching Loss')
    axes[-1].set_title('Stage 2: Flow Matching Training')
    axes[-1].set_yscale('log')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved '{FIGURES_DIR}/training_curves.png'")
    plt.close()


def print_results(metrics_learned: dict, metrics_gaussian: dict):
    """Print comparison results."""
    print(f"\n{'='*50}")
    print("RESULTS COMPARISON")
    print(f"{'='*50}")
    print(f"{'Method':<25} {'W₂':>10} {'Mean Dist':>12} {'Std Ratio':>12}")
    print(f"{'-'*50}")
    print(f"{'Learned Interpolant FM':<25} {metrics_learned['W2']:>10.4f} {metrics_learned['mean_dist']:>12.4f} {metrics_learned['std_ratio']:>12.4f}")
    print(f"{'Gaussian FM':<25} {metrics_gaussian['W2']:>10.4f} {metrics_gaussian['mean_dist']:>12.4f} {metrics_gaussian['std_ratio']:>12.4f}")
    print(f"{'='*50}")

    if metrics_learned['W2'] < metrics_gaussian['W2']:
        improvement = (1 - metrics_learned['W2']/metrics_gaussian['W2']) * 100
        print(f"✓ Learned Interpolant FM is {improvement:.1f}% better in W₂!")
    else:
        print(f"Gaussian FM has lower W₂ distance")


def run_evaluation(flow_model: FlowMatchingModel, gaussian_model: GaussianFlowMatching,
                   target_samples: np.ndarray, n_eval: int = 1000):
    """Run full evaluation and return metrics."""
    print("\n--- Generating Samples ---")

    samples_learned = flow_model.sample(n_eval, num_steps=100).numpy()
    samples_gaussian = gaussian_model.sample(n_eval, num_steps=100).numpy()

    metrics_learned = evaluate_samples(samples_learned, target_samples, "Learned Interpolant FM")
    metrics_gaussian = evaluate_samples(samples_gaussian, target_samples, "Gaussian FM")

    return samples_learned, samples_gaussian, metrics_learned, metrics_gaussian


def visualize_wasserstein_comparison(flow_interpolant: FlowInterpolant, gaussian_model: GaussianFlowMatching,
                                      X_0: torch.Tensor, X_1: torch.Tensor, target_samples: np.ndarray,
                                      n_time_steps: int = 50, n_samples: int = 500):
    """Visualize W₂ distance evolution over time using INTERPOLATION ONLY (no ODE)."""
    from utils import compute_wasserstein2

    times = np.linspace(0, 1, n_time_steps + 1)
    w2_learned_to_target = []
    w2_gaussian_to_target = []
    w2_learned_consecutive = []
    w2_gaussian_consecutive = []

    prev_samples_learned = None
    prev_samples_gaussian = None

    print("\n--- Computing Wasserstein distances over time (INTERPOLATION ONLY) ---")

    # Sample X_0 and X_1 for interpolation
    idx = np.random.choice(len(X_0), n_samples, replace=False)
    X_0_batch = X_0[idx]
    X_1_batch = X_1[idx]

    for i, t in enumerate(times):
        # Learned interpolant: X_t = tX_1 + (1-t)F_t(X_0, X_1)
        with torch.no_grad():
            samples_learned_t = flow_interpolant.sample_X_t(X_0_batch, X_1_batch, t).numpy()

        # Gaussian interpolation: X_t = tX_1 + (1-t)X_0
        with torch.no_grad():
            samples_gaussian_t = (t * X_1_batch + (1 - t) * X_0_batch).numpy()

        # W2 to target
        w2_learned_to_target.append(compute_wasserstein2(samples_learned_t, target_samples))
        w2_gaussian_to_target.append(compute_wasserstein2(samples_gaussian_t, target_samples))

        # W2 between consecutive time steps
        if prev_samples_learned is not None:
            w2_learned_consecutive.append(compute_wasserstein2(samples_learned_t, prev_samples_learned))
            w2_gaussian_consecutive.append(compute_wasserstein2(samples_gaussian_t, prev_samples_gaussian))

        prev_samples_learned = samples_learned_t.copy()
        prev_samples_gaussian = samples_gaussian_t.copy()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # W2 to target over time
    axes[0].plot(times, w2_learned_to_target, 'g-o', label='Learned Interpolant', linewidth=2, markersize=6)
    axes[0].plot(times, w2_gaussian_to_target, 'b-s', label='Gaussian FM', linewidth=2, markersize=6)
    axes[0].set_xlabel('Time t', fontsize=11)
    axes[0].set_ylabel('W₂(p_t, p_target)', fontsize=11)
    axes[0].set_title('W₂ Distance to Target Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # W2 between consecutive steps
    time_mid = (times[:-1] + times[1:]) / 2
    axes[1].plot(time_mid, w2_learned_consecutive, 'g-o', label='Learned Interpolant', linewidth=2, markersize=6)
    axes[1].plot(time_mid, w2_gaussian_consecutive, 'b-s', label='Gaussian FM', linewidth=2, markersize=6)
    axes[1].set_xlabel('Time t', fontsize=11)
    axes[1].set_ylabel('W₂(p_{t+Δt}, p_t)', fontsize=11)
    axes[1].set_title('W₂ Distance Between Consecutive Steps', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/wasserstein_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved '{FIGURES_DIR}/wasserstein_comparison.png'")
    plt.close()

    return {
        'learned_final_w2': w2_learned_to_target[-1],
        'gaussian_final_w2': w2_gaussian_to_target[-1],
        'learned_cumulative': sum(w2_learned_consecutive),
        'gaussian_cumulative': sum(w2_gaussian_consecutive)
    }
