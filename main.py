"""
Flow Matching with Learned Stochastic Interpolation
====================================================
Stage 1: Train interpolant X_t^φ = tX_1 + (1-t)F^φ_t(X_0, X_1) to minimize E[||∇²log p||²]
Stage 2: Train velocity u_t^θ with loss E[||u_t^θ(X_t^φ) - ∂_t X_t^φ||²]
Sampling: Start from F_0^φ(X_0) and solve ODE dx/dt = u_t^θ(x)
"""

from data.datasets import get_data
from models.networks import FlowInterpolant, FlowMatchingModel, GaussianFlowMatching
from training import train_interpolant, train_flow_matching_learned, train_flow_matching_gaussian
from evaluation import (
    visualize_samples,
    visualize_trajectories,
    visualize_training_curves,
    visualize_wasserstein_comparison,
    visualize_destruction_from_data,
    print_results,
    run_evaluation
)


def main():
    # ============ DATA ============
    X_0_torch, X_1_torch, X_1_samples = get_data(n_samples=1000, seed=42)

    # ============ STAGE 1: Train Interpolant ============
    flow_interpolant = FlowInterpolant(dim=2)
    gaussian_flow_matching = GaussianFlowMatching(dim=2)
    loss_history, flow_norms = train_interpolant(
        flow_interpolant, X_0_torch, X_1_torch,
        num_epochs=100, lr=3e-4,
        lambda_dt_logp=1.0,       # Time derivative loss
        lambda_grad_logp=0.0,      # Gradient norm regularization
        lambda_hessian_trace=0.0,  # Hutchinson trace estimator
        lambda_hessian_frob=0.0,   # Hessian Frobenius norm
        lambda_flow_reg=1.0,       # Flow regularization
    )

    visualize_wasserstein_comparison(flow_interpolant, gaussian_flow_matching, X_0_torch, X_1_torch, X_1_samples, n_time_steps=50)

    # ============ STAGE 2: Train Flow Matching ============
    flow_matching_learned = FlowMatchingModel(dim=2, interpolant=flow_interpolant)
    fm_losses_learned = train_flow_matching_learned(
        flow_matching_learned, X_0_torch, X_1_torch,
        num_epochs=200, lr=2e-3, batch_size=256
    )

    fm_losses_gaussian = train_flow_matching_gaussian(
        gaussian_flow_matching, X_1_torch,
        num_epochs=200, lr=2e-3, batch_size=256
    )

    # ============ EVALUATION ============
    samples_learned, samples_gaussian, metrics_learned, metrics_gaussian = run_evaluation(
        flow_matching_learned, gaussian_flow_matching, X_1_samples, n_eval=1000
    )

    print_results(metrics_learned, metrics_gaussian)

    # ============ VISUALIZATION ============
    visualize_training_curves(loss_history, fm_losses_learned, fm_losses_gaussian, flow_norms)
    visualize_samples(samples_learned, samples_gaussian, X_1_samples, metrics_learned, metrics_gaussian)
    visualize_trajectories(flow_matching_learned, gaussian_flow_matching, X_1_samples, n_traj=30)

    visualize_destruction_from_data(flow_interpolant, gaussian_flow_matching, X_1_samples, n_samples=500)

    print("\nAll visualizations saved!")


if __name__ == "__main__":
    main()
