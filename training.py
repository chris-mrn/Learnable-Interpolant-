"""Training functions for interpolant and flow matching."""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple

from models.networks import FlowInterpolant, FlowMatchingModel, GaussianFlowMatching, VelocityNetwork
from utils import hutchinson_trace_estimator, compute_hessian_frobenius_norm


def train_interpolant(flow_interpolant: FlowInterpolant, X_0: torch.Tensor, X_1: torch.Tensor,
                      num_epochs: int = 200, lr: float = 3e-4,
                      lambda_flow_reg: float = 1.0,
                      lambda_dt_logp: float = 0.0,
                      lambda_grad_logp: float = 0.0,
                      lambda_hessian_trace: float = 0.0,
                      lambda_hessian_frob: float = 0.0,
                      n_hutchinson_samples: int = 1) -> Tuple[Dict[str, List[float]], List[float]]:
    """Stage 1: Train interpolant with configurable loss terms.

    Loss = λ_dt * E[(∂_t log p_t)²]           # Time derivative loss
         + λ_grad * E[||∇ log p_t||²]          # Gradient norm regularization
         + λ_trace * E[(tr(∇² log p_t))²]      # Hessian trace (Hutchinson estimator)
         + λ_frob * E[||∇² log p_t||_F²]       # Hessian Frobenius norm
         + λ_f * E[||F||²]                     # Flow regularization

    Args:
        flow_interpolant: The flow interpolant to train
        X_0: Source samples (noise)
        X_1: Target samples (data)
        num_epochs: Number of training epochs
        lr: Learning rate
        lambda_flow_reg: Weight for ||F||² regularization
        lambda_dt_logp: Weight for (∂_t log p_t)² loss
        lambda_grad_logp: Weight for ||∇ log p_t||² loss
        lambda_hessian_trace: Weight for (tr(∇² log p_t))² loss (Hutchinson)
        lambda_hessian_frob: Weight for ||∇² log p_t||_F² loss
        n_hutchinson_samples: Number of samples for Hutchinson trace estimator

    Returns:
        loss_history: Dictionary of loss component histories
        flow_norms: History of ||F||² values
    """
    optimizer = optim.AdamW(flow_interpolant.flow.parameters(), lr=lr, weight_decay=1e-4)

    # Track all loss components
    loss_history = {
        'total': [],
        'dt_logp': [],
        'grad_logp': [],
        'hessian_trace': [],
        'hessian_frob': [],
        'flow_norm': []
    }
    flow_norms = []

    # EMA for stable loss tracking
    ema_losses = {k: None for k in loss_history.keys()}
    ema_decay = 0.9

    # Build description of active losses
    active_losses = []
    if lambda_dt_logp > 0:
        active_losses.append(f"λ_dt={lambda_dt_logp}")
    if lambda_grad_logp > 0:
        active_losses.append(f"λ_grad={lambda_grad_logp}")
    if lambda_hessian_trace > 0:
        active_losses.append(f"λ_trace={lambda_hessian_trace}")
    if lambda_hessian_frob > 0:
        active_losses.append(f"λ_frob={lambda_hessian_frob}")
    if lambda_flow_reg > 0:
        active_losses.append(f"λ_flow={lambda_flow_reg}")

    print(f"--- Stage 1: Interpolant Training ---")
    print(f"Epochs={num_epochs}, LR={lr}")
    print(f"Active losses: {', '.join(active_losses)}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        batch_size = min(512, len(X_0))
        indices = torch.randperm(len(X_0))[:batch_size]
        X_0_batch, X_1_batch = X_0[indices], X_1[indices]

        # Sample different t for each element in the batch
        t_batch = torch.rand(batch_size) * 0.88 + 0.01  # Uniform in [0.01, 0.99]
        t_batch = torch.clamp(t_batch, 0.01, 0.99)

        # Compute per-sample weights
        wt_batch = 1.0 / (1 - t_batch)**2  # Shape: (batch_size,)

        # Track per-sample losses for weighting
        per_sample_losses = torch.zeros(batch_size)
        current_losses = {k: 0.0 for k in loss_history.keys()}

        # ===== Flow Norm Regularization =====
        # Need to compute per-sample since t is different for each
        F_t_list = []
        for i in range(batch_size):
            F_t_i, _ = flow_interpolant.flow(X_0_batch[i:i+1], X_1_batch[i:i+1], t_batch[i].item())
            F_t_list.append(F_t_i)
        F_t = torch.cat(F_t_list, dim=0)
        flow_norm_per_sample = torch.norm(F_t, dim=1).pow(2)
        flow_norms.append(flow_norm_per_sample.mean().item())
        current_losses['flow_norm'] = flow_norm_per_sample.mean().item()

        if lambda_flow_reg > 0:
            per_sample_losses = per_sample_losses + lambda_flow_reg * flow_norm_per_sample

        # ===== Time Derivative Loss: (∂_t log p_t)² =====
        if lambda_dt_logp > 0:
            dt_log_prob_list = []
            for i in range(batch_size):
                dt_log_prob_i = flow_interpolant.compute_dt_log_prob(
                    X_0_batch[i:i+1], X_1_batch[i:i+1], t_batch[i].item()
                )
                dt_log_prob_list.append(dt_log_prob_i)
            dt_log_prob = torch.cat(dt_log_prob_list, dim=0)
            dt_log_prob = torch.clamp(dt_log_prob, -100.0, 100.0)
            dt_logp_per_sample = dt_log_prob ** 2
            current_losses['dt_logp'] = dt_logp_per_sample.mean().item()
            per_sample_losses = per_sample_losses + lambda_dt_logp * dt_logp_per_sample

        # ===== Gradient Norm Loss: ||∇ log p_t||² =====
        if lambda_grad_logp > 0:
            grad_norm_sq_list = []
            for i in range(batch_size):
                grad_norm_sq_i = flow_interpolant.compute_grad_log_prob(
                    X_0_batch[i:i+1], X_1_batch[i:i+1], t_batch[i].item()
                )
                grad_norm_sq_list.append(grad_norm_sq_i)
            grad_norm_sq = torch.cat(grad_norm_sq_list, dim=0)
            current_losses['grad_logp'] = grad_norm_sq.mean().item()
            per_sample_losses = per_sample_losses + lambda_grad_logp * grad_norm_sq

        # ===== Hessian Trace Loss (Hutchinson estimator): (tr(∇² log p_t))² =====
        if lambda_hessian_trace > 0:
            hessian_trace_list = []
            for i in range(batch_size):
                X_t_i, log_pt_i = flow_interpolant.compute_log_prob_for_hessian(
                    X_0_batch[i:i+1], X_1_batch[i:i+1], t_batch[i].item()
                )
                trace_est_i = hutchinson_trace_estimator(log_pt_i, X_t_i, n_samples=n_hutchinson_samples)
                hessian_trace_list.append(trace_est_i)
            trace_est = torch.cat(hessian_trace_list, dim=0)
            trace_est = torch.clamp(trace_est, -1000.0, 1000.0)
            hessian_trace_per_sample = trace_est ** 2
            current_losses['hessian_trace'] = hessian_trace_per_sample.mean().item()
            per_sample_losses = per_sample_losses + lambda_hessian_trace * hessian_trace_per_sample

        # ===== Hessian Frobenius Norm Loss: ||∇² log p_t||_F² =====
        if lambda_hessian_frob > 0:
            hessian_frob_list = []
            for i in range(batch_size):
                X_t_i, log_pt_i = flow_interpolant.compute_log_prob_for_hessian(
                    X_0_batch[i:i+1], X_1_batch[i:i+1], t_batch[i].item()
                )
                hessian_frob_i = compute_hessian_frobenius_norm(log_pt_i, X_t_i)
                hessian_frob_list.append(hessian_frob_i)
            hessian_frob = torch.cat(hessian_frob_list, dim=0)
            hessian_frob = torch.clamp(hessian_frob, 0.0, 10000.0)
            current_losses['hessian_frob'] = hessian_frob.mean().item()
            per_sample_losses = per_sample_losses + lambda_hessian_frob * hessian_frob

        # Apply per-sample time-dependent weighting, then average
        weighted_losses = wt_batch * per_sample_losses
        total_loss = weighted_losses.mean()
        current_losses['total'] = total_loss.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_interpolant.flow.parameters(), 1.0)
        optimizer.step()

        # EMA smoothing for all losses
        for k in loss_history.keys():
            if ema_losses[k] is None:
                ema_losses[k] = current_losses[k]
            else:
                ema_losses[k] = ema_decay * ema_losses[k] + (1 - ema_decay) * current_losses[k]
            loss_history[k].append(ema_losses[k])

        if (epoch + 1) % 40 == 0:
            loss_str = f"Epoch {epoch+1}/{num_epochs}"
            if lambda_dt_logp > 0:
                loss_str += f", (∂_t log p)²: {ema_losses['dt_logp']:.4f}"
            if lambda_grad_logp > 0:
                loss_str += f", ||∇log p||²: {ema_losses['grad_logp']:.4f}"
            if lambda_hessian_trace > 0:
                loss_str += f", tr(H)²: {ema_losses['hessian_trace']:.4f}"
            if lambda_hessian_frob > 0:
                loss_str += f", ||H||_F²: {ema_losses['hessian_frob']:.4f}"
            loss_str += f", ||F||²: {ema_losses['flow_norm']:.4f}"
            print(loss_str)

    print("Interpolant training complete!")
    return loss_history, flow_norms


def train_flow_matching_learned(flow_model: FlowMatchingModel, X_0: torch.Tensor, X_1: torch.Tensor,
                                 num_epochs: int = 500, lr: float = 1e-3, batch_size: int = 256):
    """Stage 2: Train velocity network with learned interpolant."""
    optimizer = optim.AdamW(flow_model.velocity_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    losses = []
    n_samples = len(X_0)

    # EMA for stable evaluation
    ema_decay = 0.999
    ema_velocity_net = VelocityNetwork(dim=flow_model.dim)
    ema_velocity_net.load_state_dict(flow_model.velocity_net.state_dict())

    print(f"\n--- Stage 2: Flow Matching Training (Learned Interpolant) ---")
    print(f"Epochs={num_epochs}, LR={lr}, Batch={batch_size}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = max(1, n_samples // batch_size)

        indices = torch.randperm(n_samples)

        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            x_0_batch = X_0[batch_idx]
            x_1_batch = X_1[batch_idx]

            t_scalar = np.random.uniform(0.001, 0.999)
            t = torch.full((len(batch_idx),), t_scalar)

            optimizer.zero_grad()

            with torch.no_grad():
                X_t, target_velocity = flow_model.interpolant.compute_velocity_batched(
                    x_0_batch, x_1_batch, t_scalar
                )

            pred_velocity = flow_model.velocity_net(X_t, t)
            loss = torch.mean((pred_velocity - target_velocity) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.velocity_net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_velocity_net.parameters(),
                                    flow_model.velocity_net.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()

        scheduler.step()
        losses.append(epoch_loss / n_batches)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    flow_model.velocity_net.load_state_dict(ema_velocity_net.state_dict())
    print("Flow matching training complete!")
    return losses


def train_flow_matching_gaussian(gaussian_model: GaussianFlowMatching, X_1: torch.Tensor,
                                  num_epochs: int = 500, lr: float = 1e-3, batch_size: int = 256):
    """Train standard Gaussian flow matching."""
    optimizer = optim.AdamW(gaussian_model.velocity_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    losses = []
    n_samples = len(X_1)

    # EMA for stable evaluation
    ema_decay = 0.999
    ema_velocity_net = VelocityNetwork(dim=gaussian_model.dim)
    ema_velocity_net.load_state_dict(gaussian_model.velocity_net.state_dict())

    print(f"\n--- Gaussian Flow Matching Training ---")
    print(f"Epochs={num_epochs}, LR={lr}, Batch={batch_size}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = max(1, n_samples // batch_size)

        indices = torch.randperm(n_samples)

        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            x_1_batch = X_1[batch_idx]
            x_0_batch = torch.randn_like(x_1_batch)

            t = torch.rand(len(batch_idx))
            t = torch.clamp(t, 0.001, 0.999)

            optimizer.zero_grad()

            t_expand = t[:, None]
            X_t = (1 - t_expand) * x_0_batch + t_expand * x_1_batch
            target_velocity = x_1_batch - x_0_batch

            pred_velocity = gaussian_model.velocity_net(X_t, t)
            loss = torch.mean((pred_velocity - target_velocity) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gaussian_model.velocity_net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_velocity_net.parameters(),
                                    gaussian_model.velocity_net.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()

        scheduler.step()
        losses.append(epoch_loss / n_batches)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]:.6f}")

    gaussian_model.velocity_net.load_state_dict(ema_velocity_net.state_dict())
    print("Gaussian flow matching training complete!")
    return losses
