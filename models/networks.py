"""Neural network models for flow matching."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class ConditionalNormalizingFlow(nn.Module):
    """Conditional normalizing flow F^φ_t(X_0, X_1)"""
    def __init__(self, dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.condition_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2)
        )

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_0.shape[0]
        t_expanded = torch.full((batch_size, 1), t, device=x_0.device)
        condition_input = torch.cat([x_1, t_expanded], dim=1)
        params = self.condition_net(condition_input)
        scale = torch.exp(torch.clamp(params[:, :self.dim], -5, 5))
        shift = params[:, self.dim:]
        y = scale * x_0 + shift
        log_det_jacobian = torch.sum(torch.log(scale), dim=1)
        return y, log_det_jacobian

    def inverse(self, y: torch.Tensor, x_1: torch.Tensor, t: float) -> torch.Tensor:
        batch_size = y.shape[0]
        t_expanded = torch.full((batch_size, 1), t, device=y.device)
        condition_input = torch.cat([x_1, t_expanded], dim=1)
        params = self.condition_net(condition_input)
        scale = torch.exp(torch.clamp(params[:, :self.dim], -5, 5))
        shift = params[:, self.dim:]
        x_0 = (y - shift) / scale
        return x_0

    def forward_unconditional(self, x_0: torch.Tensor) -> torch.Tensor:
        """F_0^φ(X_0) - no conditioning on x_1, use zeros"""
        batch_size = x_0.shape[0]
        dummy_x1 = torch.zeros(batch_size, self.dim, device=x_0.device)
        y, _ = self.forward(x_0, dummy_x1, t=0.0)
        return y


class FlowInterpolant:
    """Implements X_t^φ = tX_1 + (1-t)F^φ_t(X_0, X_1)"""
    def __init__(self, dim: int = 2):
        self.dim = dim
        self.flow = ConditionalNormalizingFlow(dim=dim)

    def sample_X_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: float) -> torch.Tensor:
        F_t, _ = self.flow(x_0, x_1, t)
        X_t = t * x_1 + (1 - t) * F_t
        return X_t

    def compute_velocity_batched(self, x_0: torch.Tensor, x_1: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute X_t and ∂_t X_t^φ in batch with same t for all samples using proper JVP"""
        batch_size = x_0.shape[0]

        # Use torch.enable_grad() in case we're called from a no_grad context
        with torch.enable_grad():
            # Create t as a tensor that requires grad
            t_tensor = torch.tensor(t, requires_grad=True)
            t_expanded = t_tensor.expand(batch_size, 1)

            # Compute F_t with gradient tracking for t
            condition_input = torch.cat([x_1, t_expanded], dim=1)
            params = self.flow.condition_net(condition_input)
            scale = torch.exp(torch.clamp(params[:, :self.dim], -5, 5))
            shift = params[:, self.dim:]
            F_t = scale * x_0 + shift

            # Compute X_t
            X_t = t_tensor * x_1 + (1 - t_tensor) * F_t

            # Compute dF_dt using JVP (derivative of F_t w.r.t. t)
            # We need grad of each component of F_t w.r.t. t
            dF_dt = torch.zeros_like(F_t)
            for i in range(self.dim):
                grad_i = torch.autograd.grad(F_t[:, i].sum(), t_tensor, create_graph=False, retain_graph=True)[0]
                dF_dt[:, i] = grad_i

            # Velocity: ∂_t X_t = x_1 - F_t + (1-t) * dF_dt
            velocity = x_1 - F_t + (1 - t) * dF_dt

        return X_t.detach(), velocity.detach()

    def log_prob_conditional(self, x: torch.Tensor, x_1: torch.Tensor, t: float) -> torch.Tensor:
        if t < 1e-6:
            x_0 = self.flow.inverse(x, x_1, t)
            log_p0 = torch.distributions.Normal(0, 1).log_prob(x_0).sum(dim=1)
            _, log_det_flow = self.flow(x_0, x_1, t)
            return log_p0 - log_det_flow

        if t > 1 - 1e-6:
            dist = torch.norm(x - x_1, dim=1)
            return -1e10 * (dist > 0.01).float()

        y = (x - t * x_1) / (1 - t + 1e-8)
        x_0 = self.flow.inverse(y, x_1, t)
        log_p0 = torch.distributions.Normal(0, 1).log_prob(x_0).sum(dim=1)
        _, log_det_flow = self.flow(x_0, x_1, t)
        log_det_scaling = self.dim * np.log(abs(1 - t) + 1e-8)
        return log_p0 - log_det_flow - log_det_scaling

    def sample_initial(self, n_samples: int) -> torch.Tensor:
        """Sample from F_0^φ(X_0) for ODE initialization"""
        x_0 = torch.randn(n_samples, self.dim)
        return self.flow.forward_unconditional(x_0)

    def compute_dt_log_prob(self, x_0: torch.Tensor, x_1: torch.Tensor, t: float) -> torch.Tensor:
        """Compute ∂_t log p_t(X_t) using automatic differentiation.

        For interpolant X_t = t*x_1 + (1-t)*F_t(x_0, x_1), the log prob is:
        log p_t(X_t) = log p_0(x_0) - log|det(∂X_t/∂x_0)|

        where ∂X_t/∂x_0 = (1-t) * scale (for affine flow F_t = scale * x_0 + shift)

        Returns per-sample ∂_t log p_t values.
        """
        batch_size = x_0.shape[0]

        with torch.enable_grad():
            # Create t as a tensor that requires grad
            t_tensor = torch.tensor(t, requires_grad=True, dtype=torch.float32)

            # Compute flow parameters with t-dependent conditioning
            t_expanded = t_tensor.expand(batch_size, 1)
            condition_input = torch.cat([x_1, t_expanded], dim=1)
            params = self.flow.condition_net(condition_input)

            log_scale = torch.clamp(params[:, :self.dim], -5, 5)

            # Log det Jacobian: sum of log((1-t) * scale) = dim*log(1-t) + sum(log_scale)
            log_one_minus_t = torch.log(torch.abs(1 - t_tensor) + 1e-8)
            log_det_jacobian = self.dim * log_one_minus_t + log_scale.sum(dim=1)

            # log p_t = log p_0(x_0) - log_det_jacobian
            # Since x_0 is fixed, ∂_t log p_t = -∂_t log_det_jacobian

            # Compute ∂_t log_det_jacobian for each sample
            dt_log_det = torch.zeros(batch_size, device=x_0.device)
            for i in range(batch_size):
                grad_i = torch.autograd.grad(
                    log_det_jacobian[i], t_tensor,
                    create_graph=True, retain_graph=True
                )[0]
                dt_log_det[i] = grad_i

            # ∂_t log p_t = -∂_t log_det_jacobian
            dt_log_prob = -dt_log_det

        return dt_log_prob

    def compute_grad_log_prob(self, x_0: torch.Tensor, x_1: torch.Tensor, t: float) -> torch.Tensor:
        """Compute ∇_{X_t} log p_t(X_t) using the change of variables formula.

        For interpolant X_t = t*x_1 + (1-t)*F_t(x_0, x_1), we have:
        log p_t(X_t) = log p_0(x_0) - log|det(∂X_t/∂x_0)|

        The gradient w.r.t. X_t requires inverting the mapping.
        Here we compute it via automatic differentiation.

        Returns:
            Gradient norm squared ||∇ log p_t||² for each sample, shape (batch_size,)
        """
        batch_size = x_0.shape[0]

        with torch.enable_grad():
            # Sample X_t and track gradients
            x_0_grad = x_0.clone().detach().requires_grad_(True)
            F_t, log_det = self.flow(x_0_grad, x_1, t)
            X_t = t * x_1 + (1 - t) * F_t

            # log p_t = log p_0(x_0) - log_det
            log_p0 = torch.distributions.Normal(0, 1).log_prob(x_0_grad).sum(dim=1)
            log_pt = log_p0 - log_det

            # Compute gradient of log_pt w.r.t. X_t via chain rule through x_0
            # ∇_{X_t} log p_t = (∂x_0/∂X_t)^T ∇_{x_0} log p_t
            # For affine flow: ∂X_t/∂x_0 = (1-t)*scale, so ∂x_0/∂X_t = 1/((1-t)*scale)

            # Compute ∇_{x_0} log p_t
            grad_x0 = torch.autograd.grad(log_pt.sum(), x_0_grad, create_graph=True)[0]

            # Get scale from flow
            t_expanded = torch.full((batch_size, 1), t)
            condition_input = torch.cat([x_1, t_expanded], dim=1)
            params = self.flow.condition_net(condition_input)
            log_scale = torch.clamp(params[:, :self.dim], -5, 5)
            scale = torch.exp(log_scale)

            # ∂x_0/∂X_t = 1 / ((1-t) * scale)
            jacobian_inv = 1.0 / ((1 - t + 1e-8) * scale)

            # ∇_{X_t} log p_t = jacobian_inv * grad_x0 (element-wise for diagonal Jacobian)
            grad_Xt = jacobian_inv * grad_x0

            # Return squared norm
            grad_norm_sq = (grad_Xt ** 2).sum(dim=1)

        return grad_norm_sq

    def compute_log_prob_for_hessian(self, x_0: torch.Tensor, x_1: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log p_t and X_t with gradients enabled for Hessian computation.

        We compute log_pt as a function of X_t by inverting the transformation.
        For X_t = t*x_1 + (1-t)*F_t where F_t = scale*x_0 + shift:
        - Solve for x_0: x_0 = (X_t - t*x_1 - (1-t)*shift) / ((1-t)*scale)
        - Then: log p_t = log p_0(x_0) - log|det(∂X_t/∂x_0)|

        Returns:
            X_t: Interpolated samples with grad enabled, shape (batch_size, dim)
            log_pt: Log probabilities as function of X_t, shape (batch_size,)
        """
        batch_size = x_0.shape[0]

        with torch.enable_grad():
            # First compute X_t without gradients
            with torch.no_grad():
                F_t, _ = self.flow(x_0, x_1, t)
                X_t_values = t * x_1 + (1 - t) * F_t

            # Create X_t as a variable
            X_t = X_t_values.clone().detach().requires_grad_(True)

            # Get flow parameters (scale and shift)
            t_expanded = torch.full((batch_size, 1), t)
            condition_input = torch.cat([x_1, t_expanded], dim=1)
            params = self.flow.condition_net(condition_input)
            log_scale = torch.clamp(params[:, :self.dim], -5, 5)
            scale = torch.exp(log_scale)
            shift = params[:, self.dim:]

            # Invert the transformation: X_t = t*x_1 + (1-t)*(scale*x_0 + shift)
            # => X_t = t*x_1 + (1-t)*scale*x_0 + (1-t)*shift
            # => x_0 = (X_t - t*x_1 - (1-t)*shift) / ((1-t)*scale)
            x_0_reconstructed = (X_t - t * x_1 - (1 - t) * shift) / ((1 - t + 1e-8) * scale)

            # Compute log p_t = log p_0(x_0) - log|det(∂X_t/∂x_0)|
            log_p0 = torch.distributions.Normal(0, 1).log_prob(x_0_reconstructed).sum(dim=1)

            # Log determinant of Jacobian: ∂X_t/∂x_0 = (1-t)*scale (diagonal)
            log_det_jacobian = self.dim * torch.log(torch.abs(torch.tensor(1 - t + 1e-8))) + log_scale.sum(dim=1)

            log_pt = log_p0 - log_det_jacobian

        return X_t, log_pt


class VelocityNetwork(nn.Module):
    """State-of-the-art velocity network u_t^θ(x) with time conditioning"""
    def __init__(self, dim: int = 2, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.dim = dim

        # Time embedding with sinusoidal encoding
        self.time_embed_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main network with skip connections
        self.input_layer = nn.Linear(dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
                'time_proj': nn.Linear(hidden_dim, hidden_dim)
            }))

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

        # Zero initialization for output
        nn.init.zeros_(self.output_layer[-1].weight)
        nn.init.zeros_(self.output_layer[-1].bias)

    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal_embedding(t)
        t_emb = self.time_mlp(t_emb)

        h = self.input_layer(x)

        for layer in self.layers:
            h_norm = layer['norm'](h)
            h_act = torch.nn.functional.silu(h_norm)
            h_linear = layer['linear'](h_act)
            h_time = layer['time_proj'](t_emb)
            h = h + h_linear + h_time

        return self.output_layer(h)


class FlowMatchingModel:
    """Flow matching model using learned interpolant"""
    def __init__(self, dim: int = 2, interpolant: FlowInterpolant = None):
        self.dim = dim
        self.interpolant = interpolant
        self.velocity_net = VelocityNetwork(dim=dim)

    @torch.no_grad()
    def sample(self, n_samples: int, num_steps: int = 100) -> torch.Tensor:
        """Sample by solving ODE from F_0^φ(X_0)"""
        x = self.interpolant.sample_initial(n_samples)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step * dt
            t_tensor = torch.full((n_samples,), t)

            # RK4 integration
            k1 = self.velocity_net(x, t_tensor)
            k2 = self.velocity_net(x + 0.5 * dt * k1, t_tensor + 0.5 * dt)
            k3 = self.velocity_net(x + 0.5 * dt * k2, t_tensor + 0.5 * dt)
            k4 = self.velocity_net(x + dt * k3, t_tensor + dt)

            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x


class GaussianFlowMatching:
    """Standard Gaussian flow matching for comparison"""
    def __init__(self, dim: int = 2):
        self.dim = dim
        self.velocity_net = VelocityNetwork(dim=dim)

    @torch.no_grad()
    def sample(self, n_samples: int, num_steps: int = 100) -> torch.Tensor:
        x = torch.randn(n_samples, self.dim)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step * dt
            t_tensor = torch.full((n_samples,), t)

            # RK4 integration
            k1 = self.velocity_net(x, t_tensor)
            k2 = self.velocity_net(x + 0.5 * dt * k1, t_tensor + 0.5 * dt)
            k3 = self.velocity_net(x + 0.5 * dt * k2, t_tensor + 0.5 * dt)
            k4 = self.velocity_net(x + dt * k3, t_tensor + dt)

            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x
