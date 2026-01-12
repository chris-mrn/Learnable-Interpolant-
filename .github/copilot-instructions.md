# Flow Matching with Learned Stochastic Interpolation

## Project Overview
This implements a **two-stage flow matching approach** for generative modeling:
- **Stage 1**: Train conditional normalizing flow `F^φ_t(X_0, X_1)` to create interpolant `X_t^φ = tX_1 + (1-t)F^φ_t(X_0, X_1)` minimizing `E[||∇²log p||²]`
- **Stage 2**: Train velocity network `u_t^θ` to match `∂_t X_t^φ` for ODE-based sampling

The key innovation: learned interpolant starting from `F_0^φ(X_0)` vs standard Gaussian noise, compared against baseline Gaussian Flow Matching.

## Architecture Components

### Core Models (`models/networks.py`)
- **`ConditionalNormalizingFlow`**: Affine coupling layer `F^φ_t(X_0, X_1)` with time/data conditioning; includes `inverse()` and `forward_unconditional()` for initialization
- **`FlowInterpolant`**: Manages interpolant `X_t^φ` and computes velocity `∂_t X_t^φ` via finite differences (`compute_velocity_batched()`)
- **`VelocityNetwork`**: State-of-the-art architecture with sinusoidal time embeddings, LayerNorm, skip connections, and zero-initialized output layer
- **`FlowMatchingModel`**: Wraps learned interpolant + velocity; samples via RK4 ODE solver starting from `F_0^φ(X_0)`
- **`GaussianFlowMatching`**: Baseline with standard Gaussian initialization for comparison

### Training Pipeline (`training.py`)
**Stage 1 (`train_interpolant`)**: 
- Loss: `E[||∇²log p_t||²]` (Hessian Frobenius norm via `utils.compute_hessian_frobenius_norm`) + `λ||F||²` flow regularization + L2 weight decay
- Sample time `t ~ U(0.2, 0.8)` to avoid boundary instabilities
- Gradient clipping (norm=1.0) essential for stability

**Stage 2 (`train_flow_matching_learned` / `train_flow_matching_gaussian`)**:
- Loss: `E[||u_t^θ(X_t) - ∂_t X_t^φ||²]` (learned) or `E[||u_t^θ(X_t) - (X_1 - X_0)||²]` (Gaussian)
- EMA (decay=0.999) for stable evaluation
- Cosine annealing scheduler, AdamW with weight decay

### Data (`data/datasets.py`)
- **`create_mixture_of_gaussians`**: 2D mixture of 3 Gaussians (70%/20%/10% split) with specified means/covariances
- **`get_data`**: Returns `(X_0, X_1, X_1_samples)` where `X_0` is Gaussian noise, `X_1` is target distribution

## Critical Implementation Details

### Numerical Stability
- **Time bounds**: `t ∈ [0.001, 0.999]` throughout; use `t ∈ [0.2, 0.8]` for Stage 1 training
- **Log-prob computation**: Special handling at `t→0` (use flow inverse) and `t→1` (penalize deviation from `X_1`)
- **Hessian computation**: Requires `X_t.requires_grad_(True)` even when sampled with `torch.no_grad()`
- **Scale clamping**: `torch.clamp(scale_logits, -5, 5)` in `ConditionalNormalizingFlow` prevents NaN

### Velocity Computation Pattern
```python
# ALWAYS compute velocity with same t for entire batch
X_t, target_velocity = interpolant.compute_velocity_batched(x_0_batch, x_1_batch, t_scalar)
pred_velocity = velocity_net(X_t, t_tensor)  # t_tensor: (batch_size,) all same value
```

### Sampling Pattern
```python
# Initialize from LEARNED flow, not raw noise
x = flow_model.interpolant.sample_initial(n_samples)  # Uses F_0^φ(X_0)
# NOT: x = torch.randn(n_samples, dim)  # Only for GaussianFlowMatching
```

## Development Workflows

### Run Full Pipeline
```bash
python main.py  # Trains both stages, evaluates, generates all visualizations
```

### Dependencies
```bash
pip install numpy scipy matplotlib torch  # Requirements in requirements.txt
```

### Output Structure
- **Figures**: `outputs/figures/` (auto-created)
  - `flow_matching_comparison.png`: Sample quality comparison
  - `training_curves.png`: Loss trajectories
  - `trajectories_*.png`: ODE flow visualization
  - `wasserstein_evolution.png`: W2 distance over time

## Key Conventions

### Hyperparameters (tune these together)
- **Interpolant**: `num_epochs=100`, `lr=3e-4`, `lambda_flow_reg=0.05` balances Hessian vs flow smoothness
- **Flow Matching**: `num_epochs=1000`, `lr=2e-3`, `batch_size=256`; reduce LR if loss diverges
- **ODE steps**: `num_steps=100` for sampling; increase if trajectories diverge

### Code Organization
- Models define network architecture + sampling logic in single class
- Training functions are pure: take model + data, return losses
- Evaluation (`evaluation.py`) handles all visualization + metrics computation
- `utils.py` provides differentiable ops (`compute_hessian_frobenius_norm`) and metrics (`compute_wasserstein2`)

### Naming Conventions
- `X_t`: Interpolant state at time `t`
- `F_t` / `F^φ_t`: Flow output (not interpolant, component of interpolant)
- `u_t^θ`: Velocity network (subscript `t` for time-dependent)
- Variables with `_torch` suffix are PyTorch tensors; without are NumPy arrays

### When Adding Features
- **New distributions**: Modify `create_mixture_of_gaussians` in `data/datasets.py`
- **Different architectures**: Inherit from base classes, ensure `forward()` and `sample()` methods
- **New losses**: Add to training functions; use `with torch.no_grad()` when sampling within loss computation
- **Visualizations**: Add to `evaluation.py`; always save to `FIGURES_DIR` with descriptive names

## Common Pitfalls
- Don't forget `.detach()` when using sampled `X_t` for gradient computation (see `X_t = X_t_samples.clone().detach().requires_grad_(True)`)
- Velocity network zero-init is crucial; random init causes divergence
- EMA model used for final evaluation, not training model
- RK4 integration order matters: don't simplify to Euler without quality degradation
