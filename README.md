# Flow Matching with Learned Stochastic Interpolation

A PyTorch implementation of a two-stage flow matching approach for generative modeling, featuring a learned interpolant that minimizes the Hessian norm of log probabilities.

## Overview

This project implements an innovative approach to generative modeling that learns an optimal interpolation path between noise and data distributions. Unlike standard flow matching methods that use Gaussian noise, we learn a conditional normalizing flow to create a smoother, more stable interpolant.

### Key Innovation

**Two-Stage Training Process:**

1. **Stage 1 - Learn the Interpolant**: Train a conditional normalizing flow `F^φ_t(X_0, X_1)` to create an optimal interpolant:
   ```
   X_t^φ = tX_1 + (1-t)F^φ_t(X_0, X_1)
   ```
   The flow is optimized to minimize various smoothness criteria including gradient norms and Hessian norms of the log probability.

2. **Stage 2 - Flow Matching**: Train a velocity network `u_t^θ` to match the time derivative of the learned interpolant, enabling efficient ODE-based sampling.

**Comparison with Baseline**: The project includes a Gaussian Flow Matching baseline for direct comparison, demonstrating the advantages of the learned interpolant approach.

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Learn_interpolant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Train Both Stages and Evaluate

Run the complete pipeline:

```bash
python main.py
```

This will:
- Train the interpolant (Stage 1)
- Train both learned and Gaussian flow matching models (Stage 2)
- Generate comparison visualizations
- Compute Wasserstein-2 distances

### Train Interpolant Only

For focused experimentation with the interpolant:

```bash
python train_interpolant_only.py
```

This script allows you to configure different loss terms and visualize the learned destruction process.

## Project Structure

```
Learn_interpolant/
├── models/
│   ├── __init__.py
│   └── networks.py           # Core model architectures
├── data/
│   ├── __init__.py
│   └── datasets.py           # Data generation (mixture of Gaussians)
├── outputs/
│   └── figures/              # Generated visualizations
├── main.py                   # Full pipeline
├── train_interpolant_only.py # Standalone interpolant training
├── training.py               # Training loops
├── evaluation.py             # Metrics and visualization
├── utils.py                  # Helper functions
└── requirements.txt
```

## Key Components

### Models (`models/networks.py`)

- **`ConditionalNormalizingFlow`**: Affine coupling layer that learns the transformation `F^φ_t(X_0, X_1)` with time and data conditioning
- **`FlowInterpolant`**: Manages the interpolant `X_t^φ` and computes velocities
- **`VelocityNetwork`**: Neural network with sinusoidal time embeddings for flow matching
- **`FlowMatchingModel`**: Complete model combining learned interpolant and velocity network
- **`GaussianFlowMatching`**: Baseline model using standard Gaussian initialization

### Loss Functions

The interpolant training supports multiple configurable loss terms:

1. **Time Derivative Loss** (`lambda_dt_logp`): `E[(∂_t log p_t)²]`
   - Penalizes rapid changes in log probability over time

2. **Gradient Norm Loss** (`lambda_grad_logp`): `E[||∇ log p_t||²]`
   - Encourages smooth gradients in the probability landscape

3. **Hessian Trace Loss** (`lambda_hessian_trace`): `E[(tr(∇² log p_t))²]`
   - Minimizes the trace of the Hessian (uses Hutchinson estimator)

4. **Hessian Frobenius Norm Loss** (`lambda_hessian_frob`): `E[||∇² log p_t||_F²]`
   - The primary smoothness criterion; minimizes curvature

5. **Flow Regularization** (`lambda_flow_reg`): `E[||F||²]`
   - Prevents the flow output from growing too large

### Time-Dependent Weighting

The training uses per-sample time-dependent weighting:
```python
wt = 1 / (1 - t)²
```
This emphasizes learning near `t→1`, where the interpolant approaches the data distribution.

## Configuration

### Interpolant Training (`train_interpolant_only.py`)

Key hyperparameters you can adjust:

```python
loss_config = {
    'lambda_dt_logp': 0.0,        # Time derivative loss weight
    'lambda_grad_logp': 1.0,      # Gradient norm loss weight
    'lambda_hessian_trace': 1.0,  # Hessian trace loss weight
    'lambda_hessian_frob': 0.0,   # Hessian Frobenius loss weight
    'lambda_flow_reg': 1.0,       # Flow regularization weight
    'num_epochs': 200,
    'lr': 3e-4,
}
```

### Flow Matching Training

Adjust in `main.py`:

```python
# For learned interpolant flow matching
train_flow_matching_learned(
    model=flow_model,
    X_0=X_0,
    X_1=X_1,
    num_epochs=1000,
    lr=2e-3,
    batch_size=256,
)
```

## Sampling

The learned model samples via ODE integration:

```python
# Initialize from learned flow (NOT Gaussian noise!)
x = flow_model.interpolant.sample_initial(n_samples)

# Integrate using RK4
samples = flow_model.sample(n_samples, num_steps=100)
```

**Important**: The learned interpolant starts from `F_0^φ(X_0)`, not standard Gaussian noise.

## Evaluation Metrics

- **Wasserstein-2 Distance**: Measures distributional similarity between generated and true samples
- **Visual Inspection**: Sample quality, trajectory smoothness, training curves
- **Component Analysis**: Individual loss terms tracked throughout training

## Advanced Features

### Per-Sample Time Sampling

Recent implementation uses different time values for each batch element:
```python
t_batch = torch.rand(batch_size) * 0.88 + 0.01  # t ∈ [0.01, 0.99]
```
This provides better coverage of the time domain and more diverse training.

### Hutchinson Trace Estimator

For efficient Hessian trace computation:
```python
trace_est = E[v^T ∇² log p_t v]  # v ~ N(0, I)
```

### Numerical Stability

- Time bounds: `t ∈ [0.001, 0.999]` to avoid singularities
- Gradient clipping: norm=1.0
- Scale clamping in flow network
- EMA for stable evaluation

## Tips and Best Practices

1. **Loss Balancing**: Start with `lambda_grad_logp=1.0`, `lambda_hessian_trace=1.0`, `lambda_flow_reg=1.0`
2. **Training Stability**: If losses diverge, reduce learning rate or increase gradient clipping
3. **ODE Steps**: Use at least 100 steps for high-quality sampling
4. **Visualization**: Check `outputs/figures/` to monitor training progress
5. **Time Weighting**: The `1/(1-t)²` weighting is crucial for learning near the data distribution

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{learned_interpolant_flow_matching,
  title={Flow Matching with Learned Stochastic Interpolation},
  year={2026},
  url={https://github.com/yourusername/Learn_interpolant}
}
```

## License

[Add your license here]

## Acknowledgments

This implementation builds on the flow matching framework and incorporates ideas from normalizing flows and optimal transport theory.
