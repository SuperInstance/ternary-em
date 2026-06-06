# ternary-em

**Expectation-Maximization with Ternary Latent Variables**

A Rust library for fitting mixture models where latent states take values in `{-1, 0, +1}` (ternary). Implements the full EM pipeline: E-step responsibility computation, M-step parameter updates, convergence detection via log-likelihood monitoring, and information-theoretic divergence measures.

## Features

- **Ternary Distributions**: `TernaryDistribution` with PMF, log-PMF, mean, variance, and sampling
- **Mixture Models**: `MixtureComponent` combining weights with ternary distributions
- **EM Algorithm**: `TernaryEM` with configurable convergence criteria
  - E-step: computes posterior responsibilities for each data point
  - M-step: updates component weights and distribution parameters
  - Convergence: monitors log-likelihood change with configurable tolerance
- **Data-Driven Initialization**: `init_from_data()` for heuristic component seeding
- **Divergence Measures**: KL divergence and Jensen-Shannon divergence between ternary distributions
- **Monotonicity Guarantee**: Log-likelihood is guaranteed to be non-decreasing (verified in tests)

## Quick Start

```rust
use ternary_em::{TernaryEM, MixtureComponent, TernaryDistribution, EMConfig};

// Prepare data: observations in {-1, 0, +1}
let data = vec![-1, -1, -1, 0, 0, 1, 1, 1, 1];

// Initialize a 2-component mixture
let components = vec![
    MixtureComponent {
        weight: 0.5,
        distribution: TernaryDistribution::new(0.7, 0.2, 0.1),
    },
    MixtureComponent {
        weight: 0.5,
        distribution: TernaryDistribution::new(0.1, 0.2, 0.7),
    },
];

// Run EM
let em = TernaryEM::with_config(components, EMConfig {
    max_iter: 500,
    tol: 1e-8,
    floor: 1e-300,
});
let result = em.fit(&data);

println!("Converged: {} in {} iterations", result.converged, result.iterations);
println!("Log-likelihood: {:.4}", result.log_likelihood);
```

## API Overview

### `TernaryDistribution`

| Method | Description |
|--------|-------------|
| `new(p_neg, p_zero, p_pos)` | Create with auto-normalization |
| `uniform()` | Equal probability 1/3 for each state |
| `pmf(x)` | Probability of value x |
| `log_pmf(x)` | Log-probability |
| `mean()` | Expected value E[X] |
| `variance()` | Var(X) |
| `sample(u)` | Deterministic sample from uniform u ∈ [0,1) |

### `TernaryEM`

| Method | Description |
|--------|-------------|
| `new(components)` | Initialize with given components |
| `with_config(components, config)` | Initialize with custom config |
| `init_from_data(k, data)` | Heuristic initialization from data |
| `e_step(data)` | Compute responsibilities |
| `m_step(data, responsibilities)` | Update parameters |
| `log_likelihood(data)` | Compute data log-likelihood |
| `fit(data)` | Run full EM to convergence |

### Divergence Measures

```rust
use ternary_em::{kl_divergence, js_divergence, TernaryDistribution};

let p = TernaryDistribution::new(0.5, 0.3, 0.2);
let q = TernaryDistribution::new(0.1, 0.3, 0.6);

let kl = kl_divergence(&p, &q);  // KL(P || Q) ≥ 0
let js = js_divergence(&p, &q);  // Symmetric, bounded [0, ln2]
```

## Mathematical Background

The EM algorithm alternates between:

1. **E-step**: Compute responsibilities γ(zₙₖ) = P(zₙ = k | xₙ, θ)
2. **M-step**: Update parameters maximizing expected complete-data log-likelihood

For ternary distributions, each component k has parameters (pₖ₋₁, pₖ₀, pₖ₊₁) and mixing weight πₖ. The M-step updates are:

- πₖ = Nₖ / N where Nₖ = Σₙ γ(zₙₖ)
- pₖₓ = Σₙ γ(zₙₖ) · 𝟙(xₙ = x) / Nₖ

## Testing

```bash
cargo test
```

16 comprehensive tests covering:
- Distribution properties (normalization, PMF, mean, variance)
- E-step validity (probabilities sum to 1, bounded in [0,1])
- M-step improvement (likelihood non-decreasing)
- Convergence on synthetic 2-component mixtures
- Known parameter recovery
- Monotonic log-likelihood
- KL and JS divergence properties
- Data-driven initialization

## License

MIT
