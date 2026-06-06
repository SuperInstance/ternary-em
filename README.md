# ternary-em

Expectation-Maximization for ternary latent variables.

When your data lives in {−1, 0, +1} and you suspect there are hidden clusters driving the pattern, you need EM that speaks ternary natively. Not a hack on top of Gaussian mixtures, not a binarization trick — a mixture model where every component is a ternary distribution, every latent state is a trit, and the math works out cleanly.

This crate implements that from scratch: ternary distributions, mixture models, full E-step/M-step cycles, convergence detection, and the information-theoretic tools (KL divergence, Jensen-Shannon) to compare what you've learned.

## Why This Exists

Ternary neural networks quantize weights to {−1, 0, +1}. When you're analyzing the statistical structure of these networks — clustering weight patterns, identifying mode collapse, fitting generative models — you need distributions that match the data's native alphabet.

Using continuous distributions and rounding is wrong because:
1. The probability mass function is discrete with exactly 3 outcomes
2. The sufficient statistics are counts, not moments
3. KL divergence has a clean closed form for ternary distributions
4. EM convergence is faster when the model matches the data topology

The key insight: a ternary distribution is fully specified by three numbers (p₋₁, p₀, p₁) that sum to 1. This makes EM iterations cheap — no matrix inversions, no Cholesky decompositions, just weighted counting.

## Quick Start

```rust
use ternary_em::{TernaryEM, MixtureComponent, TernaryDistribution, EMConfig};

// Your ternary data
let data = vec![-1i8, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1];

// Initialize a 2-component mixture
let components = vec![
    MixtureComponent {
        weight: 0.5,
        distribution: TernaryDistribution::new(0.7, 0.2, 0.1), // biased toward -1
    },
    MixtureComponent {
        weight: 0.5,
        distribution: TernaryDistribution::new(0.1, 0.2, 0.7), // biased toward +1
    },
];

// Run EM
let em = TernaryEM::with_config(components, EMConfig {
    max_iter: 500,
    tol: 1e-10,
    floor: 1e-300,
});
let result = em.fit(&data);

println!("Converged: {} in {} iterations", result.converged, result.iterations);
println!("Log-likelihood: {:.4}", result.log_likelihood);

// Inspect learned components
for (i, comp) in result.components.iter().enumerate() {
    println!("Component {}: weight={:.3}, dist=({:.3}, {:.3}, {:.3})",
        i, comp.weight,
        comp.distribution.p_neg, comp.distribution.p_zero, comp.distribution.p_pos);
}
```

## Architecture

```
┌──────────────────────────────────────────┐
│              TernaryEM                   │
│                                          │
│  fit(data) ──→ E-step ──→ M-step ──→ LL │
│       ↑          │          │        │   │
│       │          │          │        │   │
│       └──── iterate until converged ─┘   │
│                                          │
│  Config: max_iter, tol, floor            │
├──────────────────────────────────────────┤
│          TernaryDistribution             │
│  pmf(x), log_pmf(x), mean(), variance() │
│  sample(u), uniform()                    │
├──────────────────────────────────────────┤
│        MixtureComponent                  │
│  weight + TernaryDistribution            │
├──────────────────────────────────────────┤
│        Divergence Measures               │
│  kl_divergence(p, q)                     │
│  js_divergence(p, q)                     │
└──────────────────────────────────────────┘
```

### TernaryDistribution

The building block. Three probabilities summing to 1:

```rust
let d = TernaryDistribution::new(0.5, 0.3, 0.2);

d.pmf(-1);      // 0.5
d.pmf(0);       // 0.3
d.pmf(1);       // 0.2
d.mean();       // 0.2 - 0.5 = -0.3
d.variance();   // E[X²] - E[X]² = 0.7 - 0.09 = 0.61
d.log_pmf(1);   // ln(0.2)
```

Auto-normalization: `TernaryDistribution::new(1.0, 2.0, 3.0)` produces (1/6, 2/6, 3/6). Pass unnormalized counts directly from data.

Sampling is deterministic from a uniform random `u ∈ [0, 1)`:

```rust
let d = TernaryDistribution::new(0.2, 0.5, 0.3);
d.sample(0.1);  // -1 (first 20%)
d.sample(0.5);  //  0 (next 50%)
d.sample(0.9);  // +1 (last 30%)
```

### The EM Loop

```
1. E-step: For each data point, compute posterior probability of each component
   γ(zₙₖ) = πₖ · pₖ(xₙ) / Σⱼ πⱼ · pⱼ(xₙ)

2. M-step: Update parameters from weighted counts
   πₖ = Nₖ / N
   pₖ(x) = Σₙ γ(zₙₖ) · 𝟙(xₙ = x) / Nₖ

3. Check convergence: |LL_new - LL_old| < tol
```

For ternary distributions, the M-step is just weighted counting. No gradient computation, no learning rate, no optimization landscape. The E-step divides by a sum. That's the entire algorithm.

### EMConfig

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `max_iter` | 500 | Hard stop for iterations |
| `tol` | 1e-8 | Log-likelihood change threshold |
| `floor` | 1e-300 | Minimum probability to prevent `log(0)` |

The `floor` parameter is critical. Without it, a component that assigns zero probability to an observed value produces `-inf` log-likelihood, which propagates through the E-step and destroys convergence. The floor keeps everything finite.

### Data-Driven Initialization

Manual initialization works when you know your data, but `init_from_data()` provides a heuristic:

```rust
// Split data into K chunks, fit distribution to each
let em = TernaryEM::init_from_data(3, &data);
let result = em.fit(&data);
```

Each chunk gets a distribution estimated from its local statistics, plus uniform weight. Not optimal, but good enough for convergence in most cases.

## API Reference

### `TernaryDistribution`

| Method | Description |
|--------|-------------|
| `new(p_neg, p_zero, p_pos)` | Create with auto-normalization |
| `uniform()` | Equal 1/3 probabilities |
| `pmf(x)` | Probability of value x |
| `log_pmf(x)` | Log-probability (−∞ for impossible values) |
| `mean()` | Expected value: p_pos − p_neg |
| `variance()` | Var(X) |
| `sample(u)` | Deterministic sample from uniform u |

### `TernaryEM`

| Method | Description |
|--------|-------------|
| `new(components)` | EM with default config |
| `with_config(components, config)` | Custom config |
| `init_from_data(k, data)` | Heuristic K-component initialization |
| `e_step(data)` | Compute responsibility matrix |
| `m_step(data, responsibilities)` | Update parameters in place |
| `log_likelihood(data)` | Current LL |
| `fit(data)` | Run to convergence, return `EMResult` |

### `EMResult`

| Field | Description |
|-------|-------------|
| `components` | Fitted mixture components |
| `log_likelihood` | LL at convergence |
| `iterations` | Iterations used |
| `converged` | Whether it converged within max_iter |
| `ll_history` | LL per iteration (for plotting) |

### Divergence Functions

```rust
use ternary_em::{kl_divergence, js_divergence, TernaryDistribution};

let p = TernaryDistribution::new(0.5, 0.3, 0.2);
let q = TernaryDistribution::new(0.1, 0.3, 0.6);

kl_divergence(&p, &q);  // KL(P‖Q) ≥ 0, zero iff P=Q
js_divergence(&p, &q);  // Symmetric, bounded [0, ln 2]
```

KL divergence is not symmetric — `KL(P‖Q) ≠ KL(Q‖P)` in general. Jensen-Shannon fixes this by averaging against the midpoint distribution `M = (P+Q)/2`.

## Real-World Example: Weight Distribution Clustering

You've quantized a neural network layer to ternary weights and want to understand the distribution structure:

```rust
use ternary_em::{TernaryEM, TernaryDistribution, MixtureComponent, EMConfig};

// Ternary weights from a convolutional layer (flattened)
let weights: Vec<i8> = load_layer_weights("conv2"); // values in {-1, 0, 1}

// Hypothesis: weights cluster into "mostly negative", "sparse", "mostly positive"
let em = TernaryEM::init_from_data(3, &weights);
let result = em.fit(&weights);

// Identify what each component learned
for (i, comp) in result.components.iter().enumerate() {
    let d = &comp.distribution;
    let dominant = if d.p_neg > d.p_zero && d.p_neg > d.p_pos { "negative" }
               else if d.p_pos > d.p_zero && d.p_pos > d.p_neg { "positive" }
               else { "sparse" };
    
    println!("Cluster {} ({:.1}% of weights): {} dominant",
        i, comp.weight * 100.0, dominant);
}

// Compare two layers
let layer1_dist = TernaryDistribution::new(
    count_neg(&weights) as f64,
    count_zero(&weights) as f64,
    count_pos(&weights) as f64,
);
let layer2_dist = /* ... */;
println!("JS divergence between layers: {:.4}", js_divergence(&layer1_dist, &layer2_dist));
```

## Ecosystem Connections

- **`ternary-logistic`** — Classification on ternary features; EM can discover cluster structure in the feature space
- **`ternary-regression`** — Residual analysis may reveal mixture structure that EM can model
- **`ternary-fence`** — Coordinate EM iterations across distributed workers

## Performance Notes

- **Per-iteration cost**: O(N × K) where N = data points, K = components. Just multiply-accumulate on ternary PMFs.
- **Memory**: O(N × K) for the responsibility matrix. For large N, process in minibatches.
- **Convergence**: Typically 10-50 iterations for well-separated clusters. The ll_history field lets you verify monotonic increase.
- **Numerical stability**: The `floor` parameter prevents log(0). The E-step falls back to uniform responsibilities when the denominator is zero.

## Mathematical Guarantee

The log-likelihood is guaranteed to be non-decreasing across EM iterations (verified by the test suite). This is a fundamental property of EM: each M-step maximizes a lower bound on the likelihood, so the true likelihood can only stay the same or improve.

If you observe decreasing log-likelihood, check your floor parameter — it may be too large, distorting the E-step normalization.

## Open Questions

- **Online EM**: Currently batch-only. An incremental variant that updates sufficient statistics per data point would enable streaming analysis.
- **Component selection**: No BIC/AIC for choosing K. You need to run EM with multiple K values and compare.
- **Non-trivial initialization**: `init_from_data()` uses simple chunking. K-means++ style initialization adapted for ternary distances would improve convergence.

## License

MIT
