# ternary-em

Expectation-Maximization for ternary latent variables — mixture models where every component is a distribution over {-1, 0, +1}.

## The Problem

You have data that lives in {-1, 0, +1} and you suspect there are hidden clusters driving the pattern. Maybe you're analyzing weight distributions across layers of a ternary neural network. Maybe you're fitting a generative model to quantized sensor readings.

You could hack this: fit a Gaussian mixture model and round the parameters. But that's wrong in specific ways. A Gaussian has infinite support — it assigns probability to values like 0.37 and -2.1 that your data can never take. Its sufficient statistics are mean and variance, not counts. Its log-likelihood landscape is different from a discrete distribution's. And KL divergence between Gaussians doesn't have the clean interpretation it has for ternary distributions.

A ternary distribution is fully specified by three numbers (p₋₁, p₀, p₁) summing to 1. That's it. No covariance matrix, no Cholesky decomposition, no numerical issues with near-singular matrices. The M-step of EM is just weighted counting. The E-step is one division per data point per component. This is as simple as EM gets.

## The Insight

The EM algorithm for ternary mixtures is a counting machine:

1. **E-step**: For each data point xₙ and each component k, compute the posterior responsibility γ(zₙₖ) = πₖ·pₖ(xₙ) / Σⱼ πⱼ·pⱼ(xₙ). This is one multiply and one divide per data-component pair.

2. **M-step**: For each component k, the new weight is πₖ = Nₖ/N (where Nₖ = Σₙ γ(zₙₖ)). The new distribution parameters are pₖ(x) = Σₙ γ(zₙₖ)·𝟙(xₙ = x) / Nₖ. This is just counting how many -1s, 0s, and +1s each component is responsible for, weighted by the responsibilities.

3. **Log-likelihood**: LL = Σₙ ln(Σₖ πₖ·pₖ(xₙ)). Check if |LL_new - LL_old| < tol. If so, you've converged.

No gradients. No learning rates. No optimization landscape. Just counting and dividing. The log-likelihood is guaranteed to be non-decreasing across iterations — this is a fundamental property of EM, not an implementation detail.

## How It Works

### TernaryDistribution

The building block. Three probabilities summing to 1, with auto-normalization on construction. Supports PMF, log-PMF, mean (E[X] = p_pos - p_neg), variance (E[X²] - E[X]²), and deterministic sampling from a uniform random u ∈ [0, 1).

The mean is interesting: it ranges from -1 (all negative) to +1 (all positive), passing through 0 at either uniform or symmetric distributions. This gives you a natural ordering on the component space.

### The EM loop

```
Initialize: K components with weights πₖ and distributions pₖ(x)
Repeat:
    E-step:  γ(zₙₖ) = πₖ · pₖ(xₙ) / Σⱼ πⱼ · pⱼ(xₙ)    [responsibility matrix]
    M-step:  πₖ = Σₙ γ(zₙₖ) / N                            [mixing weights]
             pₖ(x) = Σₙ γ(zₙₖ)·𝟙(xₙ=x) / Σₙ γ(zₙₖ)      [ternary counts]
    Check:   LL = Σₙ ln(Σₖ πₖ·pₖ(xₙ))
Until |LL_new - LL_old| < tol or max_iter reached
```

The `floor` parameter (default 1e-300) prevents log(0). Without it, a component that assigns zero probability to an observed value produces -inf log-likelihood, which propagates through the E-step and destroys convergence. The floor is a computational necessity, not an approximation.

### Data-driven initialization

`init_from_data(k, data)` splits the data into K equal chunks and estimates a distribution from each chunk's statistics, plus uniform weights. It also adds a small Laplace-like pseudocount (0.1) to avoid zero probabilities. This is a naive heuristic — good enough for well-separated clusters, but not robust to pathological initializations.

### Divergence measures

KL(P‖Q) and JS(P‖Q) operate on the three-point PMF directly. KL is not symmetric; JS is symmetric and bounded by ln(2). These are useful for comparing fitted components across layers or across training runs.

## Code Example

```rust
use ternary_em::{
    TernaryEM, TernaryDistribution, MixtureComponent, EMConfig,
    kl_divergence, js_divergence,
};

// ── Define a mixture and run EM ──
let data = vec![
    -1i8, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 10 negatives
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,              // 10 positives
];

let init = vec![
    MixtureComponent {
        weight: 0.5,
        distribution: TernaryDistribution::new(0.7, 0.2, 0.1), // biased toward -1
    },
    MixtureComponent {
        weight: 0.5,
        distribution: TernaryDistribution::new(0.1, 0.2, 0.7), // biased toward +1
    },
];

let result = TernaryEM::with_config(init, EMConfig {
    max_iter: 500,
    tol: 1e-10,
    floor: 1e-300,
}).fit(&data);

println!("Converged: {} in {} iterations", result.converged, result.iterations);
println!("Log-likelihood: {:.4}", result.log_likelihood);

for (i, comp) in result.components.iter().enumerate() {
    let d = &comp.distribution;
    println!("Component {}: weight={:.3}, p(-1)={:.3}, p(0)={:.3}, p(+1)={:.3}",
        i, comp.weight, d.p_neg, d.p_zero, d.p_pos);
}

// ── Data-driven initialization (no manual seeding) ──
let em = TernaryEM::init_from_data(3, &data);
let result = em.fit(&data);

// ── Divergence measures ──
let p = TernaryDistribution::new(0.5, 0.3, 0.2);
let q = TernaryDistribution::new(0.1, 0.3, 0.6);

let kl = kl_divergence(&p, &q);    // KL(P‖Q) ≥ 0, zero iff P=Q
let js = js_divergence(&p, &q);    // symmetric, bounded by ln(2)
assert!((js_divergence(&p, &q) - js_divergence(&q, &p)).abs() < 1e-10);

// ── Distribution properties ──
let d = TernaryDistribution::new(0.5, 0.3, 0.2);
d.mean();       // 0.2 - 0.5 = -0.3
d.variance();   // 0.7 - 0.09 = 0.61
d.pmf(-1);      // 0.5
d.log_pmf(1);   // ln(0.2)
d.sample(0.1);  // -1 (falls in first 50%)
d.sample(0.9);  // +1 (falls in last 20%, i.e. u >= p_neg + p_zero = 0.8)

// ── Uniform distribution ──
let u = TernaryDistribution::uniform();
// p_neg = p_zero = p_pos = 1/3

// ── Log-likelihood history (for debugging convergence) ──
for (i, &ll) in result.ll_history.iter().enumerate() {
    println!("  iter {}: LL = {:.6}", i, ll);
}
```

## Module Map

Everything in `src/lib.rs`.

```
TernaryDistribution     — PMF over {-1, 0, +1}
  ::new(p_neg, p_zero, p_pos)  — auto-normalizes
  ::uniform()                   — equal 1/3 probabilities
  .pmf(x)                       — probability of x
  .log_pmf(x)                   — ln(pmf), -inf for impossible values
  .mean()                       — p_pos - p_neg
  .variance()                   — E[X²] - E[X]²
  .sample(u)                    — deterministic sample from u ∈ [0,1)

MixtureComponent         — weight + TernaryDistribution
EMConfig                 — max_iter, tol, floor
EMResult                 — components, log_likelihood, iterations, converged, ll_history

TernaryEM                — the algorithm
  ::new(components)             — default config
  ::with_config(components, cfg) — custom config
  ::init_from_data(k, data)     — heuristic initialization
  .e_step(data)                 — responsibility matrix [n × k]
  .m_step(data, responsibilities) — update parameters in place
  .log_likelihood(data)         — current LL
  .fit(data)                    — run to convergence → EMResult

kl_divergence(p, q)      — KL(P‖Q)
js_divergence(p, q)      — symmetric Jensen-Shannon divergence
```

## Design Decisions

**Auto-normalization on construction.** `TernaryDistribution::new(1.0, 2.0, 3.0)` produces (1/6, 2/6, 3/6). This lets you pass raw counts directly from data without pre-normalizing. The tradeoff: it silently fixes invalid inputs instead of erroring. If you pass (0, 0, 0), it panics.

**The `floor` parameter prevents -inf.** Without it, a component that assigns zero probability to an observed value produces log-likelihood = -∞, which makes the E-step responsibility for that component 0/0. The floor (default 1e-300) keeps everything finite. It's small enough to not affect results meaningfully, but it's a computational scaffold, not a modeling choice.

**`init_from_data` uses equal chunking.** Data point 0 goes to component 0, point 1 to component 1, ..., wrapping around. This means the initialization is sensitive to data ordering. A shuffled dataset gives different initial components than a sorted one. K-means++ style initialization (adapted for ternary distances) would be more robust.

**`fit` consumes self.** The `fit` method takes `self` (not `&mut self`), returning an `EMResult` that owns the fitted components. This means you can't resume EM after convergence or inspect intermediate state without using the `ll_history`. It also means you can't call `e_step` or `m_step` after `fit` — the `TernaryEM` struct is gone. This is a deliberate choice for API simplicity but limits flexibility.

**Sampling is deterministic from a uniform u.** The `sample` method takes `u: f64` instead of an RNG. This makes the method pure (no side effects, no mutation) and testable. The caller provides randomness; the distribution provides the mapping.

**No BIC/AIC for choosing K.** You have to run EM with multiple K values and compare log-likelihoods manually. There's no penalized likelihood criterion built in. For small K (2-5), this is fine. For model selection at scale, you'd need to add it.

## Status

- **16 tests passing.** Distribution creation and normalization, uniform, PMF, mean/variance, E-step produces valid probabilities, M-step improves likelihood, convergence on 2-component mixture, known parameter recovery, convergence detection, monotonic log-likelihood increase, KL divergence self-consistency (KL(P‖P) ≈ 0), JS symmetry, init_from_data, deterministic sampling, log-PMF.
- **Production-ready for analysis.** The EM implementation is correct, converges, and preserves the monotonic likelihood guarantee.
- **Known gaps:**
  - No online/incremental EM for streaming data
  - No BIC/AIC for automatic component selection
  - `init_from_data` is naive (equal chunking, not k-means++)
  - `fit` consumes the struct — can't inspect or resume after convergence
  - O(N × K) memory for the responsibility matrix; no minibatch support
  - No regularization to prevent component collapse

## Ecosystem

- [`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize) — produces ternary data that this crate clusters
- [`ternary-optimizer`](https://github.com/SuperInstance/ternary-optimizer) — training loop for ternary networks
- [`ternary-svm`](https://github.com/SuperInstance/ternary-svm) — classification alternative (discriminative vs generative)
- [`ternary-types`](https://github.com/SuperInstance/ternary-types) — shared trait definitions

## License

MIT
