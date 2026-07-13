//! # ternary-em
//!
//! Expectation-Maximization with ternary latent variables.
//!
//! This crate implements EM algorithms where latent states take values in
//! `{-1, 0, +1}` (ternary). It supports mixture models of ternary distributions,
//! E-step responsibility computation, M-step parameter updates, and convergence
//! detection via log-likelihood monitoring.


/// A ternary probability distribution over {-1, 0, +1}.
///
/// Probabilities must sum to 1.0 and all be non-negative.
#[derive(Debug, Clone)]
pub struct TernaryDistribution {
    /// P(X = -1)
    pub p_neg: f64,
    /// P(X = 0)
    pub p_zero: f64,
    /// P(X = +1)
    pub p_pos: f64,
}

impl TernaryDistribution {
    /// Create a new ternary distribution, normalizing if needed.
    pub fn new(p_neg: f64, p_zero: f64, p_pos: f64) -> Self {
        let sum = p_neg + p_zero + p_pos;
        assert!(sum > 0.0, "Probabilities must sum to a positive number");
        TernaryDistribution {
            p_neg: p_neg / sum,
            p_zero: p_zero / sum,
            p_pos: p_pos / sum,
        }
    }

    /// Uniform distribution: each state has probability 1/3.
    pub fn uniform() -> Self {
        TernaryDistribution {
            p_neg: 1.0 / 3.0,
            p_zero: 1.0 / 3.0,
            p_pos: 1.0 / 3.0,
        }
    }

    /// Compute the probability of a given ternary value.
    pub fn pmf(&self, x: i8) -> f64 {
        match x {
            -1 => self.p_neg,
            0 => self.p_zero,
            1 => self.p_pos,
            _ => 0.0,
        }
    }

    /// Compute the log-probability of a given ternary value.
    pub fn log_pmf(&self, x: i8) -> f64 {
        let p = self.pmf(x);
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    }

    /// Expected value: E[X] = (+1)*p_pos + (-1)*p_neg.
    pub fn mean(&self) -> f64 {
        self.p_pos - self.p_neg
    }

    /// Variance: E[X²] - E[X]².
    pub fn variance(&self) -> f64 {
        let e_x2 = self.p_neg + self.p_pos; // (-1)²*pn + 0²*p0 + 1²*pp
        let e_x = self.mean();
        e_x2 - e_x * e_x
    }

    /// Sample a value (deterministic from a uniform random `u` in [0, 1)).
    pub fn sample(&self, u: f64) -> i8 {
        if u < self.p_neg {
            -1
        } else if u < self.p_neg + self.p_zero {
            0
        } else {
            1
        }
    }
}

/// A single component in a ternary mixture model: a weight plus a ternary distribution.
#[derive(Debug, Clone)]
pub struct MixtureComponent {
    /// Mixing weight (must be non-negative; all weights sum to 1).
    pub weight: f64,
    /// The ternary distribution for this component.
    pub distribution: TernaryDistribution,
}

/// Result of running EM on a ternary mixture model.
#[derive(Debug, Clone)]
pub struct EMResult {
    /// Fitted mixture components.
    pub components: Vec<MixtureComponent>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// History of log-likelihoods per iteration.
    pub ll_history: Vec<f64>,
}

/// Configuration for the EM algorithm.
#[derive(Debug, Clone)]
pub struct EMConfig {
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood change.
    pub tol: f64,
    /// Minimum probability floor to prevent log(0).
    pub floor: f64,
}

impl Default for EMConfig {
    fn default() -> Self {
        EMConfig {
            max_iter: 500,
            tol: 1e-8,
            floor: 1e-300,
        }
    }
}

/// EM algorithm for mixtures of ternary distributions.
pub struct TernaryEM {
    config: EMConfig,
    components: Vec<MixtureComponent>,
}

impl TernaryEM {
    /// Create a new EM fitter with initial components and default config.
    pub fn new(components: Vec<MixtureComponent>) -> Self {
        assert!(!components.is_empty(), "Need at least one component");
        TernaryEM {
            config: EMConfig::default(),
            components,
        }
    }

    /// Create with custom config.
    pub fn with_config(components: Vec<MixtureComponent>, config: EMConfig) -> Self {
        assert!(!components.is_empty(), "Need at least one component");
        TernaryEM { config, components }
    }

    /// E-step: compute responsibilities (posterior probabilities) for each data point.
    ///
    /// Returns a matrix `r[n][k]` where `r[n][k]` is the responsibility of
    /// component `k` for data point `n`.
    pub fn e_step(&self, data: &[i8]) -> Vec<Vec<f64>> {
        let n = data.len();
        let k = self.components.len();
        let mut responsibilities = vec![vec![0.0; k]; n];

        for (i, &x) in data.iter().enumerate() {
            let mut total = 0.0;
            let mut unnormalized = vec![0.0; k];

            for (j, comp) in self.components.iter().enumerate() {
                let p = comp.distribution.pmf(x).max(self.config.floor);
                unnormalized[j] = comp.weight * p;
                total += unnormalized[j];
            }

            if total > 0.0 {
                for j in 0..k {
                    responsibilities[i][j] = unnormalized[j] / total;
                }
            } else {
                // Uniform fallback
                let uniform = 1.0 / k as f64;
                for j in 0..k {
                    responsibilities[i][j] = uniform;
                }
            }
        }

        responsibilities
    }

    /// M-step: update component parameters from responsibilities.
    ///
    /// Updates weights and distribution parameters in place.
    pub fn m_step(&mut self, data: &[i8], responsibilities: &[Vec<f64>]) {
        let n = data.len();
        let k = self.components.len();

        for j in 0..k {
            let n_k: f64 = data.iter().enumerate().map(|(i, _)| responsibilities[i][j]).sum();
            if n_k < self.config.floor {
                continue;
            }

            // Update weight
            self.components[j].weight = n_k / n as f64;

            // Update distribution: weighted counts
            let mut count_neg = 0.0;
            let mut count_zero = 0.0;
            let mut count_pos = 0.0;

            for (i, &x) in data.iter().enumerate() {
                let r = responsibilities[i][j];
                match x {
                    -1 => count_neg += r,
                    0 => count_zero += r,
                    1 => count_pos += r,
                    _ => {}
                }
            }

            let total = count_neg + count_zero + count_pos;
            if total > 0.0 {
                self.components[j].distribution = TernaryDistribution {
                    p_neg: count_neg / total,
                    p_zero: count_zero / total,
                    p_pos: count_pos / total,
                };
            }
        }
    }

    /// Compute the log-likelihood of the data under the current model.
    pub fn log_likelihood(&self, data: &[i8]) -> f64 {
        let mut ll = 0.0;
        for &x in data {
            let mut p_x = 0.0;
            for comp in &self.components {
                p_x += comp.weight * comp.distribution.pmf(x).max(self.config.floor);
            }
            ll += p_x.max(self.config.floor).ln();
        }
        ll
    }

    /// Run the EM algorithm to convergence.
    pub fn fit(mut self, data: &[i8]) -> EMResult {
        let mut ll_history = Vec::new();
        let mut prev_ll = f64::NEG_INFINITY;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            // E-step
            let responsibilities = self.e_step(data);

            // M-step
            self.m_step(data, &responsibilities);

            // Log-likelihood
            let ll = self.log_likelihood(data);
            ll_history.push(ll);
            iterations = iter + 1;

            // Convergence check
            if (ll - prev_ll).abs() < self.config.tol {
                converged = true;
                break;
            }
            prev_ll = ll;
        }

        EMResult {
            components: self.components,
            log_likelihood: *ll_history.last().unwrap_or(&0.0),
            iterations,
            converged,
            ll_history,
        }
    }

    /// Initialize a K-component mixture with heuristic seeding from data.
    pub fn init_from_data(k: usize, data: &[i8]) -> Self {
        let n = data.len();
        assert!(n > 0 && k > 0);

        let weight = 1.0 / k as f64;
        let chunk_size = (n + k - 1) / k;

        let components: Vec<MixtureComponent> = (0..k)
            .map(|j| {
                let start = j * chunk_size;
                let end = ((j + 1) * chunk_size).min(n);
                let chunk = &data[start..end];

                let count_neg = chunk.iter().filter(|&&x| x == -1).count() as f64;
                let count_zero = chunk.iter().filter(|&&x| x == 0).count() as f64;
                let count_pos = chunk.iter().filter(|&&x| x == 1).count() as f64;

                MixtureComponent {
                    weight,
                    distribution: TernaryDistribution::new(
                        count_neg + 0.1,
                        count_zero + 0.1,
                        count_pos + 0.1,
                    ),
                }
            })
            .collect();

        TernaryEM::new(components)
    }
}

/// Compute the Kullback-Leibler divergence KL(P || Q) between two ternary distributions.
pub fn kl_divergence(p: &TernaryDistribution, q: &TernaryDistribution) -> f64 {
    let mut kl = 0.0;
    for x in [-1_i8, 0, 1] {
        let pi = p.pmf(x).max(1e-300);
        let qi = q.pmf(x).max(1e-300);
        kl += pi * (pi / qi).ln();
    }
    kl
}

/// Jensen-Shannon divergence between two ternary distributions.
pub fn js_divergence(p: &TernaryDistribution, q: &TernaryDistribution) -> f64 {
    let m = TernaryDistribution {
        p_neg: 0.5 * (p.p_neg + q.p_neg),
        p_zero: 0.5 * (p.p_zero + q.p_zero),
        p_pos: 0.5 * (p.p_pos + q.p_pos),
    };
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_distribution_creation() {
        let d = TernaryDistribution::new(1.0, 2.0, 3.0);
        assert!((d.p_neg - 1.0 / 6.0).abs() < 1e-10);
        assert!((d.p_zero - 2.0 / 6.0).abs() < 1e-10);
        assert!((d.p_pos - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_distribution_sums_to_one() {
        let d = TernaryDistribution::new(0.3, 0.5, 0.2);
        assert!((d.p_neg + d.p_zero + d.p_pos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_distribution() {
        let d = TernaryDistribution::uniform();
        assert!((d.p_neg - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.p_zero - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.p_pos - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pmf() {
        let d = TernaryDistribution::new(1.0, 1.0, 1.0);
        assert!((d.pmf(-1) - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.pmf(0) - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.pmf(1) - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.pmf(2)).abs() < 1e-10);
    }

    #[test]
    fn test_mean_and_variance() {
        let d = TernaryDistribution::new(0.2, 0.3, 0.5);
        assert!((d.mean() - (0.5 - 0.2)).abs() < 1e-10);
        let var = d.variance();
        assert!(var >= 0.0);
        assert!((var - (0.7 - 0.3_f64.powi(2))).abs() < 1e-10);
    }

    #[test]
    fn test_e_step_produces_valid_probabilities() {
        let components = vec![
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.8, 0.1, 0.1),
            },
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.1, 0.1, 0.8),
            },
        ];
        let em = TernaryEM::new(components);
        let data = vec![-1, 1, 0, -1, 1];
        let resp = em.e_step(&data);

        assert_eq!(resp.len(), 5);
        for row in &resp {
            assert_eq!(row.len(), 2);
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Responsibilities must sum to 1");
            for &r in row {
                assert!(r >= 0.0 && r <= 1.0, "Responsibilities must be in [0,1]");
            }
        }
    }

    #[test]
    fn test_m_step_improves_likelihood() {
        let components = vec![
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.5, 0.3, 0.2),
            },
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.2, 0.3, 0.5),
            },
        ];
        let mut em = TernaryEM::new(components.clone());
        let data = vec![-1, -1, -1, 1, 1, 1, 0, 0];

        let ll_before = em.log_likelihood(&data);
        let resp = em.e_step(&data);
        em.m_step(&data, &resp);
        let ll_after = em.log_likelihood(&data);

        // Require a STRICT increase. The previous `>= ll_before - 1e-6` assertion
        // is satisfied by equality, so it passed even when the M-step was a
        // no-op that left the parameters untouched. Here the init is provably
        // not a fixed point, so one real E+M step must strictly raise the
        // log-likelihood (measured delta ≈ +0.049).
        assert!(
            ll_after > ll_before,
            "M-step must strictly increase likelihood (a no-op M-step would tie): \
             before={}, after={}",
            ll_before,
            ll_after
        );
    }

    #[test]
    fn test_converges_on_two_component_mixture() {
        // Generate data from a known 2-component mixture
        let comp1 = TernaryDistribution::new(0.8, 0.1, 0.1);
        let comp2 = TernaryDistribution::new(0.1, 0.1, 0.8);
        let data: Vec<i8> = (0..200)
            .map(|i| {
                if i < 100 {
                    comp1.sample(i as f64 / 300.0)
                } else {
                    comp2.sample((i as f64 - 100.0) / 300.0 + 0.33)
                }
            })
            .collect();

        let init = vec![
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.7, 0.2, 0.1),
            },
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.1, 0.2, 0.7),
            },
        ];

        let em = TernaryEM::with_config(
            init,
            EMConfig {
                max_iter: 500,
                tol: 1e-10,
                floor: 1e-300,
            },
        );
        let result = em.fit(&data);

        assert!(result.converged, "EM should converge");
        assert!(result.iterations < 500);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_known_parameter_recovery() {
        // Simple case: one component dominates -1, another dominates +1
        let data = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 10 neg
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10 pos
        ];

        let init = vec![
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.6, 0.2, 0.2),
            },
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.2, 0.2, 0.6),
            },
        ];

        let result = TernaryEM::new(init).fit(&data);
        assert!(result.converged);

        // One component should have high p_neg, the other high p_pos
        let (c0, c1) = (&result.components[0], &result.components[1]);
        let c0_pos = c0.distribution.p_neg > c0.distribution.p_pos;
        let dominant_neg = if c0_pos { c0 } else { c1 };
        let dominant_pos = if c0_pos { c1 } else { c0 };

        assert!(
            dominant_neg.distribution.p_neg > 0.5,
            "One component should dominate -1"
        );
        assert!(
            dominant_pos.distribution.p_pos > 0.5,
            "One component should dominate +1"
        );
    }

    #[test]
    fn test_convergence_detection() {
        // Trivial single-component data — should converge very fast
        let data = vec![1, 1, 1, 1, 1];
        let init = vec![MixtureComponent {
            weight: 1.0,
            distribution: TernaryDistribution::new(0.1, 0.1, 0.8),
        }];
        let result = TernaryEM::with_config(
            init,
            EMConfig {
                max_iter: 1000,
                tol: 1e-10,
                floor: 1e-300,
            },
        )
        .fit(&data);

        assert!(result.converged);
        assert!(result.iterations < 50);
        assert!(
            result.components[0].distribution.p_pos > 0.9,
            "Should find high p_pos for all-1 data"
        );
    }

    #[test]
    fn test_log_likelihood_increases_monotonically() {
        let data = vec![-1, -1, 0, 0, 1, 1];
        let init = vec![
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.4, 0.3, 0.3),
            },
            MixtureComponent {
                weight: 0.5,
                distribution: TernaryDistribution::new(0.3, 0.3, 0.4),
            },
        ];

        // Log-likelihood of the *initial* parameters, before any EM update.
        let ll_init = TernaryEM::new(init.clone()).log_likelihood(&data);
        let result = TernaryEM::new(init).fit(&data);

        // (1) Monotonicity: across iterations the log-likelihood must be
        // non-decreasing up to floating-point rounding. Exact-arithmetic EM is
        // monotone, but double precision can wiggle by ~1e-15, so use a tight
        // documented tolerance rather than claiming bit-exact monotonicity.
        for window in result.ll_history.windows(2) {
            assert!(
                window[1] >= window[0] - 1e-9,
                "Log-likelihood must be non-decreasing within float tolerance: {} -> {}",
                window[0],
                window[1]
            );
        }

        // (2) The fitted model must strictly beat the initial one. This is what
        // makes the test meaningful: a broken (no-op) M-step that never updates
        // parameters leaves ll_final == ll_init, which check (1) alone cannot
        // detect because a flat sequence trivially satisfies non-decreasing.
        // Here the init is provably improvable, so EM must raise the LL.
        assert!(
            result.log_likelihood > ll_init,
            "EM must strictly improve log-likelihood from the init: init={}, final={}",
            ll_init,
            result.log_likelihood
        );
    }

    #[test]
    fn test_kl_divergence() {
        let p = TernaryDistribution::new(0.5, 0.3, 0.2);
        let same = TernaryDistribution::new(0.5, 0.3, 0.2);
        assert!(kl_divergence(&p, &same) < 1e-10, "KL(P||P) should be ~0");

        let q = TernaryDistribution::new(0.1, 0.3, 0.6);
        assert!(kl_divergence(&p, &q) > 0.0, "KL(P||Q) > 0 when P ≠ Q");
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = TernaryDistribution::new(0.5, 0.3, 0.2);
        let q = TernaryDistribution::new(0.1, 0.3, 0.6);
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        assert!((js_pq - js_qp).abs() < 1e-10, "JS divergence should be symmetric");
        assert!(js_pq >= 0.0);
    }

    #[test]
    fn test_init_from_data() {
        let data = vec![-1, -1, -1, 0, 0, 0, 1, 1, 1];
        let em = TernaryEM::init_from_data(3, &data);
        assert_eq!(em.components.len(), 3);
        let total_weight: f64 = em.components.iter().map(|c| c.weight).sum();
        assert!((total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_deterministic() {
        let d = TernaryDistribution::new(0.2, 0.5, 0.3);
        assert_eq!(d.sample(0.1), -1);
        assert_eq!(d.sample(0.3), 0);
        assert_eq!(d.sample(0.8), 1);
    }

    #[test]
    fn test_log_pmf() {
        let d = TernaryDistribution::new(0.5, 0.3, 0.2);
        assert!((d.log_pmf(-1) - 0.5_f64.ln()).abs() < 1e-10);
        assert!((d.log_pmf(0) - 0.3_f64.ln()).abs() < 1e-10);
        assert!((d.log_pmf(1) - 0.2_f64.ln()).abs() < 1e-10);
        assert_eq!(d.log_pmf(2), f64::NEG_INFINITY);
    }
}
