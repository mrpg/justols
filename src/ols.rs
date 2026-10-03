use std::collections::HashMap;
use std::hash::Hash;

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::{Coefficient, Error, Fit, Term, WaldTest};

/// An OLS regression with an intercept, to be estimated with [`fit`](Self::fit).
///
/// The data are borrowed, so building a model is cheap.
#[derive(Debug, Clone)]
#[must_use]
pub struct Ols<'a> {
    y: &'a [f64],
    regressors: Vec<(String, &'a [f64])>,
    clusters: Option<Vec<usize>>,
}

impl<'a> Ols<'a> {
    /// An intercept-only regression of the outcome `y`.
    pub fn new(y: &'a [f64]) -> Self {
        Self {
            y,
            regressors: Vec::new(),
            clusters: None,
        }
    }

    /// Adds a regressor with one value per observation.
    pub fn regressor(mut self, name: impl Into<String>, values: &'a [f64]) -> Self {
        self.regressors.push((name.into(), values));
        self
    }

    /// Adds several regressors at once.
    pub fn regressors<N: Into<String>>(
        mut self,
        regressors: impl IntoIterator<Item = (N, &'a [f64])>,
    ) -> Self {
        self.regressors
            .extend(regressors.into_iter().map(|(n, v)| (n.into(), v)));
        self
    }

    /// Clusters the standard errors by `labels`, one per observation.
    /// Observations with equal labels are in the same cluster.
    pub fn cluster<L: Hash + Eq>(mut self, labels: impl IntoIterator<Item = L>) -> Self {
        let mut ids = HashMap::new();
        let clusters = labels
            .into_iter()
            .map(|label| {
                let next = ids.len();
                *ids.entry(label).or_insert(next)
            })
            .collect();
        self.clusters = Some(clusters);
        self
    }

    /// Estimates the model.
    ///
    /// # Errors
    ///
    /// If the inputs have different lengths, contain non-finite values, have
    /// fewer observations than parameters, or are perfectly collinear.
    pub fn fit(&self) -> Result<Fit, Error> {
        self.validate()?;

        let n = self.y.len();
        let k = self.regressors.len() + 1;
        let y = DVector::from_column_slice(self.y);
        let x = DMatrix::from_fn(n, k, |i, j| match j {
            0 => 1.0,
            _ => self.regressors[j - 1].1[i],
        });

        let xtx_inv = (x.transpose() * &x).try_inverse().ok_or(Error::Singular)?;
        let beta = &xtx_inv * (x.transpose() * &y);
        let fitted = &x * &beta;
        let residuals = &y - &fitted;

        let ss_res: f64 = residuals.iter().map(|e| e * e).sum();
        let y_mean = y.mean();
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        let leverages: Vec<f64> = x
            .row_iter()
            .map(|row| (row * &xtx_inv * row.transpose())[(0, 0)])
            .collect();

        let df_resid = n - k;
        let n_clusters = self
            .clusters
            .as_ref()
            .map(|ids| ids.iter().max().map_or(0, |max| max + 1));

        // HC3 scores: each row of X weighted by e_i / (1 - h_ii).
        let scores: Vec<DVector<f64>> = (0..n)
            .map(|i| x.row(i).transpose() * (residuals[i] / (1.0 - leverages[i])))
            .collect();
        let press = (df_resid > 0).then(|| {
            (0..n)
                .map(|i| (residuals[i] / (1.0 - leverages[i])).powi(2))
                .sum::<f64>()
        });

        let inference_df = n_clusters.map_or(df_resid, |g| g.saturating_sub(1));
        let vcov = (df_resid > 0 && inference_df > 0).then(|| {
            let meat = match (&self.clusters, n_clusters) {
                (Some(ids), Some(g)) => {
                    let mut sums = vec![DVector::zeros(k); g];
                    for (score, &id) in scores.iter().zip(ids) {
                        sums[id] += score;
                    }
                    outer_sum(&sums, k)
                }
                _ => outer_sum(&scores, k),
            };
            &xtx_inv * meat * &xtx_inv
        });

        let terms = std::iter::once(Term::Intercept).chain(
            self.regressors
                .iter()
                .map(|(name, _)| Term::Regressor(name.clone())),
        );
        let coefficients = terms
            .enumerate()
            .map(|(j, term)| {
                let estimate = beta[j];
                let std_error = vcov.as_ref().map(|v| v[(j, j)].sqrt());
                let t_stat = std_error.map(|se| estimate / se);
                Coefficient {
                    term,
                    estimate,
                    std_error,
                    t_stat,
                    p_value: t_stat.map(|t| t_p_value(t, inference_df)),
                }
            })
            .collect();

        let wald_test = vcov
            .as_ref()
            .and_then(|v| wald_test(&beta, v, inference_df));

        Ok(Fit {
            coefficients,
            fitted: fitted.data.into(),
            residuals: residuals.data.into(),
            leverages,
            r_squared: 1.0 - ss_res / ss_tot,
            press,
            ss_tot,
            wald_test,
            n_clusters,
            condition_number: condition_number(x),
        })
    }

    fn validate(&self) -> Result<(), Error> {
        let n = self.y.len();
        for (name, values) in &self.regressors {
            if values.len() != n {
                return Err(Error::LengthMismatch {
                    name: name.clone(),
                    expected: n,
                    found: values.len(),
                });
            }
        }
        if let Some(ids) = &self.clusters
            && ids.len() != n
        {
            return Err(Error::ClusterLengthMismatch {
                expected: n,
                found: ids.len(),
            });
        }
        if let Some(row) = self.y.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteOutcome { row });
        }
        for (name, values) in &self.regressors {
            if let Some(row) = values.iter().position(|v| !v.is_finite()) {
                return Err(Error::NonFiniteRegressor {
                    name: name.clone(),
                    row,
                });
            }
        }
        let parameters = self.regressors.len() + 1;
        if n < parameters {
            return Err(Error::TooFewObservations {
                observations: n,
                parameters,
            });
        }
        Ok(())
    }
}

fn outer_sum(vectors: &[DVector<f64>], k: usize) -> DMatrix<f64> {
    vectors
        .iter()
        .fold(DMatrix::zeros(k, k), |acc, v| acc + v * v.transpose())
}

fn t_p_value(t: f64, df: usize) -> f64 {
    let dist = StudentsT::new(0.0, 1.0, df as f64).expect("df is positive");
    2.0 * (1.0 - dist.cdf(t.abs()))
}

fn wald_test(beta: &DVector<f64>, vcov: &DMatrix<f64>, df_denom: usize) -> Option<WaldTest> {
    let q = beta.len() - 1;
    if q == 0 {
        return None;
    }
    let slopes = beta.rows(1, q);
    let vcov_inv = vcov.view((1, 1), (q, q)).clone_owned().try_inverse()?;
    let f_stat = (slopes.transpose() * vcov_inv * slopes)[(0, 0)] / q as f64;
    let dist = FisherSnedecor::new(q as f64, df_denom as f64).expect("df is positive");
    Some(WaldTest {
        f_stat,
        df_num: q,
        df_denom,
        p_value: 1.0 - dist.cdf(f_stat),
    })
}

fn condition_number(x: DMatrix<f64>) -> f64 {
    let sv = x.singular_values();
    let max = sv.max();
    let min = sv.min();
    if min > 0.0 { max / min } else { f64::INFINITY }
}
