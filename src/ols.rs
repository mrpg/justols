use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
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
    inputs: Vec<Input<'a>>,
    clusters: Option<Vec<usize>>,
}

/// A regressor or factor as added to the model, before validation.
#[derive(Debug, Clone)]
enum Input<'a> {
    Regressor {
        name: String,
        values: &'a [f64],
    },
    Factor {
        name: String,
        /// The level of each observation, as an index into `levels`.
        codes: Vec<usize>,
        /// The distinct levels in order of first appearance.
        levels: Vec<String>,
        reference: String,
    },
}

impl Input<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Regressor { name, .. } | Self::Factor { name, .. } => name,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Regressor { values, .. } => values.len(),
            Self::Factor { codes, .. } => codes.len(),
        }
    }
}

/// The columns of the design matrix besides the intercept.
type Columns<'a> = Vec<(Term, Cow<'a, [f64]>)>;

impl<'a> Ols<'a> {
    /// An intercept-only regression of the outcome `y`.
    pub fn new(y: &'a (impl AsRef<[f64]> + ?Sized)) -> Self {
        Self {
            y: y.as_ref(),
            inputs: Vec::new(),
            clusters: None,
        }
    }

    /// Adds a regressor with one value per observation.
    ///
    /// Names must not contain `:`, which separates factors from their levels.
    pub fn regressor(
        mut self,
        name: impl Into<String>,
        values: &'a (impl AsRef<[f64]> + ?Sized),
    ) -> Self {
        self.inputs.push(Input::Regressor {
            name: name.into(),
            values: values.as_ref(),
        });
        self
    }

    /// Adds several `(name, values)` regressors at once.
    pub fn regressors<N, V>(mut self, regressors: impl IntoIterator<Item = (N, &'a V)>) -> Self
    where
        N: Into<String>,
        V: AsRef<[f64]> + ?Sized + 'a,
    {
        for (name, values) in regressors {
            self = self.regressor(name, values);
        }
        self
    }

    /// Adds a categorical regressor with one label per observation, coded as
    /// one dummy for each level except `reference`.
    ///
    /// Levels are identified by their [`Display`] form, and their dummies
    /// are named `name:level`, in order of first appearance. Hence `name`
    /// must not contain `:`, but levels may.
    ///
    /// ```
    /// let y = [1.0, 2.0, 4.0, 3.0, 6.0, 5.0];
    /// let group = ["a", "a", "b", "b", "c", "c"];
    /// let fit = justols::Ols::new(&y).factor("group", group, "a").fit()?;
    ///
    /// let terms: Vec<_> = fit.coefficients().iter().map(|c| c.term.to_string()).collect();
    /// assert_eq!(terms, ["!Intercept", "group:b", "group:c"]);
    /// assert_eq!(fit.coefficient("group:c").unwrap().estimate, 4.0);
    /// # Ok::<(), justols::Error>(())
    /// ```
    pub fn factor<L: Display>(
        mut self,
        name: impl Into<String>,
        labels: impl IntoIterator<Item = L>,
        reference: impl Display,
    ) -> Self {
        let mut ids = HashMap::new();
        let mut levels = Vec::new();
        let codes = labels
            .into_iter()
            .map(|label| {
                *ids.entry(label.to_string()).or_insert_with_key(|level| {
                    levels.push(level.clone());
                    levels.len() - 1
                })
            })
            .collect();
        self.inputs.push(Input::Factor {
            name: name.into(),
            codes,
            levels,
            reference: reference.to_string(),
        });
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
    /// fewer observations than parameters, or are perfectly collinear; if a
    /// name contains `:` or two inputs have the same name; or if a factor has
    /// a single level or lacks its reference level.
    pub fn fit(&self) -> Result<Fit, Error> {
        let columns = self.columns()?;

        let n = self.y.len();
        let k = columns.len() + 1;
        let y = DVector::from_column_slice(self.y);
        let x = DMatrix::from_fn(n, k, |i, j| match j {
            0 => 1.0,
            _ => columns[j - 1].1[i],
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

        let terms =
            std::iter::once(Term::Intercept).chain(columns.into_iter().map(|(term, _)| term));
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
            condition_number: condition_number(&x),
        })
    }

    /// Validates the inputs and expands factors into dummies.
    fn columns(&self) -> Result<Columns<'a>, Error> {
        let n = self.y.len();
        let mut columns = Columns::new();
        let mut names = HashSet::new();
        for input in &self.inputs {
            let name = input.name();
            if name.contains(':') {
                return Err(Error::InvalidName { name: name.into() });
            }
            // Without ':' in names, unique names imply unique terms.
            if !names.insert(name) {
                return Err(Error::DuplicateName { name: name.into() });
            }
            if input.len() != n {
                return Err(Error::LengthMismatch {
                    name: name.into(),
                    expected: n,
                    found: input.len(),
                });
            }
            match input {
                Input::Regressor { values, .. } => {
                    if let Some(row) = values.iter().position(|v| !v.is_finite()) {
                        return Err(Error::NonFiniteRegressor {
                            name: name.into(),
                            row,
                        });
                    }
                    columns.push((Term::Regressor(name.into()), Cow::Borrowed(*values)));
                }
                Input::Factor {
                    codes,
                    levels,
                    reference,
                    ..
                } => {
                    let reference =
                        levels.iter().position(|l| l == reference).ok_or_else(|| {
                            Error::UnknownReference {
                                factor: name.into(),
                                reference: reference.clone(),
                            }
                        })?;
                    if levels.len() < 2 {
                        return Err(Error::SingleLevel {
                            factor: name.into(),
                        });
                    }
                    for (id, level) in levels.iter().enumerate().filter(|&(id, _)| id != reference)
                    {
                        let dummy = codes.iter().map(|&code| f64::from(code == id)).collect();
                        let term = Term::Level {
                            factor: name.into(),
                            level: level.clone(),
                        };
                        columns.push((term, Cow::Owned(dummy)));
                    }
                }
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
        let parameters = columns.len() + 1;
        if n < parameters {
            return Err(Error::TooFewObservations {
                observations: n,
                parameters,
            });
        }
        Ok(columns)
    }
}

fn outer_sum(vectors: &[DVector<f64>], k: usize) -> DMatrix<f64> {
    vectors
        .iter()
        .fold(DMatrix::zeros(k, k), |acc, v| acc + v * v.transpose())
}

// statrs panics on a NaN argument, which arises as 0 / 0 in exact fits.
fn t_p_value(t: f64, df: usize) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let dist = StudentsT::new(0.0, 1.0, df as f64).expect("df is positive");
    2.0 * dist.sf(t.abs())
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
        p_value: if f_stat.is_nan() {
            f64::NAN
        } else {
            dist.sf(f_stat)
        },
    })
}

fn condition_number(x: &DMatrix<f64>) -> f64 {
    let sv = x.clone().singular_values();
    let max = sv.max();
    let min = sv.min();
    if min > 0.0 { max / min } else { f64::INFINITY }
}
