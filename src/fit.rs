use std::fmt;

/// A term of the model: the intercept or a named regressor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Intercept,
    Regressor(String),
}

impl Term {
    /// The regressor name, or `None` for the intercept.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Intercept => None,
            Self::Regressor(name) => Some(name),
        }
    }
}

/// Displays the intercept as `!Intercept`, which cannot clash with a CSV
/// column name in practice, and a regressor as its name.
impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Intercept => f.write_str("!Intercept"),
            Self::Regressor(name) => f.write_str(name),
        }
    }
}

/// An estimated coefficient with its HC3 (or clustered HC3) inference.
///
/// The inference fields are `None` when there are no residual degrees of
/// freedom, or when clustering with a single cluster.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Coefficient {
    pub term: Term,
    pub estimate: f64,
    pub std_error: Option<f64>,
    /// `estimate / std_error`.
    pub t_stat: Option<f64>,
    /// Two-sided p-value of `t_stat` with [`Fit::inference_df`] degrees of
    /// freedom.
    pub p_value: Option<f64>,
}

/// A robust Wald test of the joint null that all slopes are zero.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct WaldTest {
    /// The Wald statistic divided by `df_num`, an F-statistic.
    pub f_stat: f64,
    pub df_num: usize,
    pub df_denom: usize,
    pub p_value: f64,
}

/// A fitted OLS model. Returned by [`Ols::fit`](crate::Ols::fit).
#[derive(Debug, Clone)]
pub struct Fit {
    pub(crate) coefficients: Vec<Coefficient>,
    pub(crate) fitted: Vec<f64>,
    pub(crate) residuals: Vec<f64>,
    pub(crate) leverages: Vec<f64>,
    pub(crate) r_squared: f64,
    pub(crate) press: Option<f64>,
    pub(crate) ss_tot: f64,
    pub(crate) wald_test: Option<WaldTest>,
    pub(crate) n_clusters: Option<usize>,
    pub(crate) condition_number: f64,
}

impl Fit {
    /// All coefficients, the intercept first, then the regressors in the
    /// order they were added.
    #[must_use]
    pub fn coefficients(&self) -> &[Coefficient] {
        &self.coefficients
    }

    #[must_use]
    pub fn intercept(&self) -> &Coefficient {
        &self.coefficients[0]
    }

    /// The coefficient of the regressor called `name`.
    #[must_use]
    pub fn coefficient(&self, name: &str) -> Option<&Coefficient> {
        self.coefficients[1..]
            .iter()
            .find(|c| c.term.name() == Some(name))
    }

    /// In-sample $R^2 = 1 - SSE / SST$. NaN if the outcome is constant.
    #[must_use]
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// Adjusted $R^2$. `None` without residual degrees of freedom.
    #[must_use]
    pub fn adj_r_squared(&self) -> Option<f64> {
        let n = self.n_observations() as f64;
        (self.df_resid() > 0)
            .then(|| 1.0 - (1.0 - self.r_squared) * (n - 1.0) / self.df_resid() as f64)
    }

    /// Leave-one-out $R^2 = 1 - PRESS / SST$. `None` without residual
    /// degrees of freedom.
    #[must_use]
    pub fn loo_r_squared(&self) -> Option<f64> {
        self.press.map(|press| 1.0 - press / self.ss_tot)
    }

    /// The prediction sum of squares, $\sum_i (e_i / (1 - h_{ii}))^2$.
    /// `None` without residual degrees of freedom.
    #[must_use]
    pub fn press(&self) -> Option<f64> {
        self.press
    }

    /// $\sqrt{SSE / df_{resid}}$. `None` without residual degrees of freedom.
    #[must_use]
    pub fn residual_se(&self) -> Option<f64> {
        (self.df_resid() > 0).then(|| {
            let sse: f64 = self.residuals.iter().map(|e| e * e).sum();
            (sse / self.df_resid() as f64).sqrt()
        })
    }

    /// The robust Wald test that all slopes are zero. `None` for an
    /// intercept-only model, without inference, or if the slope covariance is
    /// singular.
    #[must_use]
    pub fn wald_test(&self) -> Option<WaldTest> {
        self.wald_test
    }

    #[must_use]
    pub fn n_observations(&self) -> usize {
        self.residuals.len()
    }

    /// The number of regressors, excluding the intercept.
    #[must_use]
    pub fn df_model(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Observations minus parameters, intercept included.
    #[must_use]
    pub fn df_resid(&self) -> usize {
        self.n_observations() - self.coefficients.len()
    }

    /// The number of distinct clusters, or `None` without clustering.
    #[must_use]
    pub fn n_clusters(&self) -> Option<usize> {
        self.n_clusters
    }

    /// Degrees of freedom of the t and F reference distributions:
    /// [`df_resid`](Self::df_resid), or `n_clusters - 1` when clustering.
    #[must_use]
    pub fn inference_df(&self) -> usize {
        self.n_clusters
            .map_or(self.df_resid(), |g| g.saturating_sub(1))
    }

    /// The ratio of the largest to the smallest singular value of the design
    /// matrix, intercept column included.
    #[must_use]
    pub fn condition_number(&self) -> f64 {
        self.condition_number
    }

    /// The largest diagonal element of the hat matrix, $\max_i h_{ii}$.
    #[must_use]
    pub fn max_leverage(&self) -> f64 {
        self.leverages.iter().copied().fold(0.0, f64::max)
    }

    /// The diagonal of the hat matrix, $h_{ii}$.
    #[must_use]
    pub fn leverages(&self) -> &[f64] {
        &self.leverages
    }

    #[must_use]
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }
}
