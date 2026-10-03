use std::fmt;

/// A term of the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// The constant, which every model includes.
    Intercept,
    /// A regressor, by the name it was added with.
    Regressor(String),
    /// The dummy of a factor that is 1 where the factor equals `level`.
    Level {
        /// The factor's name.
        factor: String,
        /// The level this dummy indicates.
        level: String,
    },
}

/// Displays the intercept as `!Intercept`, a regressor as its name, and a
/// factor level as `factor:level`.
impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Intercept => f.write_str("!Intercept"),
            Self::Regressor(name) => f.write_str(name),
            Self::Level { factor, level } => write!(f, "{factor}:{level}"),
        }
    }
}

/// Compares with the [`Display`](fmt::Display) form, so
/// `term == "group:b"` works.
impl PartialEq<str> for Term {
    fn eq(&self, name: &str) -> bool {
        match self {
            Self::Intercept => name == "!Intercept",
            Self::Regressor(regressor) => regressor == name,
            Self::Level { factor, level } => name
                .split_once(':')
                .is_some_and(|(f, l)| f == factor && l == level),
        }
    }
}

impl PartialEq<&str> for Term {
    fn eq(&self, name: &&str) -> bool {
        self == *name
    }
}

/// An estimated coefficient with its HC3 (or clustered HC3) inference.
///
/// The inference fields are `None` when there are no residual degrees of
/// freedom, or when clustering with a single cluster.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Coefficient {
    /// The term this coefficient belongs to.
    pub term: Term,
    /// The OLS point estimate.
    pub estimate: f64,
    /// The HC3 or clustered HC3 standard error.
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
    /// Numerator degrees of freedom, the number of slopes.
    pub df_num: usize,
    /// Denominator degrees of freedom, [`Fit::inference_df`].
    pub df_denom: usize,
    /// The p-value of `f_stat`.
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

    /// The intercept, which every model includes.
    #[must_use]
    pub fn intercept(&self) -> &Coefficient {
        &self.coefficients[0]
    }

    /// The coefficient of the term displayed as `name`, such as `"x"` for a
    /// regressor or `"group:b"` for a factor level.
    #[must_use]
    pub fn coefficient(&self, name: &str) -> Option<&Coefficient> {
        self.coefficients.iter().find(|c| c.term == name)
    }

    /// In-sample R², `1 - SSE / SST`. NaN if the outcome is constant.
    #[must_use]
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// Adjusted R². `None` without residual degrees of freedom.
    #[must_use]
    pub fn adj_r_squared(&self) -> Option<f64> {
        let n = self.n_observations() as f64;
        (self.df_resid() > 0)
            .then(|| 1.0 - (1.0 - self.r_squared) * (n - 1.0) / self.df_resid() as f64)
    }

    /// Leave-one-out R², `1 - PRESS / SST`. `None` without residual
    /// degrees of freedom.
    #[must_use]
    pub fn loo_r_squared(&self) -> Option<f64> {
        self.press.map(|press| 1.0 - press / self.ss_tot)
    }

    /// The prediction sum of squares, `sum((e_i / (1 - h_ii))^2)`.
    /// `None` without residual degrees of freedom.
    #[must_use]
    pub fn press(&self) -> Option<f64> {
        self.press
    }

    /// The residual standard error, `sqrt(SSE / df_resid)`. `None` without residual degrees of freedom.
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

    /// The number of observations.
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

    /// The largest diagonal element of the hat matrix, `max h_ii`.
    #[must_use]
    pub fn max_leverage(&self) -> f64 {
        self.leverages.iter().copied().fold(0.0, f64::max)
    }

    /// The diagonal of the hat matrix, `h_ii`.
    #[must_use]
    pub fn leverages(&self) -> &[f64] {
        &self.leverages
    }

    /// The fitted values, `X * beta`.
    #[must_use]
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// The residuals, `y - X * beta`.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }
}

/// Formats the fit as the tab-separated table printed by the `justols`
/// command, one row per coefficient and per statistic, with `NaN` for
/// statistics that are not defined. The README documents every row.
impl fmt::Display for Fit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let or_nan = |x: Option<f64>| x.unwrap_or(f64::NAN);

        for c in &self.coefficients {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                c.term,
                c.estimate,
                or_nan(c.std_error),
                or_nan(c.t_stat),
                or_nan(c.p_value),
            )?;
        }

        let wald = self.wald_test;
        let mut row = |name: &str, value: &dyn fmt::Display| writeln!(f, "{name}\t{value}");
        row("r-squared", &self.r_squared)?;
        row("r-squared-adj", &or_nan(self.adj_r_squared()))?;
        row("r-squared-loo", &or_nan(self.loo_r_squared()))?;
        row("press", &or_nan(self.press))?;
        row("residual-se", &or_nan(self.residual_se()))?;
        row("f-stat-robust", &or_nan(wald.map(|w| w.f_stat)))?;
        row("f-pvalue-robust", &or_nan(wald.map(|w| w.p_value)))?;
        row("n", &self.n_observations())?;
        row("df-model", &self.df_model())?;
        row("df-resid", &self.df_resid())?;
        if let Some(g) = self.n_clusters {
            row("n-clusters", &g)?;
        }
        row("condition-number", &self.condition_number)?;
        row("max-leverage", &self.max_leverage())
    }
}
