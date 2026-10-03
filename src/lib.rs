//! Ordinary least squares with robust inference by default.
//!
//! Every model includes an intercept. Standard errors are HC3
//! (MacKinnon and White, 1985), or one-way clustered HC3 if the model is
//! [clustered](Ols::cluster). Beyond the usual statistics, every [`Fit`]
//! reports a robust Wald F-test, the leave-one-out R² with the PRESS
//! statistic (Hansen, *Econometrics*, Section 4.18), the condition number of
//! the design matrix and the maximum leverage.
//!
//! Statistics that are not defined, such as standard errors without residual
//! degrees of freedom, are `None`.
//!
//! ```
//! let y = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
//! let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let fit = justols::Ols::new(&y).regressor("x", &x).fit()?;
//!
//! let slope = fit.coefficient("x").unwrap();
//! assert!((slope.estimate - 0.8857142857142857).abs() < 1e-12);
//! assert!(slope.p_value.unwrap() < 0.05);
//! assert!(fit.loo_r_squared().unwrap() < fit.r_squared());
//! # Ok::<(), justols::Error>(())
//! ```
//!
//! Clustering takes one label of any hashable type per observation:
//!
//! ```
//! # let y = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
//! # let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let firm = ["a", "a", "b", "b", "c", "c"];
//! let fit = justols::Ols::new(&y).regressor("x", &x).cluster(firm).fit()?;
//! assert_eq!(fit.n_clusters(), Some(3));
//! assert_eq!(fit.inference_df(), 2);
//! # Ok::<(), justols::Error>(())
//! ```

mod error;
mod fit;
mod ols;

pub use error::Error;
pub use fit::{Coefficient, Fit, Term, WaldTest};
pub use ols::Ols;
