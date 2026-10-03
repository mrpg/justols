use thiserror::Error;

/// Why a model could not be fitted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A regressor does not have one value per observation.
    #[error("regressor '{name}' has {found} values, but the outcome has {expected}")]
    LengthMismatch {
        name: String,
        expected: usize,
        found: usize,
    },

    /// The cluster labels do not have one label per observation.
    #[error("got {found} cluster labels, but the outcome has {expected} values")]
    ClusterLengthMismatch { expected: usize, found: usize },

    /// The outcome contains a NaN or infinite value.
    #[error("outcome is not finite in row {row}")]
    NonFiniteOutcome { row: usize },

    /// A regressor contains a NaN or infinite value.
    #[error("regressor '{name}' is not finite in row {row}")]
    NonFiniteRegressor { name: String, row: usize },

    /// There are fewer observations than parameters, intercept included.
    #[error("not enough observations: {observations} rows for {parameters} parameters")]
    TooFewObservations {
        observations: usize,
        parameters: usize,
    },

    /// The regressors (with the intercept) are perfectly collinear.
    #[error("X'X is singular")]
    Singular,
}
