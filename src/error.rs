use thiserror::Error;

/// Why a model could not be fitted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A regressor or factor does not have one value per observation.
    #[error("'{name}' has {found} values, but the outcome has {expected}")]
    LengthMismatch {
        /// The name of the regressor or factor.
        name: String,
        /// The number of observations.
        expected: usize,
        /// The number of values of the regressor or factor.
        found: usize,
    },

    /// The cluster labels do not have one label per observation.
    #[error("got {found} cluster labels, but the outcome has {expected} values")]
    ClusterLengthMismatch {
        /// The number of observations.
        expected: usize,
        /// The number of cluster labels.
        found: usize,
    },

    /// The outcome contains a NaN or infinite value.
    #[error("outcome is not finite in row {row}")]
    NonFiniteOutcome {
        /// The zero-based index of the first offending observation.
        row: usize,
    },

    /// A regressor contains a NaN or infinite value.
    #[error("regressor '{name}' is not finite in row {row}")]
    NonFiniteRegressor {
        /// The regressor's name.
        name: String,
        /// The zero-based index of the first offending observation.
        row: usize,
    },

    /// There are fewer observations than parameters, intercept included.
    #[error("not enough observations: {observations} rows for {parameters} parameters")]
    TooFewObservations {
        /// The number of observations.
        observations: usize,
        /// The number of parameters, intercept included.
        parameters: usize,
    },

    /// A regressor or factor name contains `:`, which is reserved to
    /// separate factors from their levels.
    #[error("name '{name}' contains ':', which separates factors from their levels")]
    InvalidName {
        /// The offending name.
        name: String,
    },

    /// A name is used twice or collides with the reserved `!Intercept` name.
    #[error("'{name}' occurs more than once")]
    DuplicateName {
        /// The repeated name.
        name: String,
    },

    /// A factor's reference level does not occur in its labels.
    #[error("reference level '{reference}' does not occur in factor '{factor}'")]
    UnknownReference {
        /// The factor's name.
        factor: String,
        /// The missing reference level.
        reference: String,
    },

    /// A factor has a single level, so it has no dummies.
    #[error("factor '{factor}' has only one level")]
    SingleLevel {
        /// The factor's name.
        factor: String,
    },

    /// The regressors (with the intercept) are perfectly collinear.
    #[error("X'X is singular")]
    Singular,
}
