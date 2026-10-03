# justols

`justols` is a small Rust command-line tool for running OLS regressions on CSV
data. It always includes a constant and prints tab-separated output that is easy
for other programs to parse.

Unlike many minimal OLS tools, it reports HC3 standard errors, a robust Wald
F-statistic (using the HC3 covariance), a Leave-One-Out R² ($\widetilde{R}^2$)
with the raw PRESS statistic, the condition number of the design matrix, and the
maximum leverage ($\max h_{ii}$), all by default.

For more information on the LOO R², see the
[PRESS statistic](https://en.wikipedia.org/wiki/PRESS_statistic) and Bruce E.
Hansen, [*Econometrics*](https://users.ssc.wisc.edu/~behansen/econometrics/),
Section 4.18.

The tool supports intercept-only regressions, but does not support regressions
without a constant. When a statistic is not defined, such as an F-statistic for
an intercept-only model, it is printed as `NaN`.

## Building this tool

`cargo build --release`

## Usage

`cargo run --release -- data/test.csv outcome [x1] [x2] ...`

or, after building:

`target/release/justols data/test.csv outcome [x1] [x2] ...`

The first argument is the CSV file, the second is the outcome column, and any
remaining arguments are regressor columns. Column names are read from the CSV
header row.

To compute one-way clustered HC3 standard errors, pass a single cluster column:

`target/release/justols --cluster subject_id data/test.csv outcome [x1] [x2] ...`

The cluster column must be present in the CSV. Its values are read as raw CSV
fields and grouped by exact equality, so cluster IDs may be strings, integers,
or other non-numeric labels.

To include a categorical regressor, write it as `column:reference`. Its values are read as raw CSV fields, and each level other than the reference becomes a dummy named `column:level`, in order of first appearance:

`target/release/justols data.csv outcome x treatment:control`

Any column can be used this way, including integer codes such as `year:2020`. Because `:` marks factors, `justols` refuses CSV files with a `:` in any column name.

For an intercept-only regression, omit the regressor columns:

`target/release/justols data/test.csv outcome`

The resulting output can be replicated with `data/test.R`.

## Using it as a library

`justols` is also a Rust library. To use it without the command-line tool and its CSV dependency, add it from GitHub with default features off:

```toml
[dependencies]
justols = { git = "https://github.com/mrpg/justols", default-features = false }
```

Models are built from borrowed slices, and every model includes a constant:

```rust
use justols::Ols;

let fit = Ols::new(&y)
    .regressor("x1", &x1)
    .regressor("x2", &x2)
    .factor("treatment", &arms, "control") // dummies treatment:<level>
    .cluster(&firm_ids) // optional: any hashable label type
    .fit()?;

for c in fit.coefficients() {
    println!("{}: {} (p = {:?})", c.term, c.estimate, c.p_value);
}
println!("LOO R²: {:?}", fit.loo_r_squared());
print!("{fit}"); // the same table the command-line tool prints
```

`fit` returns a `justols::Error` for mismatched lengths, non-finite values, too few observations, a singular design, a name containing `:`, used twice, or equal to the reserved `!Intercept` name, or a factor without its reference level or with a single level. Statistics that are not defined, which the command-line tool prints as `NaN`, are `None` in the library. The data can be anything that is `AsRef<[f64]>`, such as slices, arrays or vectors. Run `cargo doc --open` for the full API.

## Output

Each row is tab-separated. Coefficient rows have five fields:

`name beta hc3-se t-stat p-value`

- `name`: the coefficient name. The intercept is printed as `!Intercept`, and factor dummies as `column:level`.
- `beta`: the OLS coefficient estimate.
- `hc3-se`: the HC3 heteroskedasticity-consistent standard error.
- `t-stat`: `beta / hc3-se`.
- `p-value`: the two-sided p-value for the t-statistic, using `df-resid`
  degrees of freedom. With `--cluster`, the standard error is the one-way
  clustered HC3 standard error and p-values use `n-clusters - 1` degrees of
  freedom.

The remaining rows are scalar model diagnostics:

- `r-squared`: the usual in-sample coefficient of determination,
  `1 - SSE / SST`.
- `r-squared-adj`: adjusted R-squared, penalizing for the number of fitted
  parameters.
- `r-squared-loo`: Leave-One-Out R-squared, computed as `1 - PRESS / SST`.
- `press`: the prediction sum of squares, `sum((e_i / (1 - h_ii))^2)`.
- `residual-se`: the residual standard error, `sqrt(SSE / df-resid)`.
- `f-stat-robust`: a robust Wald F-statistic for the joint null that all
  non-intercept coefficients are zero, using the HC3 covariance matrix.
- `f-pvalue-robust`: the p-value for `f-stat-robust`, using `df-model` and
  `df-resid` degrees of freedom. With `--cluster`, the denominator degrees of
  freedom are `n-clusters - 1`.
- `n`: the number of observations.
- `df-model`: the number of non-intercept regressors.
- `df-resid`: residual degrees of freedom, `n` minus the number of fitted
  parameters including the intercept.
- `n-clusters`: the number of distinct cluster values. Printed only when
  `--cluster` is used.
- `condition-number`: the ratio of the largest to smallest singular value of
  the design matrix. Larger values indicate more severe collinearity or scaling
  problems.
- `max-leverage`: the largest diagonal element of the hat matrix, `max h_ii`.

The HC3 covariance estimator follows MacKinnon and White (1985), "Some
Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite
Sample Properties". With `--cluster`, `justols` forms HC3-adjusted score
contributions for each observation, sums them within each cluster, and uses the
outer product of those cluster sums as the sandwich meat. The Leave-One-Out
R-squared is described in Hansen, Section 4.18, and PRESS is explained
[here](https://en.wikipedia.org/wiki/PRESS_statistic).

## License

0BSD
