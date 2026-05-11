# justols

`justols` is a small Rust command-line tool for running OLS regressions on CSV
data. It always includes a constant and prints tab-separated output that is easy
for other programs to parse.

Unlike many minimal OLS tools, it reports HC3 standard errors and a
Leave-One-Out R², $\widetilde{R}^2$, by default. For more information on the
latter statistic, see the [PRESS statistic](https://en.wikipedia.org/wiki/PRESS_statistic)
and Bruce E. Hansen, [*Econometrics*](https://users.ssc.wisc.edu/~behansen/econometrics/),
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

For an intercept-only regression, omit the regressor columns:

`target/release/justols data/test.csv outcome`

The resulting output can be replicated with `data/test.R`.

## License

0BSD
