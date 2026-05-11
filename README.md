# justols

This is a simple Rust tool that simply just runs OLS on CSV data. Crucially, and unlike similar tools, **by default** it reports *(i)* HC3 standard errors and *(ii)* a Leave-One-Out R², $\widetilde{R}^2$.

For more information on the latter statistic, see [here](https://en.wikipedia.org/wiki/PRESS_statistic) and Bruce E. Hansen, [*Econometrics*](https://users.ssc.wisc.edu/~behansen/econometrics/), Section 4.18.

This tool does not support regressions without a constant. The output is designed to be easy to parse by other software. This tool is faster than R and uses less memory.

## Building this tool

`cargo build --release`

## Example usage

`cargo run --release -- data/test.csv outcome [x1] [x2] ...`

The resulting output can be replicated with `data/test.R`.

## License

0BSD
