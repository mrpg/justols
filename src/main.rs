use csv::ReaderBuilder;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};
use std::{env, process};

struct ResultRow {
    name: String,
    values: Vec<f64>,
}

fn run_ols(
    data: &[Vec<f64>],
    col_indices: &[usize],
    dep_idx: usize,
    col_names: &[String],
) -> Vec<ResultRow> {
    let n = data.len();
    let k = col_indices.len() + 1;
    if n < k {
        eprintln!(
            "Error: not enough observations for model. Need at least as many rows as parameters (rows: {}, parameters: {}).",
            n, k
        );
        process::exit(1);
    }

    let y = DVector::from_fn(n, |i, _| data[i][dep_idx]);

    let x = DMatrix::from_fn(n, k, |i, j| {
        if j == 0 {
            1.0
        } else {
            data[i][col_indices[j - 1]]
        }
    });

    let xtx = x.transpose() * &x;
    let xtx_inv = xtx.try_inverse().unwrap_or_else(|| {
        eprintln!("Error: X'X is singular.");
        process::exit(1);
    });

    let condition_number = {
        let sv = x.singular_values();
        let sigma_max = sv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sigma_min = sv.iter().copied().fold(f64::INFINITY, f64::min);
        if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        }
    };
    let xty = x.transpose() * &y;
    let beta = &xtx_inv * &xty;

    let y_hat = &x * &beta;
    let residuals = &y - &y_hat;
    let n_f = n as f64;
    let k_f = k as f64;

    let ss_res: f64 = residuals.iter().map(|e| e * e).sum();
    let y_mean = y.iter().sum::<f64>() / n_f;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = 1.0 - ss_res / ss_tot;
    let df_resid = n_f - k_f;
    let adj_r_squared = if df_resid > 0.0 {
        1.0 - (1.0 - r_squared) * (n_f - 1.0) / df_resid
    } else {
        f64::NAN
    };

    let mut max_leverage = 0.0_f64;
    for i in 0..n {
        let x_i = x.row(i).transpose();
        let h_ii = (x.row(i) * &xtx_inv * &x_i)[(0, 0)];
        max_leverage = max_leverage.max(h_ii);
    }

    let mut names = Vec::new();
    names.push("!Intercept".to_string());
    for &ci in col_indices {
        names.push(col_names[ci].clone());
    }

    let mut se = DVector::from_element(k, f64::NAN);
    let mut t_stats = DVector::from_element(k, f64::NAN);
    let mut p_values = vec![f64::NAN; k];
    let mut loo_r_squared = f64::NAN;
    let mut press = f64::NAN;
    let mut residual_se = f64::NAN;
    let mut f_stat = f64::NAN;
    let mut f_p_value = f64::NAN;

    if df_resid > 0.0 {
        let mut meat = DMatrix::zeros(k, k);
        press = 0.0;
        for i in 0..n {
            let x_i = x.row(i).transpose();
            let h_ii = (x.row(i) * &xtx_inv * &x_i)[(0, 0)];
            let w = residuals[i] / (1.0 - h_ii);
            press += w * w;
            let wx = &x_i * w;
            meat += &wx * wx.transpose();
        }
        let vcov = &xtx_inv * &meat * &xtx_inv;

        loo_r_squared = 1.0 - press / ss_tot;
        se = DVector::from_fn(k, |i, _| vcov[(i, i)].sqrt());
        t_stats = DVector::from_fn(k, |i, _| beta[i] / se[i]);
        let t_dist = StudentsT::new(0.0, 1.0, df_resid).unwrap();
        p_values = t_stats
            .iter()
            .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
            .collect();
        residual_se = (ss_res / df_resid).sqrt();

        let df_reg = k_f - 1.0;
        if df_reg > 0.0 {
            let q = k - 1;
            let beta_slope = DVector::from_fn(q, |i, _| beta[i + 1]);
            let vcov_slope = DMatrix::from_fn(q, q, |i, j| vcov[(i + 1, j + 1)]);
            if let Some(vcov_slope_inv) = vcov_slope.try_inverse() {
                let wald = (beta_slope.transpose() * &vcov_slope_inv * &beta_slope)[(0, 0)];
                f_stat = wald / df_reg;
                let f_dist = FisherSnedecor::new(df_reg, df_resid).unwrap();
                f_p_value = 1.0 - f_dist.cdf(f_stat);
            }
        }
    }

    let mut rows = Vec::new();
    for i in 0..k {
        rows.push(ResultRow {
            name: names[i].clone(),
            values: vec![beta[i], se[i], t_stats[i], p_values[i]],
        });
    }
    push_scalar(&mut rows, "r-squared", r_squared);
    push_scalar(&mut rows, "r-squared-adj", adj_r_squared);
    push_scalar(&mut rows, "r-squared-loo", loo_r_squared);
    push_scalar(&mut rows, "press", press);
    push_scalar(&mut rows, "residual-se", residual_se);
    push_scalar(&mut rows, "f-stat-robust", f_stat);
    push_scalar(&mut rows, "f-pvalue-robust", f_p_value);
    push_scalar(&mut rows, "n", n as f64);
    push_scalar(&mut rows, "df-model", (k - 1) as f64);
    push_scalar(&mut rows, "df-resid", (n - k) as f64);
    push_scalar(&mut rows, "condition-number", condition_number);
    push_scalar(&mut rows, "max-leverage", max_leverage);
    rows
}

fn push_scalar(rows: &mut Vec<ResultRow>, name: &str, value: f64) {
    rows.push(ResultRow {
        name: name.to_string(),
        values: vec![value],
    });
}

fn print_results(rows: &[ResultRow]) {
    for row in rows {
        print!("{}", row.name);
        for value in &row.values {
            print!("\t{}", value);
        }
        println!();
    }
}

struct Opts {
    csv_path: String,
    dep_var: String,
    indep_vars: Vec<String>,
}

fn parse_args() -> Opts {
    let args: Vec<String> = env::args().skip(1).collect();

    let mut flags = Vec::new();
    let mut positional = Vec::new();
    for arg in &args {
        if arg.starts_with("--") {
            flags.push(arg.as_str());
        } else {
            positional.push(arg.clone());
        }
    }

    if positional.len() < 2 {
        eprintln!("Usage: justols [flags] <data.csv> <outcome> [x1] [x2] ...");
        eprintln!("Flags:");
        eprintln!("  (none currently supported)");
        process::exit(1);
    }

    for flag in &flags {
        eprintln!("Unknown flag: {}", flag);
        process::exit(1);
    }

    Opts {
        csv_path: positional[0].clone(),
        dep_var: positional[1].clone(),
        indep_vars: positional[2..].to_vec(),
    }
}

fn main() {
    let opts = parse_args();

    let csv_path = &opts.csv_path;
    let dep_var = &opts.dep_var;
    let indep_vars: Vec<&str> = opts.indep_vars.iter().map(|s| s.as_str()).collect();

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", csv_path, e);
            process::exit(1);
        });

    let headers: Vec<String> = reader
        .headers()
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let dep_idx = headers
        .iter()
        .position(|h| h == dep_var)
        .unwrap_or_else(|| {
            eprintln!(
                "Error: '{}' not found. Available: {}",
                dep_var,
                headers.join(", ")
            );
            process::exit(1);
        });

    let mut indep_indices = Vec::new();
    for var in &indep_vars {
        let idx = headers.iter().position(|h| h == var).unwrap_or_else(|| {
            eprintln!(
                "Error: '{}' not found. Available: {}",
                var,
                headers.join(", ")
            );
            process::exit(1);
        });
        indep_indices.push(idx);
    }

    let mut data: Vec<Vec<f64>> = Vec::new();
    for result in reader.records() {
        let record = result.unwrap_or_else(|e| {
            eprintln!("Error reading CSV row: {}", e);
            process::exit(1);
        });
        let mut row = vec![0.0; headers.len()];
        for &idx in std::iter::once(&dep_idx).chain(indep_indices.iter()) {
            let field = record.get(idx).unwrap_or_else(|| {
                eprintln!("Error: row has no value for '{}'.", headers[idx]);
                process::exit(1);
            });
            row[idx] = field.parse::<f64>().unwrap_or_else(|_| {
                eprintln!(
                    "Error: non-numeric value '{}' in '{}'.",
                    field, headers[idx]
                );
                process::exit(1);
            });
        }
        data.push(row);
    }

    if data.is_empty() {
        eprintln!("Error: no data rows in CSV.");
        process::exit(1);
    }

    let result = run_ols(&data, &indep_indices, dep_idx, &headers);
    print_results(&result);
}
