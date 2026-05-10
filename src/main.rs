use csv::ReaderBuilder;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};
use std::{env, process};

struct OlsResult {
    names: Vec<String>,
    beta: DVector<f64>,
    se: DVector<f64>,
    t_stats: DVector<f64>,
    p_values: Vec<f64>,
    r_squared: f64,
    adj_r_squared: f64,
    loo_r_squared: f64,
    f_stat: f64,
    f_p_value: f64,
    n: usize,
    k: usize,
    residual_se: f64,
}

fn run_ols(
    data: &[Vec<f64>],
    col_indices: &[usize],
    dep_idx: usize,
    col_names: &[String],
) -> OlsResult {
    let n = data.len();
    let k = col_indices.len() + 1;
    if n <= k {
        eprintln!(
            "Error: not enough observations for model. Need more rows than parameters (rows: {}, parameters: {}).",
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
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n_f - 1.0) / (n_f - k_f);

    // HC3: Var(b) = (X'X)^-1 X' diag(e_i^2 / (1 - h_ii)^2) X (X'X)^-1
    // Compute h_ii without forming full hat matrix
    let mut meat = DMatrix::zeros(k, k);
    let mut press = 0.0;
    for i in 0..n {
        let x_i = x.row(i).transpose();
        let h_ii = (x.row(i) * &xtx_inv * &x_i)[(0, 0)];
        let w = residuals[i] / (1.0 - h_ii);
        press += w * w;
        let wx = &x_i * w;
        meat += &wx * wx.transpose();
    }
    let vcov = &xtx_inv * &meat * &xtx_inv;

    let loo_r_squared = 1.0 - press / ss_tot;

    let se = DVector::from_fn(k, |i, _| vcov[(i, i)].sqrt());
    let t_stats = DVector::from_fn(k, |i, _| beta[i] / se[i]);
    let df_resid = n_f - k_f;
    let t_dist = StudentsT::new(0.0, 1.0, df_resid).unwrap();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
        .collect();

    let residual_se = (ss_res / df_resid).sqrt();

    let ss_reg = ss_tot - ss_res;
    let df_reg = k_f - 1.0;
    let f_stat = (ss_reg / df_reg) / (ss_res / df_resid);
    let f_dist = FisherSnedecor::new(df_reg, df_resid).unwrap();
    let f_p_value = 1.0 - f_dist.cdf(f_stat);

    let mut names = Vec::new();
    names.push("!Intercept".to_string());
    for &ci in col_indices {
        names.push(col_names[ci].clone());
    }

    OlsResult {
        names,
        beta,
        se,
        t_stats,
        p_values,
        r_squared,
        adj_r_squared,
        loo_r_squared,
        f_stat,
        f_p_value,
        n,
        k,
        residual_se,
    }
}

fn print_results(r: &OlsResult) {
    for i in 0..r.k {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            r.names[i], r.beta[i], r.se[i], r.t_stats[i], r.p_values[i]
        );
    }
    println!("r-squared\t{}", r.r_squared);
    println!("r-squared-adj\t{}", r.adj_r_squared);
    println!("r-squared-loo\t{}", r.loo_r_squared);
    println!("residual-se\t{}", r.residual_se);
    println!("f-stat\t{}", r.f_stat);
    println!("f-pvalue\t{}", r.f_p_value);
    println!("n\t{}", r.n);
    println!("df-model\t{}", r.k - 1);
    println!("df-resid\t{}", r.n - r.k);
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

    if positional.len() < 3 {
        eprintln!("Usage: justols [flags] <data.csv> <outcome> <x1> [x2] ...");
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
