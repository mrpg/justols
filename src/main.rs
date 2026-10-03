//! The `justols` command: OLS on a CSV file, printed as tab-separated rows.

use std::io::{self, Write};
use std::process::ExitCode;

use csv::ReaderBuilder;
use justols::Ols;

const USAGE: &str = "\
Usage: justols [flags] <data.csv> <outcome> [x1] [x2] ...
Flags:
  --cluster <column>    Compute one-way clustered HC3 standard errors.";

struct Opts {
    csv_path: String,
    outcome: String,
    regressors: Vec<String>,
    cluster: Option<String>,
}

/// The columns of the CSV file used by the model.
struct Data {
    outcome: Vec<f64>,
    regressors: Vec<Vec<f64>>,
    clusters: Option<Vec<String>>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args(std::env::args().skip(1))?;
    let data = read_csv(&opts)?;

    let mut model =
        Ols::new(&data.outcome).regressors(opts.regressors.iter().zip(&data.regressors));
    if let Some(labels) = &data.clusters {
        model = model.cluster(labels);
    }
    let fit = model.fit().map_err(|e| format!("Error: {e}."))?;

    write!(io::stdout().lock(), "{fit}").map_err(|e| format!("Error writing output: {e}"))
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    const MISSING_CLUSTER: &str = "Error: --cluster requires a column name.";

    let mut args = args.peekable();
    let mut positional = Vec::new();
    let mut cluster = None;
    while let Some(arg) = args.next() {
        if arg == "--cluster" {
            cluster = Some(args.next().ok_or(MISSING_CLUSTER)?);
        } else if let Some(value) = arg.strip_prefix("--cluster=") {
            if value.is_empty() {
                return Err(MISSING_CLUSTER.into());
            }
            cluster = Some(value.to_owned());
        } else if arg.starts_with("--") {
            return Err(format!("Unknown flag: {arg}"));
        } else {
            positional.push(arg);
        }
    }

    let mut positional = positional.into_iter();
    let (Some(csv_path), Some(outcome)) = (positional.next(), positional.next()) else {
        return Err(USAGE.into());
    };
    Ok(Opts {
        csv_path,
        outcome,
        regressors: positional.collect(),
        cluster,
    })
}

fn read_csv(opts: &Opts) -> Result<Data, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&opts.csv_path)
        .map_err(|e| format!("Error reading {}: {e}", opts.csv_path))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("Error reading {}: {e}", opts.csv_path))?
        .clone();

    let column = |name: &str, what: &str| {
        headers.iter().position(|h| h == name).ok_or_else(|| {
            let available: Vec<_> = headers.iter().collect();
            format!(
                "Error: {what}'{name}' not found. Available: {}",
                available.join(", ")
            )
        })
    };
    let outcome_idx = column(&opts.outcome, "")?;
    let cluster_idx = opts
        .cluster
        .as_deref()
        .map(|name| column(name, "cluster column "))
        .transpose()?;
    let regressor_idx = opts
        .regressors
        .iter()
        .map(|name| column(name, ""))
        .collect::<Result<Vec<_>, _>>()?;

    let mut data = Data {
        outcome: Vec::new(),
        regressors: vec![Vec::new(); regressor_idx.len()],
        clusters: cluster_idx.map(|_| Vec::new()),
    };
    for record in reader.records() {
        let record = record.map_err(|e| format!("Error reading CSV row: {e}"))?;
        let field = |idx: usize| {
            record
                .get(idx)
                .ok_or_else(|| format!("Error: row has no value for '{}'.", &headers[idx]))
        };

        if let (Some(idx), Some(labels)) = (cluster_idx, &mut data.clusters) {
            let label = record.get(idx).ok_or_else(|| {
                format!(
                    "Error: row has no value for cluster column '{}'.",
                    &headers[idx]
                )
            })?;
            labels.push(label.to_owned());
        }
        let number = |idx: usize| {
            let value = field(idx)?;
            value
                .parse::<f64>()
                .map_err(|_| format!("Error: non-numeric value '{value}' in '{}'.", &headers[idx]))
        };
        data.outcome.push(number(outcome_idx)?);
        for (column, &idx) in data.regressors.iter_mut().zip(&regressor_idx) {
            column.push(number(idx)?);
        }
    }

    if data.outcome.is_empty() {
        return Err("Error: no data rows in CSV.".into());
    }
    Ok(data)
}
