//! The `justols` command: OLS on a CSV file, printed as tab-separated rows.

use std::io::{self, Write};
use std::process::ExitCode;

use csv::ReaderBuilder;
use justols::Ols;

const USAGE: &str = "\
Usage: justols [flags] <data.csv> <outcome> [x1] [x2] [factor:reference] ...
Flags:
  --cluster <column>    Compute one-way clustered HC3 standard errors.
  --time                Print the time taken to fit the model to stderr.";

/// Evaluates `$code`, printing the elapsed time to stderr if `$enabled`.
macro_rules! time_it {
    ($enabled:expr, $code:expr) => {{
        let start = std::time::Instant::now();
        let result = $code;
        if $enabled {
            eprintln!("Elapsed: {:?}", start.elapsed());
        }
        result
    }};
}

struct Opts {
    csv_path: String,
    outcome: String,
    regressors: Vec<Regressor>,
    cluster: Option<String>,
    time: bool,
}

/// A regressor argument: `column`, or `column:reference` for a factor.
enum Regressor {
    Numeric(String),
    Factor { column: String, reference: String },
}

impl Regressor {
    fn parse(arg: String) -> Self {
        match arg.split_once(':') {
            Some((column, reference)) => Self::Factor {
                column: column.to_owned(),
                reference: reference.to_owned(),
            },
            None => Self::Numeric(arg),
        }
    }

    fn column(&self) -> &str {
        match self {
            Self::Numeric(column) | Self::Factor { column, .. } => column,
        }
    }
}

/// The values of a regressor column.
enum Values {
    Numeric(Vec<f64>),
    Labels(Vec<String>),
}

/// The columns of the CSV file used by the model.
struct Data {
    outcome: Vec<f64>,
    regressors: Vec<Values>,
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

    let mut model = Ols::new(&data.outcome);
    for (regressor, values) in opts.regressors.iter().zip(&data.regressors) {
        model = match (regressor, values) {
            (Regressor::Numeric(name), Values::Numeric(values)) => model.regressor(name, values),
            (Regressor::Factor { column, reference }, Values::Labels(labels)) => {
                model.factor(column, labels, reference)
            }
            _ => unreachable!("read_csv reads factors as labels"),
        };
    }
    if let Some(labels) = &data.clusters {
        model = model.cluster(labels);
    }
    let fit = time_it!(opts.time, model.fit().map_err(|e| format!("Error: {e}."))?);

    write!(io::stdout().lock(), "{fit}").map_err(|e| format!("Error writing output: {e}"))
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    const MISSING_CLUSTER: &str = "Error: --cluster requires a column name.";

    let mut args = args.peekable();
    let mut positional = Vec::new();
    let mut cluster = None;
    let mut time = false;
    while let Some(arg) = args.next() {
        if arg == "--cluster" {
            cluster = Some(args.next().ok_or(MISSING_CLUSTER)?);
        } else if let Some(value) = arg.strip_prefix("--cluster=") {
            if value.is_empty() {
                return Err(MISSING_CLUSTER.into());
            }
            cluster = Some(value.to_owned());
        } else if arg == "--time" {
            time = true;
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
        regressors: positional.map(Regressor::parse).collect(),
        cluster,
        time,
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
    if let Some(bad) = headers.iter().find(|h| h.contains(':')) {
        return Err(format!(
            "Error: column name '{bad}' contains ':', which is reserved for factors."
        ));
    }

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
        .map(|regressor| column(regressor.column(), ""))
        .collect::<Result<Vec<_>, _>>()?;

    let mut data = Data {
        outcome: Vec::new(),
        regressors: opts
            .regressors
            .iter()
            .map(|regressor| match regressor {
                Regressor::Numeric(_) => Values::Numeric(Vec::new()),
                Regressor::Factor { .. } => Values::Labels(Vec::new()),
            })
            .collect(),
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
        let number = |idx: usize, regressor: bool| {
            let value = field(idx)?;
            value.parse::<f64>().map_err(|_| {
                let name = &headers[idx];
                let mut message = format!("Error: non-numeric value '{value}' in '{name}'.");
                if regressor {
                    message.push_str(" For factors, use '");
                    message.push_str(name);
                    message.push_str(":<reference>'.");
                }
                message
            })
        };
        data.outcome.push(number(outcome_idx, false)?);
        for (values, &idx) in data.regressors.iter_mut().zip(&regressor_idx) {
            match values {
                Values::Numeric(values) => values.push(number(idx, true)?),
                Values::Labels(labels) => labels.push(field(idx)?.to_owned()),
            }
        }
    }

    if data.outcome.is_empty() {
        return Err("Error: no data rows in CSV.".into());
    }
    Ok(data)
}
