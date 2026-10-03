//! Golden tests pinning the command-line output of `justols`.
//!
//! Each case in `tests/golden/cases.txt` runs the binary and compares its
//! output with `tests/golden/<name>.tsv`. Labels must match exactly; numbers
//! must agree to a relative tolerance, since summation order may legitimately
//! change the last bits. Each case in `tests/golden/errors.txt` must fail with
//! exit code 1 and print exactly `tests/golden/err-<name>.txt` to stderr.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const REL_TOL: f64 = 1e-12;

fn run(args: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_justols"))
        .args(args.split_whitespace())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run justols")
}

fn cases(file: &str) -> Vec<(String, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(file);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (name, args) = line.split_once('\t').unwrap_or((line, ""));
            (name.to_owned(), args.to_owned())
        })
        .collect()
}

fn golden(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn numbers_match(expected: &str, actual: &str) -> bool {
    if expected == actual {
        return true;
    }
    let (Ok(e), Ok(a)) = (expected.parse::<f64>(), actual.parse::<f64>()) else {
        return false;
    };
    if e.is_nan() || a.is_nan() || e.is_infinite() || a.is_infinite() {
        return e.is_nan() && a.is_nan() || e.total_cmp(&a).is_eq();
    }
    (e - a).abs() <= REL_TOL * e.abs().max(a.abs())
}

#[test]
fn output_matches_golden_files() {
    for (name, args) in cases("cases.txt") {
        let output = run(&args);
        assert!(output.status.success(), "{name}: {output:?}");
        let actual = String::from_utf8(output.stdout).unwrap();
        let expected = golden(&format!("{name}.tsv"));

        let actual: Vec<_> = actual.lines().collect();
        let expected: Vec<_> = expected.lines().collect();
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name}: row count\n{actual:#?}"
        );
        for (a, e) in actual.iter().zip(&expected) {
            let a: Vec<_> = a.split('\t').collect();
            let e: Vec<_> = e.split('\t').collect();
            assert_eq!(a[0], e[0], "{name}: row label");
            assert_eq!(a.len(), e.len(), "{name}: field count in row {}", e[0]);
            for (x, y) in a[1..].iter().zip(&e[1..]) {
                assert!(
                    numbers_match(y, x),
                    "{name}: row {}: expected {y}, got {x}",
                    e[0]
                );
            }
        }
    }
}

#[test]
fn errors_match_golden_files() {
    for (name, args) in cases("errors.txt") {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(1), "{name}: exit code");
        assert!(output.stdout.is_empty(), "{name}: stdout should be empty");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(stderr, golden(&format!("err-{name}.txt")), "{name}: stderr");
    }
}
