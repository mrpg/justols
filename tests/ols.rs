//! Library tests. Reference values come from R (`lm` and `sandwich::vcovHC`
//! with `type = "HC3"`) on `data/test.csv`; see `data/test.R`.

#![allow(clippy::float_cmp, reason = "exact results are expected here")]

use justols::{Error, Fit, Ols, Term};

struct TestData {
    outcome: Vec<f64>,
    x: [Vec<f64>; 3],
}

fn test_data() -> TestData {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/test.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut data = TestData {
        outcome: Vec::new(),
        x: Default::default(),
    };
    for line in text.lines().skip(1) {
        let row: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
        data.outcome.push(row[0]);
        for (column, value) in data.x.iter_mut().zip(&row[1..]) {
            column.push(*value);
        }
    }
    data
}

fn fit_full(data: &TestData, clusters: Option<Vec<usize>>) -> Fit {
    let model = Ols::new(&data.outcome)
        .regressor("x1", &data.x[0])
        .regressor("x2", &data.x[1])
        .regressor("x3", &data.x[2]);
    match clusters {
        Some(labels) => model.cluster(labels),
        None => model,
    }
    .fit()
    .unwrap()
}

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    let tol = 1e-10 * expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn hc3_matches_r() {
    let fit = fit_full(&test_data(), None);

    let beta = [
        4.706_252_677_160_387,
        1.883_006_909_500_295,
        -0.498_929_175_993_411,
        2.968_591_453_116_398,
    ];
    let se = [
        0.308_167_624_613_760_7,
        0.135_552_102_441_579_1,
        0.046_418_433_390_153_69,
        0.269_392_052_104_819_4,
    ];
    for ((c, beta), se) in fit.coefficients().iter().zip(beta).zip(se) {
        assert_close(c.estimate, beta);
        assert_close(c.std_error.unwrap(), se);
        assert_close(c.t_stat.unwrap(), beta / se);
    }

    assert_close(fit.r_squared(), 0.688_292_564_606_337_9);
    assert_close(fit.adj_r_squared().unwrap(), 0.683_521_532_431_945_1);
    assert_close(fit.loo_r_squared().unwrap(), 0.675_434_335_679_513_8);
    assert_close(fit.press().unwrap(), 712.699_912_160_771_5);
    assert_close(fit.residual_se().unwrap(), 1.868_734_531_933_053);
    assert_close(fit.max_leverage(), 0.100_805_584_522_098_3);
    assert_close(fit.condition_number(), 10.635_715_678_758_856);

    let wald = fit.wald_test().unwrap();
    assert_close(wald.f_stat, 135.239_832_312_122_8);
    assert_eq!((wald.df_num, wald.df_denom), (3, 196));

    assert_eq!(fit.n_observations(), 200);
    assert_eq!(fit.df_model(), 3);
    assert_eq!(fit.df_resid(), 196);
    assert_eq!(fit.inference_df(), 196);
    assert_eq!(fit.n_clusters(), None);
}

#[test]
fn clustered_hc3_matches_r() {
    let data = test_data();
    let fit = fit_full(&data, Some((0..200).map(|i| i % 20).collect()));

    let se = [
        0.283_021_665_991_241_9,
        0.130_061_884_393_229_2,
        0.044_238_217_029_359_03,
        0.207_909_462_620_058_47,
    ];
    let p = [
        8.863_238_543_472_992e-13,
        1.024_856_624_226_458_1e-11,
        7.332_250_666_570_265e-10,
        1.306_078_931_390_567_8e-11,
    ];
    for ((c, se), p) in fit.coefficients().iter().zip(se).zip(p) {
        assert_close(c.std_error.unwrap(), se);
        // 1 - cdf loses digits below machine epsilon.
        assert!((c.p_value.unwrap() - p).abs() < 1e-14);
    }

    let wald = fit.wald_test().unwrap();
    assert_close(wald.f_stat, 217.249_688_275_255_08);
    assert_eq!((wald.df_num, wald.df_denom), (3, 19));
    assert_eq!(fit.n_clusters(), Some(20));
    assert_eq!(fit.inference_df(), 19);
}

#[test]
fn cluster_labels_can_be_any_hashable_type() {
    let data = test_data();
    let by_int = fit_full(&data, Some((0..200).map(|i| i % 20).collect()));
    let labels: Vec<String> = (0..200).map(|i| format!("firm {}", i % 20)).collect();
    let by_str = Ols::new(&data.outcome)
        .regressors([
            ("x1", &data.x[0][..]),
            ("x2", &data.x[1]),
            ("x3", &data.x[2]),
        ])
        .cluster(&labels)
        .fit()
        .unwrap();
    assert_eq!(by_int.coefficients(), by_str.coefficients());
}

#[test]
fn coefficients_are_named() {
    let fit = fit_full(&test_data(), None);
    assert_eq!(fit.intercept().term, Term::Intercept);
    assert_eq!(fit.intercept().term.to_string(), "!Intercept");
    assert_eq!(fit.coefficient("x2").unwrap().term.name(), Some("x2"));
    assert!(fit.coefficient("x4").is_none());
    let names: Vec<_> = fit
        .coefficients()
        .iter()
        .map(|c| c.term.to_string())
        .collect();
    assert_eq!(names, ["!Intercept", "x1", "x2", "x3"]);
}

#[test]
fn intercept_only() {
    let y = [1.0, 2.0, 4.0, 8.0];
    let fit = Ols::new(&y).fit().unwrap();
    assert_close(fit.intercept().estimate, 3.75);
    assert_eq!(fit.r_squared(), 0.0);
    assert_eq!(fit.df_model(), 0);
    assert!(fit.wald_test().is_none());
    assert!(fit.intercept().p_value.is_some());
}

#[test]
fn residuals_and_fitted_values_add_up() {
    let data = test_data();
    let fit = fit_full(&data, None);
    for ((y, fitted), e) in data.outcome.iter().zip(fit.fitted()).zip(fit.residuals()) {
        assert!((fitted + e - y).abs() < 1e-12);
    }
    let trace: f64 = fit.leverages().iter().sum();
    assert_close(trace, 4.0);
}

#[test]
fn saturated_model_has_no_inference() {
    let fit = Ols::new(&[1.0, 3.0])
        .regressor("x", &[0.0, 1.0])
        .fit()
        .unwrap();
    assert_close(fit.coefficient("x").unwrap().estimate, 2.0);
    assert_eq!(fit.df_resid(), 0);
    for c in fit.coefficients() {
        assert_eq!((c.std_error, c.t_stat, c.p_value), (None, None, None));
    }
    assert_eq!(fit.adj_r_squared(), None);
    assert_eq!(fit.loo_r_squared(), None);
    assert_eq!(fit.press(), None);
    assert_eq!(fit.residual_se(), None);
    assert!(fit.wald_test().is_none());
}

#[test]
fn single_cluster_has_no_inference() {
    let y = [1.0, 3.0, 2.0, 5.0];
    let x = [1.0, 2.0, 3.0, 4.0];
    let fit = Ols::new(&y)
        .regressor("x", &x)
        .cluster([0; 4])
        .fit()
        .unwrap();
    assert_eq!(fit.n_clusters(), Some(1));
    assert_eq!(fit.inference_df(), 0);
    assert!(fit.coefficients().iter().all(|c| c.std_error.is_none()));
    assert!(fit.press().is_some());
}

#[test]
fn exact_zero_fit_does_not_panic() {
    let fit = Ols::new(&[0.0; 4])
        .regressor("x", &[0.0, 1.0, 2.0, 3.0])
        .fit()
        .unwrap();
    assert!(fit.coefficient("x").unwrap().p_value.unwrap().is_nan());
}

#[test]
fn errors() {
    let y = [1.0, 2.0, 3.0];
    let x = [1.0, 0.0, 1.0];

    let short = Ols::new(&y).regressor("x", &x[..2]).fit().unwrap_err();
    assert_eq!(
        short,
        Error::LengthMismatch {
            name: "x".into(),
            expected: 3,
            found: 2
        }
    );

    let clusters = Ols::new(&y).cluster([1, 2]).fit().unwrap_err();
    assert_eq!(
        clusters,
        Error::ClusterLengthMismatch {
            expected: 3,
            found: 2
        }
    );

    let nan = Ols::new(&[1.0, f64::NAN, 3.0]).fit().unwrap_err();
    assert_eq!(nan, Error::NonFiniteOutcome { row: 1 });

    let inf = Ols::new(&y)
        .regressor("x", &[1.0, 2.0, f64::INFINITY])
        .fit()
        .unwrap_err();
    assert_eq!(
        inf,
        Error::NonFiniteRegressor {
            name: "x".into(),
            row: 2
        }
    );

    let few = Ols::new(&y)
        .regressors([("a", &x[..]), ("b", &x), ("c", &x)])
        .fit()
        .unwrap_err();
    assert_eq!(
        few,
        Error::TooFewObservations {
            observations: 3,
            parameters: 4
        }
    );

    let singular = Ols::new(&y)
        .regressor("a", &x)
        .regressor("b", &x)
        .fit()
        .unwrap_err();
    assert_eq!(singular, Error::Singular);
    assert_eq!(singular.to_string(), "X'X is singular");
}

#[test]
fn models_are_reusable() {
    let y = [1.0, 3.0, 2.0, 5.0];
    let x = [1.0, 2.0, 3.0, 4.0];
    let model = Ols::new(&y).regressor("x", &x);
    let a = model.fit().unwrap();
    let b = model.clone().cluster([0, 0, 1, 1]).fit().unwrap();
    assert_eq!(a.coefficients()[1].estimate, b.coefficients()[1].estimate);
    assert_ne!(a.coefficients()[1].std_error, b.coefficients()[1].std_error);
}
