"""Evaluate sequential M2-M5 screening from frozen prediction exports.

Input
-----
Supervisor_Export_M2_M5/M2_M5_all_validation_predictions.csv,
created by the MDD notebook's final prediction-export cell.

The input contains participant-level data and should remain in approved
research storage. This script writes aggregate results.

Usage
-----
python 03_cascade_frozen.py --predictions /approved/storage/Supervisor_Export_M2_M5/M2_M5_all_validation_predictions.csv --output-dir cascade_outputs --boot 2000

The script evaluates stored probabilities at stored development-locked
thresholds. It does not train classifiers or select thresholds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EXPECTED_COHORTS = {
    "Held-out 2024": (1227, 14),
    "Temporal 2025": (815, 16),
}
METRICS = (
    "sensitivity", "specificity", "precision", "F1", "NNA",
    "TP", "FP", "FN", "TN",
)
REQUIRED_COLUMNS = (
    "participant_id", "cohort", "strategy", "y_true",
    "predicted_probability", "locked_threshold", "predicted_class",
)


def confusion_metrics(y, positive):
    """Calculate confusion counts and threshold-based performance."""
    y = np.asarray(y, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    tp = int(np.sum(positive & y))
    fp = int(np.sum(positive & ~y))
    fn = int(np.sum(~positive & y))
    tn = int(np.sum(~positive & ~y))

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "flagged": tp + fp,
        "NNA": (tp + fp) / tp if tp else np.nan,
    }


def load_predictions(path):
    """Validate the export and retain participant-level data in memory."""
    data = pd.read_csv(path, dtype={"participant_id": "string"})
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required prediction fields contain missing values.")

    if set(data["cohort"]) != set(EXPECTED_COHORTS):
        raise ValueError("Expected Held-out 2024 and Temporal 2025 cohorts.")

    if set(data["strategy"]) != {"M2", "M5"}:
        raise ValueError("Expected exactly the M2 and M5 strategies.")

    if data.duplicated(["cohort", "strategy", "participant_id"]).any():
        raise ValueError("Duplicate participant records within a strategy/cohort.")

    for name in ("y_true", "predicted_class"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not data[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be coded as 0 or 1.")

    for name in ("predicted_probability", "locked_threshold"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        values = data[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f"{name} must contain finite values in [0, 1].")

    for strategy in ("M2", "M5"):
        thresholds = data.loc[
            data["strategy"] == strategy, "locked_threshold"
        ].unique()
        if len(thresholds) != 1:
            raise ValueError(
                f"{strategy}: expected one unchanged threshold across cohorts."
            )

    calculated_classes = (
        data["predicted_probability"] >= data["locked_threshold"]
    ).astype(int)
    if not np.array_equal(calculated_classes, data["predicted_class"]):
        raise ValueError("Stored classes disagree with probabilities and thresholds.")

    for cohort, (expected_n, expected_events) in EXPECTED_COHORTS.items():
        subset = data.loc[data["cohort"] == cohort]
        for strategy in ("M2", "M5"):
            group = subset.loc[subset["strategy"] == strategy]
            if len(group) != expected_n or group["y_true"].sum() != expected_events:
                raise ValueError(f"{cohort}/{strategy}: unexpected cohort counts.")

        m2 = subset.loc[subset["strategy"] == "M2"].set_index("participant_id")
        m5 = subset.loc[subset["strategy"] == "M5"].set_index("participant_id")
        if set(m2.index) != set(m5.index):
            raise ValueError(f"{cohort}: unmatched M2 and M5 participants.")
        if not np.array_equal(m2["y_true"], m5.loc[m2.index, "y_true"]):
            raise ValueError(f"{cohort}: outcome mismatch after ID alignment.")

    return data


def percentile_interval(values):
    """Return percentile limits, retaining undefined metrics as NaN."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([np.nan, np.nan])
    return np.quantile(finite, [0.025, 0.975])


def evaluate_predictions(data, n_bootstrap=2000, seed=42):
    """Use paired stratified participant resamples within each cohort."""
    results = []
    tradeoffs = []
    discrimination = []
    thresholds = []

    for strategy in ("M2", "M5"):
        threshold = float(
            data.loc[data["strategy"] == strategy, "locked_threshold"].iloc[0]
        )
        thresholds.append({"Strategy": strategy, "Locked_Threshold": threshold})

    for cohort in EXPECTED_COHORTS:
        subset = data.loc[data["cohort"] == cohort]
        m2 = subset.loc[subset["strategy"] == "M2"].set_index("participant_id")
        m5 = subset.loc[subset["strategy"] == "M5"].set_index("participant_id")
        m5 = m5.loc[m2.index]

        y = m2["y_true"].to_numpy(dtype=int)
        positive_m2 = m2["predicted_class"].to_numpy(dtype=bool)
        positive_m5 = m5["predicted_class"].to_numpy(dtype=bool)
        rules = {
            "M2 alone": positive_m2,
            "M5 alone": positive_m5,
            "M2 then M5": positive_m2 & positive_m5,
        }
        point = {
            name: confusion_metrics(y, positive)
            for name, positive in rules.items()
        }
        bootstrap_values = {
            name: {metric: [] for metric in METRICS}
            for name in rules
        }
        paired_tp = []
        paired_fp = []
        cases = np.flatnonzero(y == 1)
        noncases = np.flatnonzero(y == 0)
        rng = np.random.default_rng(seed)

        for _ in range(n_bootstrap):
            indices = np.concatenate([
                rng.choice(cases, size=len(cases), replace=True),
                rng.choice(noncases, size=len(noncases), replace=True),
            ])
            sampled = {}
            for name, positive in rules.items():
                sampled[name] = confusion_metrics(y[indices], positive[indices])
                for metric in METRICS:
                    bootstrap_values[name][metric].append(sampled[name][metric])
            paired_tp.append(sampled["M2 then M5"]["TP"] - sampled["M2 alone"]["TP"])
            paired_fp.append(sampled["M2 then M5"]["FP"] - sampled["M2 alone"]["FP"])

        for name, metrics in point.items():
            row = {
                "Cohort": cohort,
                "Rule": name,
                "N": len(y),
                "Events": int(y.sum()),
                "Additional_Questionnaire_N": (
                    int(positive_m2.sum()) if name == "M2 then M5"
                    else len(y) if name == "M5 alone" else 0
                ),
                **metrics,
            }
            for metric in METRICS:
                low, high = percentile_interval(bootstrap_values[name][metric])
                row[f"{metric}_CI_Lower"] = low
                row[f"{metric}_CI_Upper"] = high
            results.append(row)

        tp_low, tp_high = percentile_interval(paired_tp)
        fp_low, fp_high = percentile_interval(paired_fp)
        tradeoffs.append({
            "Cohort": cohort,
            "M2_Screen_Positive_N": int(positive_m2.sum()),
            "M2_Screen_Positive_Percent": 100 * positive_m2.mean(),
            "False_Positives_Removed": point["M2 alone"]["FP"] - point["M2 then M5"]["FP"],
            "True_Positives_Removed": point["M2 alone"]["TP"] - point["M2 then M5"]["TP"],
            "Additional_Cases_M5_Alone_vs_Cascade": (
                point["M5 alone"]["TP"] - point["M2 then M5"]["TP"]
            ),
            "Cascade_Minus_M2_TP_CI_Lower": tp_low,
            "Cascade_Minus_M2_TP_CI_Upper": tp_high,
            "Cascade_Minus_M2_FP_CI_Lower": fp_low,
            "Cascade_Minus_M2_FP_CI_Upper": fp_high,
        })
        for name, group in (("M2", m2), ("M5", m5)):
            probabilities = group["predicted_probability"].to_numpy()
            discrimination.append({
                "Cohort": cohort,
                "Strategy": name,
                "N": len(y),
                "Events": int(y.sum()),
                "AUC": roc_auc_score(y, probabilities),
                "Average_Precision": average_precision_score(y, probabilities),
            })

    return (
        pd.DataFrame(results),
        pd.DataFrame(tradeoffs),
        pd.DataFrame(discrimination),
        pd.DataFrame(thresholds),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("Supervisor_Export_M2_M5/M2_M5_all_validation_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("cascade_outputs"))
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.boot < 1:
        parser.error("--boot must be at least 1.")

    data = load_predictions(args.predictions)
    tables = evaluate_predictions(data, n_bootstrap=args.boot, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "cascade_frozen_results.csv",
        "cascade_frozen_tradeoffs.csv",
        "cascade_frozen_discrimination.csv",
        "cascade_frozen_thresholds.csv",
    )
    for table, filename in zip(tables, filenames):
        table.to_csv(args.output_dir / filename, index=False)

    print("Frozen prediction export validated for both cohorts and strategies.")
    print(f"Bootstrap resamples per cohort: {args.boot}; seed: {args.seed}")
    summary_columns = [
        "Cohort", "Rule", "N", "Events", "TP", "FP", "FN", "TN",
        "sensitivity", "specificity", "precision", "F1", "flagged", "NNA",
    ]
    print(tables[0][summary_columns].round(3).to_string(index=False))
    print("\nCascade tradeoffs:")
    print(tables[1].round(3).to_string(index=False))
    print("\nAggregate output files:")
    for filename in filenames:
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
