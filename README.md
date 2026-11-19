# SUT Student Depression Screening

Analysis code for the retrospective study *Development and Temporal Validation of a Low-Burden Machine Learning Model for Screening Clinically Diagnosed Major Depressive Disorder Among University Students*.

The primary outcome is clinically diagnosed major depressive disorder (MDD). Secondary outcomes are binary questionnaire-defined depression risk and suicide risk.

## Files

| File | Purpose |
| --- | --- |
| `01_mdd_prediction.ipynb` | Develop and evaluate the seven screening strategies, including the primary M2 AdaBoost model, calibration, decision curve analysis, feature importance, subgroup analyses, and sensitivity analyses |
| `02_depression_and_suicide_risk_prediction.ipynb` | Compute exploratory associations and develop, validate, and recalibrate the outcome-specific M2 models for depression risk and suicide risk |
| `requirements.txt` | Supplied package-version snapshot captured from the Python environment |
| `03_cascade_frozen.py` | Evaluate M2 alone, M5 alone, and sequential M2-M5 screening from stored validation probabilities and development-locked thresholds |

Aggregate tables and embedded figures supplied with the notebooks are retained. Participant-level dataframe previews have been removed from both the saved outputs and the commands that displayed them. Data-loading paths remain visible.

## Environment

The supplied analysis environment uses Python 3.10.10 (64-bit Windows). Package versions are recorded in `requirements.txt`. Key versions are:

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| pandas | 2.1.4 |
| SciPy | 1.11.4 |
| scikit-learn | 1.4.2 |
| imbalanced-learn | 0.12.3 |
| PyCaret | 3.3.2 |
| matplotlib | 3.7.5 |
| seaborn | 0.13.2 |
| statsmodels | 0.14.2 |

Use a separate notebook kernel for each notebook. The cascade script requires NumPy, pandas, and scikit-learn. Corrected exploratory correlation cells use `scipy.stats.rankdata` and analytical averaging over tied predictor orderings.

Package versions were captured from the analysis environment. The package `package_name==0.1` was excluded from `requirements.txt` after the study author confirmed that it is not used by the analytical code. The original environment snapshot is retained separately for provenance. The requirements include `pywin32==311`, which is specific to Windows. Installation in a clean environment has not been independently tested.

## Input data

Participant-level datasets and prediction exports are stored separately from this repository. Access is subject to the applicable institutional and research ethics approvals. These inputs are needed for execution.

| File | Role |
| --- | --- |
| `2567-2Q9Q8Q.csv` | 2024 analytic cohort, 4,090 students and 46 recorded MDD cases |
| `2568-2Q9Q8Q.csv` | 2025 temporal validation cohort, 815 students and 16 recorded MDD cases |
| `2567-2Q9Q8Q-1.csv` | 2024 administrative variables and outcomes for the exploratory correlation matrices |
| `2567-2Q9Q8Q-2.csv` | 2024 behavioral variables and outcomes for the exploratory correlation matrices |
| `2567-2Q9Q8Q-3.csv` | 2024 questionnaire items and outcomes for the exploratory correlation matrices |

The notebook input paths are relative to the kernel working directory. Set that directory to an approved local analysis directory containing the inputs, or update the `pd.read_csv` calls and path variables to the authorized file locations. The three exploratory subsets must preserve the same observations, coding, and outcome values as the corresponding 2024 analytic data.

## Notebook execution

1. Run `01_mdd_prediction.ipynb` sequentially in its own kernel. The 2024 analytic cohort is split into development (2,863 students, 32 MDD cases) and held-out validation (1,227 students, 14 MDD cases). The 2025 cohort is used for temporal validation.
2. Run `02_depression_and_suicide_risk_prediction.ipynb` in a separate kernel. Its initial exploratory cells require the three additional subset CSVs. To run the secondary prediction models alone, start at **Part S0** and continue through **Part S6** and the final secondary prediction-export cell.
3. The corrected correlation cells compute original Chatterjee xi with tied-outcome ranks and analytically average over all orderings within tied predictor values. Matrix rows represent X and columns represent Y. Constant outcome columns produce NaN. Full-precision matrices remain in memory even when a display rounds to one decimal place.
4. Figure filenames, palettes, and display precision follow the supplied notebooks. Annotated heatmaps with a 0-1 color scale map negative estimates to the lowest color. The underlying estimates retain their signs.
5. Generated outputs use relative directories beneath the working directory. The two display variants of each exploratory heatmap use the same filename, so the later annotated variant replaces the earlier unannotated image.
6. The final prediction-export cells create participant-level files for local verification. Keep those exports in approved research storage. Review saved notebook outputs for participant-level data before future uploads.

## Frozen cascade analysis

The MDD notebook's final prediction-export cell writes:

```text
Supervisor_Export_M2_M5/
    M2_M5_all_validation_predictions.csv
    M2_M5_export_summary.csv
    M2_heldout2024_predictions.csv
    M2_temporal2025_predictions.csv
    M5_heldout2024_predictions.csv
    M5_temporal2025_predictions.csv
```

The combined CSV is the input to the cascade script. It must contain:

```text
participant_id, cohort, strategy, y_true,
predicted_probability, locked_threshold, predicted_class
```

Expected cohort labels are `Held-out 2024` and `Temporal 2025`. Strategy labels are `M2` and `M5`. Participant identifiers must pair the same observations across the two strategies within each cohort.

First run the primary analysis cells needed to populate `heldout_predictions`, `temporal_predictions`, and `locked_thresholds`, then run the cell headed `EXPORT PACKAGE FOR SUPERVISOR`. If those objects already exist in the active kernel from the completed analysis, run the export cell directly. Confirm `FINAL STATUS: PASS`. Verify that `original_index` identifies the same participants across models; matching outcomes alone is insufficient to establish identity.

Run from the directory containing the script:

```bash
python 03_cascade_frozen.py --predictions "/approved/storage/Supervisor_Export_M2_M5/M2_M5_all_validation_predictions.csv" --output-dir cascade_outputs --boot 2000 --seed 42
```

Replace the example input path with the real authorized path. If the combined CSV is in `Supervisor_Export_M2_M5` under the working directory, the default command is:

```bash
python 03_cascade_frozen.py
```

The script checks cohort sizes, event counts, unique participant records, paired outcomes, stored classifications, probability bounds, and unchanged thresholds across cohorts. It evaluates sequential screening by requiring positive classifications from both M2 and M5. It uses the same stratified participant bootstrap resample for all three rules within each cohort.

Aggregate outputs are:

- `cascade_frozen_results.csv` with confusion counts, performance estimates, and percentile bootstrap intervals
- `cascade_frozen_tradeoffs.csv` with false-positive reductions, true-positive losses, and paired differences
- `cascade_frozen_discrimination.csv` with AUC and average precision for each stored model
- `cascade_frozen_thresholds.csv` with the thresholds read from the export

## Verified frozen cascade results

The study author and supervising investigator verified the frozen model code and results. The results below follow their confirmed frozen-analysis report and correspond to Multimedia Appendix 2, Table S15. Earlier reconstruction outputs are superseded and are not part of the manuscript analysis.

M2 uses the administrative-plus-2Q strategy with AdaBoost; M5 uses the conventional administrative-plus-questionnaire strategy with Extra Trees. Classification uses the full-precision development-locked thresholds stored in the prediction exports (approximately 0.530354017385 for M2 and 0.645655950103 for M5). Rounded display values must not replace these stored thresholds during calculation.

| Validation cohort | Screening rule | TP | FP | FN | Screen-positive | Sensitivity (95% CI) | Precision (95% CI) | NNA (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Held-out 2024 | M2 alone | 7 | 84 | 7 | 91 | 0.500 (0.286-0.786) | 0.077 (0.040-0.118) | 13.0 (8.5-25.0) |
| Held-out 2024 | M5 alone | 9 | 41 | 5 | 50 | 0.643 (0.429-0.857) | 0.180 (0.113-0.262) | 5.6 (3.8-8.8) |
| Held-out 2024 | M2 then M5 | 6 | 32 | 8 | 38 | 0.429 (0.143-0.714) | 0.158 (0.071-0.258) | 6.3 (3.9-14.0) |
| Temporal 2025 | M2 alone | 10 | 38 | 6 | 48 | 0.625 (0.375-0.875) | 0.208 (0.132-0.297) | 4.8 (3.4-7.6) |
| Temporal 2025 | M5 alone | 16 | 11 | 0 | 27 | 1.000 (1.000-1.000) | 0.593 (0.471-0.762) | 1.7 (1.3-2.1) |
| Temporal 2025 | M2 then M5 | 10 | 7 | 6 | 17 | 0.625 (0.375-0.875) | 0.588 (0.409-0.812) | 1.7 (1.2-2.4) |

TP = true positives; FP = false positives; FN = false negatives; NNA = number needed to assess per confirmed MDD case, calculated as screen-positive / TP. Intervals are 95% percentile intervals from 2,000 stratified participant-level bootstrap resamples. The reported intervals are transcribed from the confirmed frozen-analysis report; individual bootstrap endpoints can vary with the random seed and implementation.

Compared with M2 alone, the cascade removed 52 of 84 false positives in held-out 2024 and 31 of 38 in temporal 2025, while losing one and zero true positives, respectively. Applying M5 to everyone identified three and six additional MDD cases compared with the cascade. Under the sequential screening scenario, the seven additional questionnaire items would be administered to the M2 screen-positive subgroup: 91/1,227 (7.4%) and 48/815 (5.9%). These figures describe a retrospective screening scenario rather than a prospectively implemented workflow.

This was an exploratory post hoc analysis based on 14 and 16 MDD events. Questionnaire items used by M5 may also contribute to the referral pathway leading to clinical diagnosis, creating a potential for incorporation bias. Prospective evaluation is required before clinical implementation.

## Reproducibility and model availability

The frozen cascade verification described above was conducted by the study author and supervising investigator using the study prediction exports. Participant-level datasets and prediction exports are not distributed with this code package. Reproduction requires authorized access to those inputs and the documented analysis environment. No independent end-to-end rerun was performed during README preparation.

Use the same cohort definitions, variable coding, development split, preprocessing, and threshold procedure as the final manuscript. Input checks in the cascade script verify file consistency; they do not validate source-data coding.

The cascade script evaluates existing predictions without refitting models or selecting thresholds. Its reference counts are documented here for comparison and are not hard-coded into its calculations. Fitted model objects and serialized preprocessing pipelines are not included in this package; the notebooks provide the analytical code.

## Study approval

Suranaree University of Technology Research Ethics Committee, Project No. EC-67-205 (COA No. 184/2567).
