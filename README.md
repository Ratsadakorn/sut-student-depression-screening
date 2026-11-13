# SUT Student Depression Screening

Analysis notebooks for a retrospective study of first-year university students at Suranaree University of Technology. The primary outcome is clinically diagnosed major depressive disorder (MDD). The secondary outcomes are questionnaire-defined depression risk and suicide risk.

| Notebook | Analysis |
| --- | --- |
| [01_mdd_prediction.ipynb](01_mdd_prediction.ipynb) | MDD model development, 2024 held-out evaluation, 2025 temporal validation, calibration, and sensitivity analyses |
| [02_depression_and_suicide_risk_prediction.ipynb](02_depression_and_suicide_risk_prediction.ipynb) | Outcome-specific model development, validation, and recalibration for depression risk and suicide risk |

## Environment

The notebooks record Python 3.10.10. Open them in Jupyter or VS Code using the research Python environment. Dependencies referenced in the notebooks include NumPy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn, SciPy, statsmodels, IPython, PyCaret (primary notebook), and xicorpy (secondary exploratory cells). Use the package versions from the original analysis environment when reproducing the results.

## Data access and execution

Participant-level datasets are stored separately from this repository. Access is subject to the applicable institutional and research ethics approvals.

| Input expected by the notebooks | Role |
| --- | --- |
| `2567-2Q9Q8Q.csv` | 2024 cohort used for the development and held-out partitions |
| `2568-2Q9Q8Q.csv` | 2025 temporal validation cohort |

1. Obtain authorized access to the study files. Keep the files outside the Git repository and update the relevant `pd.read_csv` calls and path variables to point to the approved storage location. Review local output paths before execution.
2. Run the primary notebook in order using its own kernel. For the secondary-outcome models, start at **Part S0** and proceed through **Part S6**. Earlier exploratory cells in the secondary notebook reference three additional local CSV files (`2567-2Q9Q8Q-1.csv` through `2567-2Q9Q8Q-3.csv`). These files are needed only to rerun those exploratory cells.
3. Keep regenerated participant-level outputs in approved storage. Notebook cell outputs have been cleared in the copies prepared for this repository.

## Study approval

Suranaree University of Technology Research Ethics Committee, Project No. EC-67-205 (COA No. 184/2567).
