# Independent Model Validation Pack for Credit Risk (PD) with Proxy LGD/EAD/EL

Reproducibility, out-of-time validation, calibration, drift monitoring, robustness testing, and challenger benchmarking for a credit risk PD model using LendingClub loan data.

This repository contains an end-to-end **independent model validation** project for a **Probability of Default (PD)** model, with an extension to **proxy LGD/EAD/Expected Loss (EL)** for directional stress analysis.

The project is framed as a **model risk validation** exercise from a validator perspective, not as a pure model development project. The focus is on whether the model remains usable and governable under temporal change.

## Quick access

If you want the final deliverable first, open the report directly:

- [Model Validation Report (PDF)](reports/Model_Validation_Report.pdf)

## What this project does

The validation pack covers:

- **Champion PD model reproduction** using logistic regression
- **Feature governance and leakage prevention** using underwriting-time variables only
- **True out-of-time (OOT) validation** using a temporal split
- **Performance assessment** (ranking and classification diagnostics)
- **Calibration assessment** (reliability and calibration diagnostics)
- **Drift and stability monitoring** (PSI and score distribution shift)
- **Sensitivity analysis** with controlled input shocks
- **Stress scenario analysis** with PD and EL proxy impact
- **Challenger benchmark** using LightGBM with SHAP interpretability
- **Report-ready outputs** (figures, CSV tables, LaTeX tables, final PDF report)

## Project objective

The goal is to produce an **Independent Model Validation Report** that clearly demonstrates:

- independent validation of a **champion PD model** (logistic regression),
- explicit **leakage control** (underwriting-time features only),
- **OOT validation** with a time-based split,
- evaluation of **performance, calibration, and drift**,
- **sensitivity and stress testing**,
- benchmark against a **challenger (LightGBM)** with interpretability evidence,
- and an extended **PD × LGD × EAD proxy = EL proxy** view for directional portfolio risk analysis.

The final deliverable is intended to read like a validation report with governance conclusions and remediation recommendations, not a notebook-based model demo.

## Dataset

### Source
- LendingClub Loan Data ([Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv))

This dataset supports:

- **PD target construction** from `loan_status`
- **LGD proxy** using `recoveries`
- **EAD proxy** using `funded_amnt` and, when available, `total_rec_prncp`

It also includes many leakage-prone post-outcome variables, which makes it useful for demonstrating feature governance and leakage controls.

### Typical fields used (depending on dataset version)

Core fields:
- `issue_d`
- `loan_status`
- `funded_amnt`
- `recoveries`
- `total_rec_prncp` (if available)

Typical underwriting variables:
- `annual_inc`
- `dti`
- `int_rate`
- `term`
- `grade` / `sub_grade`
- `emp_length`
- `home_ownership`
- `purpose`
- `verification_status`
- `delinq_2yrs`
- `inq_last_6mths`
- `open_acc`
- `total_acc`
- `revol_util`
- `revol_bal`
- `pub_rec`
- `earliest_cr_line`

## Validation design and methodology

### 1) PD target definition (closed outcomes only)

To avoid unresolved labels and target contamination, the PD target is defined on **closed outcomes only**:

- `default_flag = 1` for:
  - `Charged Off`
  - `Default`
- `default_flag = 0` for:
  - `Fully Paid`

Excluded statuses include unresolved or intermediate states such as:
- `Current`
- `Late`
- `In Grace Period`
- other non-final outcomes

This supports observed default discrimination and calibration analysis on resolved loans.

### 2) Out-of-time (OOT) validation design

A true temporal holdout is implemented using `issue_d` (origination date):

- **Train**: vintages up to **2017**
- **OOT**: **2018** vintage

This design is used instead of a random split in order to evaluate model behavior under temporal distribution change.

### 3) Feature governance and leakage prevention

A key requirement of the validation is proving that the models only use information available at underwriting time.

#### Allowed features
Only **underwriting-time variables** are used for the PD champion and challenger.

#### Excluded leakage-prone variables
Post-origination and post-outcome variables are excluded, including columns related to:
- payments (`pymnt`, `last_pymnt`, `total_rec_...`)
- recoveries (`recover...`)
- outstanding principal (`out_prncp...`)
- settlements
- hardship
- servicing and collection outcomes

#### Controls implemented
- whitelist-based feature policy for model inputs
- automated leakage-candidate scan using column-name patterns

This is a core part of the validation, since performance driven by post-outcome fields would invalidate the underwriting PD use case.

### 4) Champion model reproduction (logistic regression)

The champion is implemented as a reproducible **logistic regression pipeline** with explicit preprocessing:

- numeric preprocessing:
  - train-based median imputation
  - standardization
- categorical preprocessing:
  - imputation
  - one-hot encoding with unknown-category handling
- missingness governance:
  - features with high missingness are excluded under a documented threshold policy

The aim is to reproduce a realistic, auditable bank-style baseline under validation constraints.

### 5) Validation suite (performance, calibration, drift, robustness)

The project includes a validation suite that goes beyond ranking metrics.

#### Performance / discrimination
Representative outputs include:
- AUC / Gini-style summaries
- KS-related summaries
- PR-AUC and other ranking diagnostics (as exported in report artifacts)

#### Calibration
Calibration is treated as a first-class validation topic:
- reliability curves (Train and OOT)
- calibration diagnostics exported in summary tables
- interpretation of OOT over-prediction / under-prediction behavior

This matters because a model can remain useful for ranking while becoming miscalibrated out of time.

#### Drift and stability
Temporal stability is evaluated using:
- **PSI by feature**
- **PSI on predicted score / PD**
- score distribution shift (KS-based diagnostic)

This supports a monitoring-oriented interpretation with stable, moderate-drift, and material-drift thresholds.

### 6) Sensitivity analysis and stress testing

#### Sensitivity analysis
Controlled perturbations are applied to selected key drivers (for example income, DTI, utilization-type variables depending on final feature availability):

- `±5%`
- `±10%`

For each perturbation, the project evaluates:
- change in mean predicted PD
- change in tail PD (such as 95th percentile)
- **Spearman rank correlation** versus the baseline ranking

This separates probability-level sensitivity from rank-order robustness.

#### Stress scenarios
Stylized **mild** and **severe** scenarios are applied to OOT observations using directional shocks such as:
- lower income
- higher DTI
- higher utilization (scenario dependent)

The project evaluates:
- PD shift (mean and tail)
- **EL proxy shift** using PD, LGD proxy, and EAD proxy

This provides a directional vulnerability analysis at portfolio level.

### 7) Challenger benchmark (LightGBM + SHAP)

A **LightGBM challenger** is trained using the **same governance-approved feature set** as the champion.

The comparison is designed to answer a validation question:
- does a more flexible model improve OOT performance, calibration, and stability without relying on suspicious behavior?

The challenger is evaluated on:
- OOT discrimination
- calibration characteristics
- score stability / drift
- interpretability using SHAP (global bar plot)

SHAP is used as a sanity check to verify that major drivers are plausible and consistent with the feature policy.

### 8) Proxy LGD, EAD, and Expected Loss extension

To extend the analysis beyond PD, the project includes a lightweight proxy loss-stack module.

#### EAD proxy
- preferred (if available): `funded_amnt - total_rec_prncp`
- fallback: `funded_amnt`

#### LGD proxy (defaults only)
A recoveries-based proxy, clipped to the `[0, 1]` range.

#### Expected Loss proxy
Computed directionally as:

`EL_proxy = PD_pred × LGD_avg × EAD_proxy`

#### Important limitation
This module is intended for:
- directional stress and vulnerability analysis

It is not intended as:
- a production LGD/EAD framework
- a regulatory capital model
- an IFRS9 production implementation

## Main findings (current project narrative)

The exact values depend on the current dataset version and generated artifacts, but the report narrative is based on the following observed patterns.

### Champion (logistic regression)
- **Reasonable OOT discrimination**, so it remains useful as a ranking model
- **OOT calibration deterioration**
  - predicted PD tends to be above observed OOT default rate
  - calibration diagnostics worsen versus Train
- This is interpreted as **population drift and base-rate shift**, not immediate model failure

### Drift
- PSI identifies **material drift** in key underwriting variables, especially:
  - `revol_util`
- Additional drift appears in variables such as:
  - `application_type`
  - `int_rate`
- These shifts are plausible in terms of portfolio mix, pricing, and credit-cycle conditions

### Sensitivity
- Ranking remains highly stable under tested shocks
- Spearman rank correlation remains close to 1
- `dti` appears as a key sensitivity-relevant variable in mean and tail PD impact

### Stress and EL proxy
- Baseline < Mild < Severe in both PD and EL proxy
- The monotonic increase is qualitatively coherent and suitable for directional risk analysis
- LGD proxy levels tend to be high, which is consistent with low recoveries on unsecured charged-off consumer loans

### Challenger (LightGBM)
- Improved OOT performance and calibration characteristics
- Lower score drift and stronger stability evidence
- SHAP drivers are plausible and do not show leakage-type variables

### Validation conclusion in the report
- **Champion**: Moderate Model Risk
- **Challenger**: credible promotion / redevelopment candidate, subject to governance and monitoring controls

## Repository structure

```text
.
├── README.md
├── data
│   ├── interim
│   ├── processed
│   └── raw
│       ├── LCDataDictionary.xlsx
│       └── loan.csv
├── notebooks
│   ├── 00_data_audit.ipynb
│   ├── 01_champion_reproduction.ipynb
│   ├── 02_validation_suite.ipynb
│   ├── 03_challenger_model.ipynb
│   └── 04_report_figures.ipynb
├── reports
│   ├── Model_Validation_Report.tex
│   ├── Model_Validation_Report.pdf
│   ├── figures/
│   └── tables/
├── scripts
│   ├── 00_download_data_kagglehub.py
│   ├── 01_make_dataset.py
│   ├── 02_train_models.py
│   ├── 03_run_validation.py
│   ├── 04_run_challenger.py
│   ├── 05_generate_report_artifacts.py
│   └── 06_export_tables_to_latex.py
└── src
    ├── config.py
    ├── data/
    ├── models/
    ├── reporting/
    ├── utils/
    └── validation/
````

## Pipeline overview (scripts)

### `scripts/00_download_data_kagglehub.py`

Dataset download helper (if using the KaggleHub flow).

### `scripts/01_make_dataset.py`

Builds the modeling dataset:

* cleaning
* target definition
* leakage screening
* feature engineering
* temporal split preparation

### `scripts/02_train_models.py`

Trains the **champion logistic regression** model and stores predictions / model artifacts.

### `scripts/03_run_validation.py`

Runs the validation suite on champion outputs:

* performance metrics
* calibration diagnostics
* drift and stability diagnostics
* sensitivity analysis
* stress analysis
* proxy LGD/EAD/EL summaries

### `scripts/04_run_challenger.py`

Trains and evaluates the **LightGBM challenger** and generates challenger comparison outputs and SHAP artifacts.

### `scripts/05_generate_report_artifacts.py`

Generates report-ready artifacts such as:

* performance and calibration summary tables (Train / OOT)
* champion vs challenger OOT comparison table
* score stability comparison table
* PSI table
* sensitivity table
* stress table
* LGD-by-grade summary table
* figures (reliability curves, score shift histogram, PSI drivers, SHAP bar plot)

### `scripts/06_export_tables_to_latex.py`

Converts CSV tables into LaTeX table artifacts:

* splits wide tables into parts (`*_part1.tex`, `*_part2.tex`, ...)
* generates wrapper files (`*_table.tex`) for clean inclusion in the main report
* supports A4-friendly PDF layout without dropping columns

## How to run (end to end)

Assumes the Python environment is set up and the dataset is available in `data/raw/`.

### 1) Build the dataset

```bash
python scripts/01_make_dataset.py
```

### 2) Train the champion model

```bash
python scripts/02_train_models.py
```

### 3) Run the validation suite

```bash
python scripts/03_run_validation.py
```

### 4) Train and evaluate the challenger

```bash
python scripts/04_run_challenger.py
```

### 5) Generate report artifacts (CSV tables and figures)

```bash
python scripts/05_generate_report_artifacts.py
```

### 6) Export CSV tables to LaTeX wrappers and parts

```bash
python scripts/06_export_tables_to_latex.py
```

### 7) Compile the LaTeX report

```bash
cd reports
latexmk -pdf -interaction=nonstopmode Model_Validation_Report.tex
```

## Generated report outputs

### Figures (`reports/figures`)

Typical outputs include:

* `champion_reliability_train.png`
* `champion_reliability_oot.png`
* `champion_score_hist_train_vs_oot.png`
* `psi_top_drivers.png`
* `challenger_shap_bar.png`

### Tables (`reports/tables`)

CSV outputs include:

* `model_summary_train_oot.csv`
* `oot_compare_champion_vs_challenger.csv`
* `score_stability_compare.csv`
* `psi_table.csv`
* `sensitivity_table.csv`
* `stress_table.csv`
* `lgd_by_grade.csv`

## Limitations

* The dataset is a public LendingClub dataset, not internal production bank data
* PD target construction uses **closed outcomes only**, which is appropriate for this validation setup but not a full production default-observation framework
* LGD/EAD/EL are **proxy-based** and intended for directional analysis
* Stress scenarios are stylized and not linked to a formal macro forecasting framework
* Conclusions should be interpreted as a validation simulation with governance discipline, not as production approval

## Monitoring and remediation themes (from the report)

The final report includes governance-oriented recommendations such as:

* monitor PSI for key features and score on a monthly or quarterly basis
* track calibration level and calibration slope on recent vintages
* monitor key segments (for example grade, term, purpose) for localized degradation
* define escalation thresholds for moderate and material drift
* apply periodic recalibration if OOT calibration degrades materially
* if the challenger is promoted, apply the same monitoring and interpretability standards

## Requirements

Install project dependencies:

```bash
pip install -r requirements.txt
```

For report compilation, install a LaTeX distribution with:

* `pdflatex`
* `latexmk`

## Author

**Noel P.**

Independent validation project focused on credit risk model validation with an emphasis on out-of-time generalization, calibration, drift monitoring, robustness, challenger governance, and reproducibility.