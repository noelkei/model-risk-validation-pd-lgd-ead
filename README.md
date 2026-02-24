# Independent Model Validation Pack (PD + Proxy LGD/EAD/EL)

An end-to-end **independent model validation** project for **credit risk (PD)** with an extension to **proxy LGD/EAD/EL**, built on the **LendingClub** dataset.

This repository is designed as a **Model Risk Validation** deliverable (not just a model training project). It reproduces a **champion PD model (logistic regression)**, performs **out-of-time (OOT) validation**, evaluates **calibration and drift**, runs **sensitivity/stress testing**, and benchmarks a **challenger (LightGBM)** with interpretability.

---

## Project Objective

The goal is to produce an **Independent Model Validation Report** that demonstrates:

- independent validation of a **PD champion model** (logistic regression),
- explicit **leakage control** (underwriting-time variables only),
- **true OOT validation** with a temporal split,
- assessment of **performance + calibration + drift**,
- **sensitivity and stress testing**,
- benchmark vs **challenger (LightGBM) + SHAP**,
- and an extended **credit-risk stack view** via **PD × LGD × EAD proxies (EL proxy)**.

The emphasis is on **model risk governance and validation discipline**, not only predictive accuracy.

---

## Validation Scope (What This Covers)

### Champion (PD)
- Logistic regression pipeline (auditable / reproducible baseline)
- Underwriting-only features (leakage prevention)
- Train vs OOT evaluation

### Validation Suite
- Discrimination metrics (AUC, KS, PR-AUC, etc.)
- Calibration diagnostics (reliability curves, calibration intercept/slope)
- OOT degradation analysis
- Drift monitoring:
  - feature PSI
  - score PSI
  - score distribution shift (KS)

### Robustness
- Sensitivity analysis with input shocks (±5%, ±10%)
- Stress scenarios (mild / severe)

### Challenger
- LightGBM benchmark with same feature set (fair comparability)
- OOT comparison (performance / calibration / stability)
- SHAP global interpretability (sanity check + governance support)

### Proxy Loss Stack Extension
- Proxy LGD (recoveries-based, defaults only)
- Proxy EAD (funded amount / funded minus repaid principal, if available)
- Expected Loss proxy: `PD_pred × LGD_avg × EAD_proxy`

> **Important:** LGD/EAD/EL are **proxies for directional risk analysis**, not production-ready regulatory models.

---

## Dataset

**LendingClub Loan Data (Kaggle)**

The project uses LendingClub public loan data because it supports:
- **PD target definition** from `loan_status` (closed outcomes),
- **LGD proxy** using `recoveries`,
- **EAD proxy** using `funded_amnt` and optionally `total_rec_prncp`.

### Core fields used (depending on dataset version)
- `issue_d`, `loan_status`
- `funded_amnt`, `recoveries`, `total_rec_prncp` (if present)
- borrower/application variables such as:
  - `annual_inc`, `dti`, `int_rate`, `term`, `grade`, `emp_length`
  - `home_ownership`, `purpose`, `verification_status`
  - `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `total_acc`
  - `revol_util`, `revol_bal`, `pub_rec`, `earliest_cr_line`

---

## Methodology Summary

## 1) Target Definition (PD)
Only **closed outcomes** are used to avoid unresolved labels:
- `default_flag = 1` for `Charged Off` / `Default`
- `default_flag = 0` for `Fully Paid`
- Exclude `Current`, `Late`, `In Grace Period`, etc.

## 2) OOT Validation Design
Temporal split using `issue_d`:
- **Train:** historical vintages up to 2017
- **OOT:** 2018 vintage

(Adjustable based on the available dataset range, preserving temporal holdout logic.)

## 3) Leakage Prevention (Feature Governance)
Only **underwriting-time variables** are allowed in the PD model.

Excluded (post-origination / post-outcome) fields include patterns such as:
- payments (`pymnt`, `last_pymnt`, `total_rec_`)
- recoveries (`recover`)
- outstanding principal (`out_prncp`)
- collections, settlements, hardship-related fields, etc.

Leakage control is a core part of the validation governance story.

## 4) Champion Model
- Logistic Regression (champion)
- Numeric preprocessing: median imputation + scaling
- Categorical preprocessing: imputation + one-hot encoding (`handle_unknown`)
- Missingness policy (drop very high-missing features)

## 5) Challenger Model
- LightGBM
- Same approved feature set as the champion (fair comparison)
- Evaluated on OOT performance, calibration, and stability
- Interpreted using SHAP (global feature importance bar plot)

---

## Main Findings (Typical Outcome / Current Project Narrative)

### Champion (Logistic Regression)
- **Reasonable OOT discrimination** (still useful as a ranking model)
- **Calibration degrades OOT**
  - predicted PD tends to exceed observed OOT default rate
  - calibration slope/intercept worsen vs Train
- Interpreted as **population drift / base-rate shift**, not necessarily total model failure

### Drift
- PSI highlights **material drift** in key inputs, especially:
  - `revol_util` (often RED / material drift)
  - `application_type`, `int_rate` (moderate-to-high drift)
- Consistent with changes in portfolio mix / policy / credit cycle

### Sensitivity
- Rank ordering remains very stable under ±5% / ±10% shocks
  - Spearman correlation ~ 1
- `dti` tends to be the most sensitivity-relevant tested driver

### Stress + EL Proxy
- Baseline < Mild < Severe in PD and EL proxy (monotonic increase)
- LGD proxy tends to be high (low recoveries in unsecured charged-off consumer loans)
- Useful as a **directional vulnerability analysis**, not regulatory LGD modeling

### Challenger (LightGBM)
- Improves OOT discrimination (AUC / PR-AUC)
- Improves OOT calibration characteristics
- Often lower score drift / stronger stability metrics
- SHAP drivers remain plausible and governance-consistent (no leakage-type variables)

### Validation Conclusion
- **Champion:** Moderate Model Risk  
  (usable ranking model, but calibration governance + monitoring needed)
- **Challenger:** credible promotion / redevelopment candidate, subject to governance controls

---

## Repository Structure

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
│   ├── figures
│   └── tables
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

---

## Pipeline Overview (Scripts)

## `00_download_data_kagglehub.py`

Downloads the LendingClub dataset (if using KaggleHub flow).

## `01_make_dataset.py`

Builds the modeling dataset:

* cleaning
* target definition
* leakage screening
* feature engineering
* temporal split preparation

## `02_train_models.py`

Trains the **champion logistic regression** model and stores predictions/artifacts.

## `03_run_validation.py`

Runs the validation suite on champion predictions:

* performance
* calibration
* drift
* sensitivity
* stress
* proxy LGD/EAD/EL summaries

## `04_run_challenger.py`

Trains/evaluates the **LightGBM challenger** and generates challenger comparison outputs + SHAP artifacts.

## `05_generate_report_artifacts.py`

Builds final report artifacts (CSV tables + figures), including:

* model summary train/OOT
* champion vs challenger OOT comparison
* score stability comparison
* PSI table
* sensitivity table
* stress table
* LGD-by-grade summary
* figures (reliability curves, score shift, PSI drivers, SHAP)

## `06_export_tables_to_latex.py`

Converts report CSV tables into LaTeX table artifacts:

* splits wide tables into parts (`*_part1.tex`, `*_part2.tex`, ...)
* builds a wrapper (`*_table.tex`) that stacks parts vertically
* supports clean inclusion in the main LaTeX report

---

## How to Run (End-to-End)

> Assumes Python environment is set up and the dataset is available in `data/raw/`.

### 1) Build / clean dataset

```bash
python scripts/01_make_dataset.py
```

### 2) Train champion

```bash
python scripts/02_train_models.py
```

### 3) Run validation suite

```bash
python scripts/03_run_validation.py
```

### 4) Train/evaluate challenger

```bash
python scripts/04_run_challenger.py
```

### 5) Generate report artifacts (tables + figures)

```bash
python scripts/05_generate_report_artifacts.py
```

### 6) Export CSV tables to LaTeX wrappers/parts

```bash
python scripts/06_export_tables_to_latex.py
```

### 7) Compile the LaTeX report

```bash
cd reports
latexmk -pdf -interaction=nonstopmode Model_Validation_Report.tex
```

---

## Report Outputs

### Figures (`reports/figures`)

Typical outputs include:

* `champion_reliability_train.png`
* `champion_reliability_oot.png`
* `champion_score_hist_train_vs_oot.png`
* `psi_top_drivers.png`
* `challenger_shap_bar.png`

### Tables (`reports/tables`)

CSV + LaTeX wrappers/parts, e.g.:

* `model_summary_train_oot.csv` + `model_summary_train_oot_table.tex`
* `oot_compare_champion_vs_challenger.csv` + wrapper
* `score_stability_compare.csv` + wrapper
* `psi_table.csv` + wrapper
* `sensitivity_table.csv` + wrapper
* `stress_table.csv` + wrapper
* `lgd_by_grade.csv` + wrapper

---

## LaTeX Report Notes (Practical)

The report uses a wrapper/parts approach for tables to avoid layout issues in A4:

* wide tables are split into **index + up to 4 data columns per part**
* parts are stacked vertically in a single logical table
* **captions/labels live only in `Model_Validation_Report.tex`**
* wrappers (`*_table.tex`) and parts (`*_partN.tex`) should not include `\caption` or `\label`

This avoids common LaTeX issues such as:

* nested table environments (`Not in outer par mode`)
* duplicate labels
* inconsistent table width handling

---

## Model Risk Framing (Why This Project Is Different)

This project is intentionally framed as an **independent validation exercise**, not just model development.

It focuses on the question:

> Is the model still usable and governable under temporal change?

That is evaluated through:

* **ranking / discrimination**
* **calibration**
* **drift / stability**
* **robustness under sensitivity + stress**
* **comparative challenger evidence**
* **clear limitations documentation** (LGD/EAD/EL proxies)

---

## Limitations

* **Dataset is public LendingClub data**, not a bank internal production dataset
* PD target is based on **closed outcomes only**, which is appropriate for this validation setup but not equivalent to a production default observation framework
* LGD/EAD/EL are **proxy-based** and intended for **directional analysis**
* Stress scenarios are stylized and not linked to a formal macroeconomic forecasting framework

---

## Suggested Monitoring / Governance Actions (From Validation)

* Track PSI (features + score) monthly/quarterly
* Monitor calibration-in-the-large and calibration slope on recent vintages
* Perform segmented monitoring (grade / term / purpose)
* Define AMBER/RED escalation thresholds for drift
* Recalibrate the champion periodically if OOT calibration degrades materially
* If challenger is promoted, apply the same monitoring and interpretability governance

---

## Requirements

Install dependencies from:

```bash
pip install -r requirements.txt
```

You may also need LaTeX for report compilation:

* `latexmk`
* `pdflatex` (e.g., via TeX Live / MacTeX)

---

## Author

**Noel P.**

Independent validation project focused on **credit risk model validation (PD)** with **drift, calibration, robustness, and challenger governance**.

