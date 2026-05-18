# Which Features Are Most Predictive of Employment Outcomes: Experience, Skills or Education?

This project investigates which feature groups are most predictive of employment outcomes using the Stack Overflow Developer Survey dataset. The project compares traditional background information such as education and experience against multiple technical skill representations, including sparse and semantic embedding-based features.

---

# Project Deliverables

- [Final Report](reports/Group_5_Report.pdf)
- [Presentation Slides](slides/Group_5_Which_Features_Are_Most_Predictive_of_Hiring_Outcomes_.pdf)

---

# Team Members

* Jade Beeks
* Avisa Ansari
* Marie Chikovani
* Massimo Vitale
* Faris Selimovic
* Maya Tamimi

---

# Dataset

Dataset used:

* Stack Overflow Developer Survey Dataset

Target variable:

* `Employed`

  * 1 = employed
  * 0 = not employed

Dataset size before cleaning:

* 73,462 rows × 15 columns

Dataset size after cleaning:

* 73,462 rows × 12 columns

---

# Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── stackoverflow_full.csv
│   └── processed/
├── report/
├── slides/
├── src/
    ├── results
│   ├── __init__.py
│   ├── baseline.py
│   ├── combined.py
│   ├── config.py
│   ├── data_utils.py
│   ├── education.py
│   ├── evaluate_skills.py
│   ├── evaluation.py
│   ├── experience.py
│   ├── feature_engineering.py
│   ├── process_skills.py
│   └── skills.py
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Reproducibility and Design

The project was implemented using modular stand-alone Python scripts and reusable preprocessing pipelines to improve reproducibility and avoid notebook execution-order issues.

Running `main.py` executes the full pipeline:

* baseline experiment
* education experiment
* experience experiment
* skills experiment
* combined experiment
* generated visualizations in the results folder

---

# Quick Start

## 1. Create a Virtual Environment

```bash
python3.11 -m venv .venv
```

## 2. Activate the Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is empty or incomplete, install the main packages manually:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn sentence-transformers torch
```

---

# Running the Full ML Pipeline

Run all commands from the repository root directory.

## Full pipeline

```bash
python main.py
```

This will:

* load the dataset
* clean the data
* split into train/validation/test sets
* train the models
* evaluate the models
* generate plots and comparison tables
* run diagnostics for the skills experiments

---

# Pipeline Overview

The machine learning pipeline follows these stages:

1. Load raw Stack Overflow survey data
2. Remove leakage-prone columns
3. Handle missing values
4. Split into train/validation/test sets
5. Fit preprocessing on training data only
6. Generate feature representations:

   * education
   * experience
   * counts skills
   * grouped skills
   * embeddings skills
   * binary skills
7. Train machine learning models
8. Evaluate performance through metrics
9. Evaluation performance through visualizations


---

# Feature Representations

## Baseline Feature Set

The baseline model used simple structured respondent information before introducing richer skill representations.

### Included Features

#### Education Features
- `EdLevel` (ordinal encoded)
- categorical background variables
  - one-hot encoded

#### Experience Features
- `YearsCode`
- `YearsCodePro`
  - standardized using `StandardScaler`

## Education Features

* ordinal encoding for `EdLevel`
* one-hot encoding for categorical background variables

## Experience Features

* `YearsCode`
* `YearsCodePro`
* standardized using `StandardScaler`

## Skills Representations

### Counts

Counts the number of distinct technical skills.

### Grouped

Maps technologies into broad domains:

* frontend
* backend
* databases
* cloud
* DevOps
* programming languages
* data science
* mobile

### Embeddings

Uses:

* `SentenceTransformer`
* model: `all-MiniLM-L6-v2`

Embeddings convert skill profiles into dense semantic vectors that capture similarity relationships between technologies.

### Binary

Creates one binary feature per skill:

* 1 = respondent has the skill
* 0 = respondent does not have the skill

---

# Outputs

Pipeline outputs are saved under:

* `src/results/`
* `src/processed/`

Expected outputs include:

* trained models
* evaluation metrics
* ROC-AUC comparisons
* figures and plots
* diagnostic results

---

# Key Findings

Main findings:

* education features alone were weak predictors
* experience-only models performed poorly
* technical skills carried the strongest predictive signal
* semantic embeddings provided the strongest balance between performance and likely generalisability
* binary skill representations achieved near-perfect validation performance and were removed from final combined model selection

---

# References

Main libraries used:

* Scikit-learn
* Pandas
* NumPy
* SentenceTransformers
