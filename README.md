# Prophet

**PR**ediction **O**f Disease **PHE**no**T**ypes

A high-throughput EHR phenotyping framework combining machine learning and natural language processing (NLP) for scalable identification of neurological diagnoses, severity scales, and outcomes from electronic health records.

## Overview

Prophet provides a modular, configurable pipeline for phenotyping patients from EHR data. It extracts features from clinical notes (NLP keyword matching with negation detection), ICD codes, CPT codes, and medications, then classifies patients using pre-trained machine learning models. The framework is designed for population-scale deployment, processing millions of clinical notes in hours.

## Phenotypes

Prophet includes pre-trained models for 17 neurological phenotypes:

| Phenotype | Model | Data Sources |
|-----------|-------|-------------|
| Brain tumor | `brain_tumor` | Notes, ICD |
| Cardiac arrest | `ca` | Notes, ICD |
| Congestive heart failure | `chf` | Notes, ICD, Meds |
| Epilepsy | `epilepsy` | Notes, ICD, Meds |
| Intracranial hemorrhage | `ich` | Notes, ICD |
| Ischemic stroke | `is` | Notes, ICD |
| Mild cognitive impairment / Alzheimer's disease | `mci` | Notes, ICD, Meds |
| Narcolepsy | (see [NAX-Narcolepsy](https://github.com/bdsp-core/NAX-Narcolepsy)) | Notes, ICD, Meds |
| Neuroinfectious diseases | (external) | Notes |
| Parkinson's disease | `pd` | Notes, ICD, Meds |
| Subarachnoid hemorrhage | `sah` | Notes, ICD |
| Subdural hematoma | `sdh` | Notes, ICD, CPT |
| Traumatic brain injury | `tbi` | Notes, ICD |
| NIH Stroke Scale | (external) | Notes |
| Modified Rankin Scale | (external) | Notes |
| Atrial fibrillation | `af` | ICD |
| Coronary artery disease | `cad` | ICD, CPT |

## Installation

```bash
pip install git+https://github.com/bdsp-core/prophet.git
```

## Quick Start

```python
from prophet import Prophet
import polars as pl

# Initialize
model = Prophet()

# List available models
print(model.get_available_models())

# Check expected data format for a phenotype
print(model.get_data_format('epilepsy'))

# Load your data
notes = pl.read_parquet('notes.parquet')
icd = pl.read_parquet('icd.parquet')
med = pl.read_parquet('med.parquet')

# Run prediction
results = model.predict(
    phenotype='epilepsy',
    note=notes,
    icd=icd,
    med=med
)
```

See `example/` for sample input data formats.

## Data Format

Each phenotype expects one or more input DataFrames (Polars or Pandas):

- **`note`** -- Clinical notes with columns: `patient_id`, `note_date`, `note_text`
- **`icd`** -- ICD codes with columns: `patient_id`, `icd_date`, `icd_code`
- **`med`** -- Medications with columns: `patient_id`, `med_date`, `med_name`
- **`cpt`** -- CPT codes (select phenotypes): `patient_id`, `cpt_date`, `cpt_code`

Use `model.get_data_format(phenotype)` to check which DataFrames are required for a specific phenotype.

## Dataset

The annotated training data and model evaluation results are available on the Brain Data Science Platform:

**[Prediction of Phenotypes (PROPHET) on BDSP](https://bdsp.io/content/480zbyvqxlq0agc6myih/)**

The dataset spans 18,282 patients and 34,162 annotated clinical visits from six U.S. academic medical centers (BIDMC, MGH, BCH, Stanford, Emory, Kaiser Permanente). Access requires BDSP credentialing and a signed Data Use Agreement.

## Project Structure

```
prophet/
├── src/prophet/
│   ├── prophet.py              # Main Prophet class
│   ├── base/
│   │   └── _basemodel.py       # Base model class for all phenotypes
│   ├── models/                  # Pre-trained phenotype models
│   │   ├── epilepsy/            # Each phenotype has:
│   │   │   ├── config.yaml      #   - Feature definitions (keywords, ICD, meds)
│   │   │   ├── predictor.py     #   - Prediction pipeline
│   │   │   └── *.joblib         #   - Pre-trained classifier
│   │   ├── brain_tumor/
│   │   ├── ca/                  # Cardiac arrest
│   │   ├── chf/                 # Congestive heart failure
│   │   ├── ich/                 # Intracranial hemorrhage
│   │   ├── is/                  # Ischemic stroke
│   │   ├── mci/                 # MCI / Alzheimer's
│   │   ├── pd/                  # Parkinson's disease
│   │   ├── sah/                 # Subarachnoid hemorrhage
│   │   ├── sdh/                 # Subdural hematoma
│   │   ├── tbi/                 # Traumatic brain injury
│   │   └── template/            # Template for adding new phenotypes
│   └── utils/
│       ├── model_comp.py        # Model training & evaluation framework
│       └── preprocessing.py     # Data preprocessing utilities
├── example/                     # Sample input data
│   ├── notes.csv
│   ├── icd_codes.csv
│   └── medications.csv
├── pyproject.toml
└── LICENSE
```

## Adding New Phenotypes

Prophet's modular design supports rapid integration of additional phenotypes. See `src/prophet/models/template/` for a template and `example.ipynb` for a walkthrough.

## Dependencies

polars, scikit-learn, NLTK, Ray, joblib, pyarrow, seaborn, matplotlib, xgboost

## Citation

If you use Prophet in your research, please cite:

> Turley N, Fernandes MB, Sartipi S, Wu H, Lam A, et al. High-Throughput EHR Phenotyping: A Multi-site Annotated Dataset of Neurologic Diagnoses, Severity Scales, and Outcomes. *In preparation.*

## License

CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International). See [LICENSE](LICENSE) for details.
