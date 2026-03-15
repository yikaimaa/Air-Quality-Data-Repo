# Air-Quality-Data-Repo

A machine learning research repository for **PM2.5 air pollution forecasting**, combining environmental monitoring data, meteorological signals, and wildfire activity indicators to model short-term air quality dynamics.

The project investigates whether **predictive analytics can improve next-day air-quality forecasting**, supporting proactive environmental risk management and public-health decision making.

The repository contains the full pipeline for:

- environmental data integration
- data cleaning and quality control
- time-series feature engineering
- predictive modeling (statistical and deep learning)
- model evaluation and diagnostics
- research reporting and reproducibility

## Overview

This repository is designed to support an end-to-end air quality workflow:

- collecting and organizing air quality-related datasets
- cleaning and preprocessing raw data
- building and evaluating predictive models
- storing configuration files and trained model outputs
- documenting results in a paper/report format

The main application focus is understanding and modeling **PM2.5 trends**, potentially using environmental and external signals such as weather and wildfire conditions.

## Repository Structure

```text
Air-Quality-Data-Repo/
├── .github/workflows/      # GitHub Actions workflows
├── Datasets/               # Raw and/or processed datasets
├── configs/                # Configuration files for experiments or pipelines
├── model/                  # Saved model artifacts / fitted models
├── paper/                  # Paper, report, Quarto/LaTeX files, references, outputs
├── scripts/                # Data processing, training, evaluation, and utility scripts
├── Air-Quality-Data-Repo.Rproj
├── README.md
├── pm25_trend.png
├── pyproject.toml
└── requirements.txt
```

## Features

- Centralized storage for air quality project assets
- Support for environmental data integration
- Scripts for preprocessing, modeling, and evaluation
- Config-based experiment organization
- Paper/report folder for communicating findings
- Reproducible workflow setup with Python dependencies listed in both `pyproject.toml` and `requirements.txt`

## Tech Stack

This repository uses Python 3.10+ and includes packages such as:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `requests`
- `joblib`
- `torch`
- `statsmodels`

These libraries support data wrangling, statistical analysis, visualization, machine learning, and model development.

## Installation

```bash
git clone https://github.com/yikaimaa/Air-Quality-Data-Repo.git
cd Air-Quality-Data-Repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Getting Started

A typical workflow in this repo may look like:

1. Place or update data files inside `Datasets/`
2. Adjust experiment settings in `configs/`
3. Run preprocessing or feature-generation scripts from `scripts/`
4. Train and evaluate models
5. Save trained outputs to `model/`
6. Summarize findings in `paper/`

## Example Workflow

```bash
# 1. create / activate environment
python -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run project scripts
python scripts/<your_script_name>.py
```
for scripts in order of file .github\workflows\workflow_test.yml

## Continuous Integration

This repository includes a **GitHub Actions workflow** located in:

```
.github/workflows/workflow_test.yml
```

Whenever changes are pushed to the repository, the workflow automatically:

- installs project dependencies
- runs the project pipeline
- generates required outputs

The results are uploaded as **GitHub Actions artifacts**, which can be downloaded from the workflow run page.

## Outputs

This repository includes outputs such as:

- cleaned datasets
- exploratory plots
- PM2.5 trend visualizations
- trained model files
- evaluation summaries
- paper/report deliverables

For example, `pm25_trend.png` appears to be one of the project visualization assets.

## Reproducibility

To keep experiments reproducible:

- store raw data separately from processed outputs
- document configuration choices in `configs/`
- save trained models in `model/`
- keep analysis and write-up materials in `paper/`
- use a virtual environment for dependency isolation

## Suggested Improvements

A few additions that could make this repo even easier to use:

- add a Data Dictionary section describing each dataset
- include script-by-script usage examples
- document the exact training/evaluation pipeline
- add sample commands for reproducing key results
- clarify which files are raw inputs vs generated outputs

## Contributing

Contributions are welcome through issues and pull requests. Suggested contributions include:

- improving documentation
- cleaning and validating datasets
- adding new modeling approaches
- improving reproducibility and automation
- extending evaluation and visualization

## License

Add a license file if you want others to know how this repository can be used.

## Acknowledgments

This repository appears to support an academic/project workflow around air quality data, PM2.5 analysis, and environmental modeling. Thanks to all contributors and data providers involved in assembling the datasets and project materials.
