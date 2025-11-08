# RegAtlas: A Calibrated Multi-Omics Framework for post-GWAS Regulatory Inference Across Psychiatric Disorders

This project provides a configurable machine learning pipeline to integrate data from GWAS, eQTL, and RNA-seq datasets to model and prioritize genetic variants associated with complex diseases.

The script trains, evaluates, and interprets two models (a Multi-Layer Perceptron and LightGBM) and performs a full analysis including calibration, interpretability (SHAP), and feature-set ablation studies.

## Table of Contents
1.  [Installation](#installation)
2.  [File Structure & Data](#file-structure--data)
3.  [Usage (Running the Pipeline)](#usage-running-the-pipeline)
4.  [Pipeline Analysis & Output](#pipeline-analysis--output)
5.  [License](#license)

## Installation

1.  Clone this repository:
    ```bash
    git clone [git clone https://github.com/mansoor315/Psychiatric_GWAS_Project.git]
cd Psychiatric_GWAS_Project
    ```

2.  Install the required dependencies (it's recommended to use a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
    *(**Note**: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your environment after installing the packages).*

## File Structure & Data

The pipeline is run using `run_pipeline.py`. It expects your data files to be in a directory (e.g., `data/`).

### Recommended Project Structure
````

.
├── data/
│   ├── rna\_seq\_eQTLs\_gwas\_schizophrenia.csv
│   └── rna\_seq\_eQTLs\_gwas\_bipolar.csv
│
├── run\_pipeline.py
├── requirements.txt
└── README.md

````

### Expected Data Format

The script requires an input CSV with a specific set of columns (or a subset of them). The **most critical** columns are:

* **Label-defining columns**: `log2FoldChange`, `padj`, `pval_nominal` (used to create the `label`)
* **Grouping column**: `gene_id` (used for the grouped train/test split)
* **Feature columns**: `PVAL`, `BETA`, `SE`, `tss_distance`, `slope`, `baseMean`, `lfcSE`, `af`, etc. (used for model training)

The script will automatically engineer features like `PVAL_neglog10`, `tss_bin`, and `maf_bin` if the base columns are present.

## Usage (Running the Pipeline)

The pipeline is run from the command line using `run_pipeline.py`. It requires two arguments:

* `-i` or `--input`: The path to your input CSV file.
* `-o` or `--output_tag`: A short, unique name for your analysis (e.g., "SCZ" or "BIP"). This tag will be used to create the results directory (e.g., `results_SCZ/`).

---
### Example 1: Running the Schizophrenia (SCZ) Analysis

```bash
python run_pipeline.py --input "data/rna_seq_eQTLs_gwas_schizophrenia.csv" --output_tag "SCZ"
````

This command will:

1.  Read the Schizophrenia data.
2.  Run the full analysis pipeline.
3.  Save all plots, metrics, and result CSVs into a new folder named `results_SCZ/`.

-----

### Example 2: Running the Bipolar Disorder (BIP) Analysis

```bash
python run_pipeline.py --input "data/rna_seq_eQTLs_gwas_bipolar.csv" --output_tag "BIP"
```

This command will:

1.  Read the Bipolar Disorder data.
2.  Run the *exact same* reproducible analysis.
3.  Save all plots, metrics, and result CSVs into a new folder named `results_BIP/`.

## Pipeline Analysis & Output

Running the pipeline executes several key analysis steps and generates a corresponding output folder (`results_TAG/`) containing:

1.  **Model Training**: Trains both an MLP (TensorFlow/Keras) and a LightGBM model.
2.  **Probability Calibration**: Uses Isotonic Regression (trained on the validation set) to create well-calibrated probabilities for the final LightGBM model.
3.  **Model Evaluation**: Generates and saves plots for:
      * ROC Curves (with 95% bootstrap CIs)
      * Precision-Recall Curves (with 95% bootstrap CIs)
      * Calibration Curves (and Brier Score)
      * Confusion Matrices
4.  **Interpretability (SHAP)**: Computes SHAP values for the LightGBM model and saves:
      * SHAP bar plot (feature importance)
      * SHAP beeswarm plot (feature impact)
      * SHAP dependence plots for top features
5.  **Ablation Study**: Trains and evaluates models on different feature subsets (e.g., GWAS-only, GWAS+eQTL) and saves a bar plot comparing their PR-AUC scores.
6.  **Data Export**: Saves CSV files with detailed test-set results:
      * `test_results_full.csv`: All test-set identifiers with true labels and probabilities from all models.
      * `novel_hits_top0.5pct.csv`: A high-precision list of "novel" candidate variants (e.g., high model probability but not genome-wide significant).
      * `precision_at_k.csv`: A table of model precision at different top-K thresholds.

## License

This project is licensed under the MIT License.

```
```
