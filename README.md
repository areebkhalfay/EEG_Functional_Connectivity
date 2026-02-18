# Modeling Functional Connectivity During Movie Watching with Intracranial EEG

**Authors:** Reilly Jensen, Gertrude Asumpaame Alayine, Areeb Khalfay  
**Course:** BMED 7610: Quantitative Neuroscience (Georgia Tech)
**Code:** All code displayed in this repository and the entirety of report portions on EEGNet was written by Areeb Khalfay.

## Project Overview
This project investigates how the human brain processes naturalistic visual stimuli (movies) using intracranial EEG (iEEG). We utilized Deep Learning (**EEGNet Convolutional Neural Network**) to distinguish between "continuous cuts" and "scene changes" in the movie *"Bang! You're Dead"* (Alfred Hitchcock).

The codebase has been created with the original research data into a modular Python pipeline for reproducibility.

## Project Structure
```text
movie-ieeg-connectivity/
│
├── data/                   # (Excluded from git) Place NWB files here
│   ├── sub-CS41/           # Subject folders matching NWB structure
│   └── scenecut_info.csv   # Scene cut metadata
│
├── docs/
│   └── BMED_7610_Final_Report.pdf  # Full scientific report
│
├── results/                # Generated figures and logs
│
├── src/                    # Source code modules
│   ├── analysis.py         # Visualization & connectivity matrices
│   ├── config.py           # Paths and hyperparameters
│   ├── data_loader.py      # NWB file parsing and slicing
│   ├── dataset.py          # PyTorch Dataset/Loader logic
│   ├── model.py            # EEGNet Neural Network architecture
│   └── train.py            # Training loops
│
├── main.py                 # Entry point to run the analysis
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Methodology & Report

* **Data Source:** 16 epilepsy patients with stereotactic EEG (sEEG) recordings.
* **Model:** A custom PyTorch implementation of **EEGNet** (Lawhern et al., 2018).
* **Analysis:**
    * **Functional Connectivity:** Coherence and Phase Slope Index (PSI) in Theta, Alpha, and Beta bands.
    * **Deep Learning:** Leave-One-Subject-Out (LOSO) cross-validation to classify scene transitions.
    * **Interpretability:** Analysis of spatial weights to infer connectivity patterns learned by the CNN.

For a deep dive into the neuroscience background, the `EEGNet` architecture, and our specific findings regarding functional connectivity in the Theta/Alpha bands, please refer to our **Final Report**:

**[Full Report (PDF)](docs/BMED_7610_Final_Report.pdf)**

## Results Summary
* **FC Analysis:** Scene changes showed slightly higher theta-band coherence between the Hippocampus and vmPFC compared to continuous cuts.
* **EEGNet:** The model achieved a mean classification accuracy of **63%** (13% above chance), demonstrating that the network could learn generalizable patterns of scene processing across different subjects.

## Setup & Installation

### 1. Prerequisite: Data Access

**Note:** The neural data files are very large and are thus NOT included in this repository.

To reproduce these results, you must download the dataset separately:

1. **Download the Data:**
* Access the dataset at the [DANDI Archive (Dandiset 000623)](https://dandiarchive.org/dandiset/000623) or refer to the original release instructions: [Rutishauser Lab GitHub](https://github.com/rutishauserlab/bmovie-release-NWB-BIDS)
* You need the `.nwb` files for subjects `sub-CS41` through `sub-CS62`.


2. **Organize the Data:**
* Place the downloaded folders inside the `data/` directory.
* Ensure the structure matches: `data/sub-CS41/sub-CS41_ses-P41CSR1_behavior+ecephys.nwb`.


3. **Metadata:**
* Ensure `scenecut_info.csv` is present in the `data/` folder (or update the path in `src/config.py`).



### 2. Environment Setup

Clone the repository and install the required libraries:

```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/movie-ieeg-connectivity.git](https://github.com/YOUR_USERNAME/movie-ieeg-connectivity.git)
cd movie-ieeg-connectivity

# Install dependencies
pip install -r requirements.txt

```

## Usage

### Running the Full Pipeline

To run the Leave-One-Subject-Out (LOSO) training and evaluation loop, simply execute the main script:

```bash
python main.py

```

This script will:

1. Load all available subject data specified in `src/config.py`.
2. Train the **EEGNet** model on N-1 subjects and test on the held-out subject.
3. Print accuracy metrics to the console.
4. Generate connectivity heatmaps and accuracy plots in the `results/` folder.

### Configuration

You can adjust hyperparameters without touching the core code. Open `src/config.py` to modify:

* `DATA_ROOT`: Path to your downloaded NWB files.
* `NUM_EPOCHS`: Training duration (default: 20).
* `BATCH_SIZE`: Training batch size (default: 64).
* `SUBJECT_IDS`: List of subjects to include in the analysis.

## Code Module Guide

If you wish to extend or modify specific parts of the pipeline, here is where to look:

* **`src/model.py`**: Contains the PyTorch definition of **EEGNet**. Modify this to change the CNN architecture (kernels, filters, dropout).
* **`src/data_loader.py`**: Handles the heavy lifting of reading NWB files. Check `make_one_window_per_cut` to see how we sliced EEG data relative to movie timestamps.
* **`src/train.py`**: Contains the training loop (`train_one_fold`). This is where the optimizer, loss function, and backpropagation happen.
* **`src/analysis.py`**: Contains the logic for extracting spatial weights from the model to visualize functional connectivity (brain region correlations).