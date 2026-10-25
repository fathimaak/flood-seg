# flood-seg

Cross-event generalization of flood extent segmentation on Sen1Floods11: U-Net vs. Prithvi-EO.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision
```

For Windows, activate the environment using:

```cmd
.venv\Scripts\activate.bat
```

For CUDA-enabled PyTorch, see the official installation instructions at https://pytorch.org/get-started/locally/.

## Data

```bash
python scripts/download_data.py
python scripts/check_data.py
```

The dataset download includes hand-labelled Sentinel-2 chips, labels, and split CSV files.

## Layout

* `src/` — Dataset and utility code
* `scripts/` — Download and validation scripts
* `configs/` — Experiment configurations
* `notebooks/` — Exploratory notebooks
* `results/` — Metrics and figures
* `app/` — Application demo
* `report/` — Project report
* `tests/` — Automated tests
