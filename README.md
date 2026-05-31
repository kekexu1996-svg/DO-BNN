# DO-BNN
# DO-BNN Wheat LAI/LNA Prediction

This repository provides a reference implementation of the dual-output Bayesian neural network (DO-BNN) with extreme sample mining (ESM) used for collaborative prediction of wheat leaf area index (LAI) and leaf nitrogen accumulation (LNA) from multimodal canopy features.

The code follows the manuscript settings:

- inputs: spectral, RGB texture, and depth-derived structural features
- outputs: LAI and LNA
- model: two Bayesian fully connected hidden layers with 32 and 16 neurons
- activation: ReLU
- prior: zero-mean Gaussian prior for Bayesian layer weights
- posterior approximation: mean-field Gaussian variational posterior
- optimizer: Adam, learning rate 0.001, batch size 16
- training: up to 800 epochs with early stopping
- MC samples: 20 during training, 100 during inference
- joint loss weights: lambda_1 = 1.0, lambda_2 = 1.0, lambda_3 = 1e-4
- ESM: at most three training samples per mining iteration, reweighted to 2.0

## Repository layout

```text
dobnn_wheat/
  dobnn/
    model.py       # Bayesian layers and DO-BNN model
    features.py    # NDRE/RVI, CHCM structural features, HSV-GLCM texture features
    train.py       # ELBO-style training, MC inference, evaluation
    esm.py         # extreme sample mining and reweighting
    data.py        # CSV loading, year split, z-score normalization
    metrics.py     # R2, RMSE, RRMSE
    cli.py         # command line training entry point
  configs/
    default.yaml   # default experiment configuration
  examples/
    demo_synthetic.py
  scripts/
    train.py
  requirements.txt
  pyproject.toml
```

## Installation

```bash
cd dobnn_wheat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Data format

Prepare a CSV file such as `data/wheat_features.csv`. Each row should represent one plot-level sample. The default configuration expects:

```text
year,NDRE,RVI,CVI,H_mean,H_CV,HP95th,SH_ENT,SH_ASM,SH_CON,SH_COR,SS_ENT,SS_ASM,SS_CON,SS_COR,SV_ENT,SV_ASM,SV_CON,SV_COR,LAI,LNA
```

The default split matches the manuscript design:

- 2022 and 2023 samples are used for training and validation
- 2024 samples are used as the independent test set
- the validation set is split only from the training years

If your feature names differ, edit `configs/default.yaml`.

## Feature extraction helpers

The package includes helper functions for common preprocessing steps:

```python
from dobnn.features import (
    compute_ndre,
    compute_rvi,
    canopy_height_map_from_depth,
    extract_structural_features,
    extract_hsv_glcm_features,
)
```

These helpers cover the manuscript's representative spectral indices, CHCM-style depth correction, canopy structural descriptors, and HSV-GLCM texture descriptors. They are intended as transparent reference utilities; adapt camera intrinsics, ground masks, and texture quantization settings to your acquisition system.

## Quick start with synthetic data

The manuscript data are not included in this repository. To verify that the pipeline works, generate a small synthetic dataset:

```bash
cd dobnn_wheat
python examples/demo_synthetic.py
python scripts/train.py --config configs/default.yaml
```

Outputs are written to `outputs/dobnn_f4/`:

```text
dobnn_state_dict.pt    # trained model weights
metrics.json           # R2, RMSE, and RRMSE on the test set
history.json           # training and validation loss history
scalers.npz            # feature and target normalization statistics
test_predictions.npz   # predictive mean, uncertainty, and ground truth
sample_weight.npy      # final ESM sample weights
```

## Training on real data

1. Place the feature table at `data/wheat_features.csv`, or change `data.csv_path` in `configs/default.yaml`.
2. Confirm that `feature_cols`, `target_cols`, `train_years`, and `test_years` match your dataset.
3. Run:

```bash
python scripts/train.py --config configs/default.yaml
```

## Configuration notes

The main parameters can be changed in `configs/default.yaml`:

```yaml
training:
  learning_rate: 0.001
  batch_size: 16
  epochs: 800
  patience: 80
  mc_train_samples: 20
  lambda_lai: 1.0
  lambda_lna: 1.0
  lambda_kl: 0.0001

esm:
  enabled: true
  iterations: 5
  max_samples_per_iteration: 3
  selected_sample_weight: 2.0
```

## Method summary

All input features and target values are standardized using z-score normalization based on the training set. Predictions are inverse-transformed before reporting metrics. The DO-BNN predicts LAI and LNA simultaneously. During inference, repeated stochastic forward passes are used to estimate both predictive mean and sample-wise uncertainty.

The ESM procedure is conducted only on the training set. Candidate samples are selected using three criteria: high predictive uncertainty, potential maximum response, and potential minimum response. Selected samples are not copied into the validation or test set; instead, their training weights are increased.

## Citation

If you use this implementation, please cite the associated manuscript:

```bibtex
@article{xu_dobnn_wheat_2026,
  title = {Wheat Growth Parameters Prediction based on Dual Output Bayesian Neural Network using Multi-Modal Information},
  author = {Xu, Ke and Cao, Dahui and Tai, Qinyue and Wang, Haiting and Ni, Jun and Hu, Yaocong and Yang, Huicheng},
  year = {2026}
}
```

## License

This project is released under the MIT License.
