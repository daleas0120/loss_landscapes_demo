# loss_landscapes_demo

Jupyter notebook which calculates the loss landscape of an ALIGNN model.

<div align="center">
<img src="loss_contours.png" width="45%"/>

<img src="loss_surface.png" width="45%"/>
</div>

## Installation

`conda env create -f environment.yml`

The following packages are installed using `conda`:
- python==3.10
- [alignn](https://github.com/usnistgov/alignn)
- matplotlib
- torchinfo
- ipykernel
- ipympl
- jupyter


An additional package, `loss_landscapes` is cloned from [this repo](https://github.com/marcellodebernardi/loss-landscapes) as a command in the notebook.

Any attempt at a `cuda` compatible installation is omitted, and the `dgl` dependency for `alignn` is ignored.

# Notes from Yao Fehlis

# Machine Learning Project Repository

This repository contains the implementation of a machine learning pipeline, including training, prediction, and Hessian calculation. The project is structured for modularity and ease of use.

## 📂 Folder Structure
```
project_root/ │── configs/ # Configuration files (YAML format) │ ├── train_config.yaml # Training configuration parameters  │── data/ # Dataset storage │ ├── raw/ # Raw input data │ ├── processed/ # Preprocessed data  │ ├── figures/  │ ├── results/ # Model outputs and results │── models/ # Saved trained models │── src/ # Source code │ ├── scripts/ # User-run scripts │ │ ├── train.py # Training script │ │ ├── predict.py # Prediction script │ │ ├── ... │ ├── utils/ # Supporting functions │ │ ├── utils_2.py # Yao Fehlis's utils which is separate than Ashley's utils │ │ ├── model.py # Model architecture definition │ │ ├── ... │── notebooks/ # Jupyter notebooks for analysis and experiments │── requirements.txt # Python dependencies │── README.md # Project documentation (this file)
```

To run code, run like this:
```
python -m src.scripts.train
```
