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
- [alignn==2024.8.30](https://github.com/usnistgov/alignn)
- matplotlib
- torchinfo
- torchdata
- ipykernel
- jupyter
- ipympl
- pymatgen
- pip:
    - torch==2.4
    - torchdiffeq

A patched version of, `loss_landscapes` is included in the repo. The notebook downloads the original package, which has a bug preventing the hessians from being used to make the landscapes.

**Note**: 2025 Feb 08: Cuda requirement introduced, ALIGNN version pinned to 2024.8.30, torch version pinned to 2.4, dgl version pinned to be compatible.  

> Implementation currently assumes GPU will be available for use.

- Newer ALIGNN versions have a different number of required inputs to the model

## Workflow

1.  `alignn_pretrained.ipynb`: calculates a loss landscape for random directions. 
2.  `demo_calculate_hessians.ipynb`: calculates the hessians for an alignn model after Botcher et al.
3.  `demo_hessian_landscapes.ipynb`: loads previously calculated hessians and uses them to create directioned versions of the loss landscape