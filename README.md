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

A patched version of, `loss_landscapes` is included in the repo. The notebook downloads the original package, which has a bug preventing the hessians from being used to make the landscapes.

Any attempt at a `cuda` compatible installation is omitted, and the `dgl` dependency for `alignn` is ignored.
