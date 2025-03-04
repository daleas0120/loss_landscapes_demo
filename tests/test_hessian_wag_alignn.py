import copy
import torch
import glob
import json

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import ipywidgets as widgets

from alignn.pretrained import *

from src.utils import *
from src import botcher_hessian_alignn as hess
from src import botcher_utilities as util
from src.hessian_wag import get_hessian_wag
from src.hessian_wag import get_sample_from_normal_dist_of_models


def zero_out_model(model):
    new_model_state_dict = {}
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        new_model_state_dict[layer_name] = torch.zeros(layer_wts.shape, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model


def make_constant_model(model, const):
    new_model_state_dict = {}
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        new_model_state_dict[layer_name] = torch.full(layer_wts.shape, const, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model

def make_normal_dist_model(model, mean, std):
    new_model_state_dict = {}
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        mean_vec = torch.full(layer_wts.shape, mean, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
        std_vec = torch.full(layer_wts.shape, std, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
        new_model_state_dict[layer_name] = torch.normal(mean_vec, std_vec,)
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model



