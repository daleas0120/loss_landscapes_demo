import sys
import copy
import typing
import torch
import torch.nn
import numpy as np
from tqdm import trange
from abc import ABC, abstractmethod
sys.path.append('../')

from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to
# from loss_landscapes.metrics.metric import Metric

from src.utils import get_hessians

class Metric(ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass

class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, model, data_loader):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model
        self.data_loader = data_loader

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        outputs = []
        pred_outputs = []
        device = next(model_wrapper.modules[0].parameters()).device

        for batch in self.data_loader:
            outputs.append(batch[2].to(device))
            y = model_wrapper.forward((batch[0].to(device), batch[1].to(device)))
            pred_outputs.append(y.expand(1))

        outputs = torch.cat(outputs)
        pred_outputs = torch.cat(pred_outputs)
        
        return self.loss_fn(pred_outputs, outputs)


class LossPyTorch(Metric):
    def __init__(self, loss_fn, model, data_loader):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model
        self.data_loader = data_loader

    def __call__(self, model) -> float:
        outputs = []
        pred_outputs = []
        device = next(model.parameters()).device

        for batch in self.data_loader:
            outputs.append(batch[2].to(device))
            y = model.forward((batch[0].to(device), batch[1].to(device)))
            pred_outputs.append(y.expand(1))

        outputs = torch.cat(outputs)
        pred_outputs = torch.cat(pred_outputs)
        
        return self.loss_fn(pred_outputs, outputs)

def get_sample_from_normal_dist_of_models(model_mu, model_std, og_model):
    new_model = copy.deepcopy(og_model)
    new_model_state_dict = {}
    for layer_name, layer_param in new_model.named_parameters():
        if layer_name not in model_mu.state_dict().keys():
            continue
        try:
            layer_wts = torch.normal(model_mu.state_dict()[layer_name], model_std.state_dict()[layer_name])
        except:
            layer_wts = new_model.state_dict()[layer_name]
        new_model_state_dict[layer_name] = layer_wts
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model

def square_model_wts(model):
    squared_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = torch.square(layer_wts)
    squared_model.load_state_dict(updated_model_state_dict, strict=False)
    return squared_model

def sqrt_model_wts(model):
    updated_model_state_dict = {}
    sqrt_model = copy.deepcopy(model)
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        layer_wts = torch.sqrt(layer_wts)
        updated_model_state_dict[layer_name] = layer_wts
    sqrt_model.load_state_dict(updated_model_state_dict, strict=False)
    return sqrt_model

def unwrap_model(wrapped_model):
    return wrapped_model.modules[0]

def get_normed_model(model, n_summations):
    normed_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = layer_wts / n_summations

    normed_model.load_state_dict(updated_model_state_dict, strict=False)
    return normed_model

def zero_out_model(model):
    new_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        new_model_state_dict[layer_name] = torch.zeros(layer_wts.shape, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model

def make_normal_dist_model(model, mean, std):
    new_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        mean_vec = torch.full(layer_wts.shape, mean, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
        std_vec = torch.full(layer_wts.shape, std, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
        new_model_state_dict[layer_name] = torch.normal(mean_vec, std_vec)
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model

def make_constant_model(model, const):
    new_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        new_model_state_dict[layer_name] = torch.full(layer_wts.shape, const, dtype=layer_wts.dtype, requires_grad=layer_wts.requires_grad)
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_model_state_dict, strict=False)
    return new_model

def get_scaled_model(model, scalar: float):
    scaled_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = layer_wts * scalar

    scaled_model.load_state_dict(updated_model_state_dict, strict=False)
    return scaled_model

def add_eps_to_model_wts(model):
    eps_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    # for layer_name in model.state_dict().keys():
    for layer_name, layer_param in model.named_parameters():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = torch.add(layer_wts, 1E-6)
    eps_model.load_state_dict(updated_model_state_dict, strict=False)
    return eps_model

def get_stddev_model(mu_model, X2_model, n_samples, epsilon=1E-12):
    stddev_model = zero_out_model(copy.deepcopy(mu_model))
    updated_model_state_dict = {}
    theta_SWA = square_model_wts(mu_model)
    theta_bar = get_normed_model(X2_model, n_samples)

    for layer_name in theta_SWA.state_dict().keys():
        if layer_name.split('.')[-1] == 'num_batches_tracked':
            updated_model_state_dict[layer_name] = theta_SWA.state_dict()[layer_name]
            continue
        
        assert layer_name in theta_bar.state_dict().keys(), 'Missing layer in theta_bar: ' + layer_name
        
        bar = theta_bar.state_dict()[layer_name]
        mu_SWA = theta_SWA.state_dict()[layer_name]
        
        if not torch.allclose(bar, mu_SWA):
            tmp = torch.sub(bar, mu_SWA)
        else:
            tmp = torch.zeros(bar.shape)

        updated_model_state_dict[layer_name] = tmp
    stddev_model.load_state_dict(updated_model_state_dict, strict=False)
    return sqrt_model_wts(stddev_model)

@torch.no_grad()
def get_direction(custom_model, steps):
    # for layer_name in custom_model.state_dict():
    for layer_name, layer_param in custom_model.named_parameters():
        custom_model.state_dict()[layer_name].float().div_(steps)
    return custom_model

@torch.no_grad()
def shift_center(custom_model_start, custom_model_dir_one, custom_model_dir_two, steps=1.):
    # for layer_name in custom_model_start.state_dict():
    for layer_name, layer_param in custom_model_start.named_parameters():
        dir_one = custom_model_dir_one.state_dict()[layer_name].float().div_(2.)
        dir_two = custom_model_dir_two.state_dict()[layer_name].float().div_(2.)
        custom_model_start.state_dict()[layer_name].float().sub_(dir_one)
        custom_model_start.state_dict()[layer_name].float().sub_(dir_two)
    return custom_model_start

@torch.no_grad()
def add_two_models(custom_model_A, custom_model_B):
    # for layer_name in custom_model_A.state_dict():
    for layer_name, layer_param in custom_model_A.named_parameters():
        addend = custom_model_B.state_dict()[layer_name]
        custom_model_A.state_dict()[layer_name].float().add_(addend)
    return custom_model_A

@torch.no_grad()
def sub_two_models(custom_model_A, custom_model_B):
    # for layer_name in custom_model_A.state_dict():
    for layer_name, layer_param in custom_model_A.named_parameters():
        subtrahend = custom_model_B.state_dict()[layer_name]
        custom_model_A.state_dict()[layer_name].float().sub_(subtrahend)
    return custom_model_A

@torch.no_grad()
def hessian_wag_torch(
    model_start, model_end_one, model_end_two, metric, 
    distance=1, steps=20, deepcopy_model=False, loss_threshold=1.0):
    
    dir_one = get_direction(copy.deepcopy(model_end_one), steps)
    dir_two = get_direction(copy.deepcopy(model_end_two), steps)
    start_point = shift_center(copy.deepcopy(model_start), copy.deepcopy(model_end_one), copy.deepcopy(model_end_two), steps)

    del model_end_one, model_end_two

    data_matrix = []

    summed_model = copy.deepcopy(model_start).to('cpu')

    ugly_var_list = []
    threshold_loss = loss_threshold
    model_ll_coords = []

    for i in trange(steps, desc='Getting Averaged Model'):
        data_column = []

        for j in range(steps):

            if i % 2 == 0:
                start_point = add_two_models(start_point, dir_two)
                loss = metric(start_point)
                data_column.append(loss)
            else:
                start_point = sub_two_models(start_point, dir_two)
                loss = metric(start_point)
                data_column.insert(0, loss)
            
            if loss <= threshold_loss:
                summed_model = add_two_models(summed_model, copy.deepcopy(start_point).to('cpu'))
                model_ll_coords.append((i, j, loss))

        data_matrix.append(data_column)
        start_point = add_two_models(start_point, dir_one)

    if len(model_ll_coords) > 1:
        averaged_model = get_normed_model(summed_model, len(model_ll_coords))
        
        start_point = shift_center(copy.deepcopy(model_start), dir_one, dir_two, steps)
        var_model = square_model_wts(sub_two_models(start_point, averaged_model)).to('cpu')

        for i in trange(steps, desc='Getting Stddev Model'):
            for j in range(steps):

                if i % 2 == 0:
                    start_point = add_two_models(start_point, dir_two)
                    loss = data_matrix[i][j] #metric(start_point)
                else:
                    start_point = sub_two_models(start_point, dir_two)
                    loss = data_matrix[i][j]
                
                if loss <= threshold_loss:
                    var_model = add_two_models(var_model, square_model_wts(sub_two_models(start_point, averaged_model)))
            
            start_point = add_two_models(start_point, dir_one)
        
        stddev_model = sqrt_model_wts(get_normed_model(var_model, len(model_ll_coords)))
    else:
        averaged_model = start_point
        stddev_model = start_point
        print('No ensemble of models found. Suggest increasing loss threshold. Returning final models.')
            
    return data_matrix, averaged_model, stddev_model, model_ll_coords

def hessian_wag(model_start: typing.Union[torch.nn.Module, ModelWrapper],
                model_end_one: typing.Union[torch.nn.Module, ModelWrapper],
                model_end_two: typing.Union[torch.nn.Module, ModelWrapper],
                metric: Metric, distance=1, steps=20, deepcopy_model=False, loss_threshold=1.0) -> np.ndarray:
    """
    :param model_start: the model defining the origin point of the plane in parameter space
    :param model_end_one: the model representing the end point of the first direction defining the plane
    :param model_end_two: the model representing the end point of the second direction defining the plane
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    model_end_one_wrapper = wrap_model(copy.deepcopy(model_end_one) if deepcopy_model else model_end_one)
    model_end_two_wrapper = wrap_model(copy.deepcopy(model_end_two) if deepcopy_model else model_end_two)

    start_point = model_start_wrapper.get_module_parameters()
    dir_one = model_end_one_wrapper.get_module_parameters() / steps
    dir_two = model_end_two_wrapper.get_module_parameters() / steps

    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)

    data_matrix = []

    summed_model = wrap_model(copy.deepcopy(model_start))
    var_model = wrap_model(square_model_wts(copy.deepcopy(model_start)))
    ugly_var_list = []
    threshold_loss = loss_threshold
    model_ll_coords = []

    for i in trange(steps, desc='Getting Averaged Model'):
        data_column = []

        for j in range(steps):

            if i % 2 == 0:
                start_point.add_(dir_two)
                loss = metric(model_start_wrapper)
                data_column.append(loss)
            else:
                start_point.sub_(dir_two)
                loss = metric(model_start_wrapper)
                data_column.insert(0, loss)

            if loss <= threshold_loss:
                summed_model.get_module_parameters().add_(start_point)
                model_ll_coords.append((i, j, loss))

                model_plus_eps = unwrap_model(copy.deepcopy(model_start_wrapper))
                sqrd_model = square_model_wts(model_plus_eps)
                var_model.get_module_parameters().add_(wrap_model(sqrd_model).get_module_parameters())

        data_matrix.append(data_column)
        start_point.add_(dir_one)

    if len(model_ll_coords) > 1:
        averaged_model = get_normed_model(unwrap_model(summed_model), len(model_ll_coords))
        stddev_model = get_stddev_model(averaged_model, unwrap_model(var_model), len(model_ll_coords))
    else:
        print('No ensemble of models found. Suggest increasing loss threshold. Returning final models.')
        averaged_model = unwrap_model(model_start_wrapper)
        stddev_model = unwrap_model(model_start_wrapper)

    return data_matrix, averaged_model, stddev_model, model_ll_coords

def get_hessian_wag(dataloader, loss_func, func, STEPS, model_end_one=None, model_end_two=None, loss_threshold=1.0):
    if model_end_one is None and model_end_two is None:
        model_end_one, model_end_two = get_hessians(func, dataloader, model_wt_dict, loss_func)

    metric = LossPyTorch(loss_func, func.eval(), dataloader)

    try:
        loss_data_fin, mu_model, stddev_model, model_ll_coords = hessian_wag_torch(
            model_start=func.eval(), 
            model_end_one=model_end_one.eval(),
            model_end_two=model_end_two.eval(),
            metric=metric, steps=STEPS, deepcopy_model=True,
            loss_threshold=loss_threshold
        )
        return loss_data_fin, mu_model, stddev_model, model_ll_coords
    except Exception as e:
        print(f'{e} batch id: {i}')
        return None, None, None