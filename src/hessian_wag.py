import sys
import copy
import typing
import torch.nn
import numpy as np
from tqdm import trange
from abc import ABC, abstractmethod
sys.path.append('../')

from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to
from loss_landscapes.metrics.metric import Metric

from src.utils import get_hessians

'''class Metric(ABC):
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
        outputs = ()
        pred_outputs = ()
        device = next(model_wrapper.modules[0].parameters()).device

        for i, batch in (enumerate(self.data_loader)):
            outputs = outputs + tuple([batch[2].to(device)])
            y = model_wrapper.forward((batch[0].to(device), batch[1].to(device)))
            pred_outputs = pred_outputs + tuple([y.expand(1)])

        outputs = torch.cat(outputs)
        pred_outputs = torch.cat(pred_outputs)
        
        err = self.loss_fn(pred_outputs, outputs)
        return err
'''

## Yao Fehlis
def check_nan_in_model(model):
    """
    Checks if any parameter in the model contains NaN values.
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():  # Check if any NaN exists in the tensor
            print(f"⚠️ NaN detected in {name}")
            return True  # Stop at first NaN detection
    print("✅ No NaN values found in model parameters.")
    return False  # No NaN found


def get_sample_from_normal_dist_of_models(model_mu, model_std, og_model):
    new_model = copy.deepcopy(og_model)
    new_model_state_dict = {}
    for layer_name in new_model.state_dict().keys():
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
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = torch.square(layer_wts)
    squared_model.load_state_dict(updated_model_state_dict, strict=False)
    return squared_model

## Yao Fehlis
def sqrt_model_wts(model):
    """
    Computes the square root of each layer in the model while checking for NaN values.
    
    - If a layer contains NaN values, it prints a warning.
    - If a layer has negative values, it skips that layer.
    - Returns a model with square-rooted weights.
    """
    updated_model_state_dict = {}
    sqrt_model = copy.deepcopy(model)  # Deep copy to avoid modifying original model

    for layer_name, layer_wts in model.state_dict().items():
        if torch.isnan(layer_wts).any():
            print(f"⚠️ Warning: NaN detected in {layer_name}, skipping this layer.")
            updated_model_state_dict[layer_name] = layer_wts  # Keep original weights
            continue
        
        if (layer_wts < 0).any():
            print(f"⚠️ Warning: Negative values in {layer_name}, skipping sqrt computation.")
            updated_model_state_dict[layer_name] = layer_wts  # Keep original weights
            continue
        
        # Compute the square root safely
        updated_model_state_dict[layer_name] = torch.sqrt(layer_wts)

    # Load the modified state_dict into the copied model
    sqrt_model.load_state_dict(updated_model_state_dict, strict=False)
    
    return sqrt_model

def unwrap_model(wrapped_model):
    og_model = wrapped_model.modules[0]
    return og_model

def get_normed_model(model, n_summations):
    normed_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = layer_wts / n_summations

    normed_model.load_state_dict(updated_model_state_dict, strict=False)
    return normed_model

def get_scaled_model(model, scalar: float):
    scaled_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = layer_wts*scalar

    scaled_model.load_state_dict(updated_model_state_dict, strict=False)
    return scaled_model

def add_eps_to_model_wts(model):
    eps_model = copy.deepcopy(model)
    updated_model_state_dict = {}
    print("******** Before addding episilon")
    check_nan_in_model(eps_model)
    
    for layer_name in model.state_dict().keys():
        layer_wts = model.state_dict()[layer_name]
        updated_model_state_dict[layer_name] = torch.add(layer_wts, 1E-6)
    eps_model.load_state_dict(updated_model_state_dict, strict=False)
    print("******** After addding episilon")
    check_nan_in_model(eps_model)
    return eps_model


def get_stddev_model(mu_model, var_model, n_samples):
    stddev_model = copy.deepcopy(mu_model)
    updated_model_state_dict = {}
    theta_SWA = square_model_wts(add_eps_to_model_wts(mu_model))

    print("*********** theta_SWA")
    check_nan_in_model(theta_SWA)

    theta_bar = get_normed_model(get_scaled_model(square_model_wts(var_model), n_samples), n_samples - 1.0)

    for layer_name in theta_SWA.state_dict().keys():
        assert layer_name in theta_bar.state_dict().keys(), 'Missing layer in theta_bar: '+layer_name
        mu_SWA = theta_SWA.state_dict()[layer_name]
        bar = theta_bar.state_dict()[layer_name]
        tmp = torch.sub(bar, mu_SWA)
        if torch.any(tmp < 0):
            bp = 0
        updated_model_state_dict[layer_name] = torch.sub(bar, mu_SWA)
    stddev_model.load_state_dict(updated_model_state_dict, strict=False)
    print("*********** stddev_model")
    check_nan_in_model(stddev_model)

    print("*********** sqrt_model_wts(stddev_model)")
    check_nan_in_model(sqrt_model_wts(stddev_model))

    return sqrt_model_wts(stddev_model)


def get_ugly_stddev_model(mu_model,list_of_sample_models, n_samples):
    return 0 



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


    # compute direction vectors
    start_point = model_start_wrapper.get_module_parameters()
    dir_one = (model_end_one_wrapper.get_module_parameters()) / steps
    dir_two = (model_end_two_wrapper.get_module_parameters()) / steps


    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)

    data_matrix = []
    # evaluate loss in grid of (steps * steps) points, where each column signifies one step
    # along dir_one and each row signifies one step along dir_two. The implementation is again
    # a little convoluted to avoid constructive operations. Fundamentally we generate the matrix
    # [[start_point + (dir_one * i) + (dir_two * j) for j in range(steps)] for i in range(steps].
    
    
    ### WAG VARIABLES ###
    summed_model = wrap_model((copy.deepcopy(model_start)))
    var_model = wrap_model(square_model_wts(copy.deepcopy(model_start)))
    ugly_var_list = []
    # initial_loss = metric(model_start_wrapper)
    threshold_loss = loss_threshold
    model_ll_coords = []

    for i in trange(steps, desc='Getting Averaged Model'):
        data_column = []

        for j in range(steps):

            # for every other column, reverse the order in which the column is generated
            # so you can easily use in-place operations to move along dir_two
            if i % 2 == 0:
                start_point.add_(dir_two)
                loss = metric(model_start_wrapper)
                data_column.append(loss)
            else:
                start_point.sub_(dir_two)
                loss = metric(model_start_wrapper)
                data_column.insert(0, loss)
#### START WAG####
            if loss <= threshold_loss:
                # print((loss, threshold_loss))
                # Take care of summing model weights here
                summed_model.get_module_parameters().add_(start_point)
                model_ll_coords.append((i, j, loss))

                sqrd_model_plus_eps = add_eps_to_model_wts(unwrap_model(copy.deepcopy(model_start_wrapper)))
                sqrd_model = square_model_wts(sqrd_model_plus_eps)
                var_model.get_module_parameters().add_(wrap_model(sqrd_model).get_module_parameters())
                # ugly_var_list.append(unwrap(copy.deepcopy(model_start_wrapper)))
#### END WAG ####
        data_matrix.append(data_column)
        start_point.add_(dir_one)

    if len(model_ll_coords) > 1:
        averaged_model = get_normed_model(unwrap_model(summed_model), len(model_ll_coords))
        stddev_model = get_stddev_model(averaged_model, unwrap_model(var_model), len(model_ll_coords))
        # stddev_model = get_ugly_stddev_model(averaged_model, ugly_var_list)
    else:
        print('No ensemble of models found.  Suggest increasing loss threshold. Returning final models.')
        averaged_model = unwrap_model(model_start_wrapper)
        stddev_model = unwrap_model(model_start_wrapper)

    return data_matrix, averaged_model, stddev_model, model_ll_coords


def get_hessian_wag(dataloader, loss_func, func, STEPS, model_end_one=None, model_end_two=None, loss_threshold=1.0):

    if (model_end_one == None) and (model_end_two == None):
        model_end_one, model_end_two = get_hessians(func, dataloader, model_wt_dict, loss_func)

    metric = Loss(loss_func, func.eval(), dataloader)
    try:
        loss_data_fin, mu_model, stddev_model, model_ll_coords = hessian_wag(
            model_start=func.eval(), 
            model_end_one=model_end_one.eval(),
            model_end_two=model_end_two.eval(),
            metric=metric, steps=STEPS, deepcopy_model=True,
            loss_threshold = loss_threshold
            )
        
        # loss_landscapes_list.append(loss_data_fin)
        return loss_data_fin, mu_model, stddev_model, model_ll_coords
    except Exception as e:
        print(e+'batch id: '+str(i))
        return None, None, None
    # std_loss_landscape = np.std(tmp, axis=2)

    
