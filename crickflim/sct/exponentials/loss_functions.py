from abc import abstractmethod, ABC
from typing import Callable

import numpy as np

from crickflim.sct.exponentials.param_tuple import param_tuple_to_pdf

class LossFunction(ABC):
    """ Wrapper class to keep track of allowed / implemented loss functions """

    @abstractmethod
    def compute_loss(self, params : tuple, data : np.ndarray)->float:
        """ Returns the loss value of the model prediction """
        pass

    def from_data(self, data)->Callable[[tuple], float]:
        """ Reeturns a function that can be used in scipy.optimize.minimize """
        def loss(params):
            return self.compute_loss(params, data)
        return loss
    
    def from_data_frac(self, curr_params, data)->Callable[[tuple], float]:
        """
        Returns a function that can be used in scipy.optimize.minimize
        but only uses the fraction parameters
        """

        def loss(frac_params):
            new_params = list(curr_params)
            new_params[1:-2:2] = frac_params

            return self.compute_loss(tuple(new_params), data)
        return loss

    def __call__(self, params : tuple, data : np.ndarray)->float:
        return self.compute_loss(params, data)
    
class ChiSquared(LossFunction):

    def compute_loss(
            self,
            params: tuple,
            data: np.ndarray,
            exclude_wraparound: bool = True
        ) -> float:
        """ Returns the chi-squared value of the model prediction vs the data """
        predicted = np.sum(data)*param_tuple_to_pdf(np.arange(data.size), params)
        min_bin = int(params[-2] - params[-1]) if exclude_wraparound else 0 # one sigma earlier than mean only
        return np.sum(((predicted[min_bin:] - data[min_bin:]) ** 2) / predicted[min_bin:])
    
class MSE(LossFunction):

    def compute_loss(
            self,
            params: tuple,
            data: np.ndarray,
            exclude_wraparound: bool = True
        ) -> float:
        """ Returns the mean squared error value of the model prediction vs the data """
        predicted = np.sum(data)*param_tuple_to_pdf(np.arange(data.size),params)
        min_bin = int(params[-2] - params[-1]) if exclude_wraparound else 0 # one sigma earlier than mean only
        return np.sqrt(((predicted[min_bin:] - data[min_bin:]) ** 2).sum())
