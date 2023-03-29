import numpy as np

from scipy.stats import exponnorm


def param_tuple_to_pdf(x_axis, param_tuple):
    """
    Convert a tuple of parameters to a PDF.
    """
    pdist = np.zeros(x_axis.shape)
    irf_mean, irf_sigma = param_tuple[-2], param_tuple[-1]
    for tau, frac in zip(param_tuple[:-2:2], param_tuple[1:-2:2]):
        pdist += frac * exponnorm.pdf(x_axis, tau/irf_sigma, loc=irf_mean, scale=irf_sigma)
    return pdist