from dataclasses import dataclass
from typing import List

from scipy.stats import exponnorm as emg
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import minimize
import numpy as np
import warnings

from crickflim.sct.exponentials.loss_functions import LossFunction, ChiSquared, MSE

@dataclass
class Exp():
    tau : float
    frac : float

    @property
    def param(self)->list[float]:
        return [self.tau, self.frac]

@dataclass
class Irf():
    mean : float
    sigma : float

    @property
    def param(self)->list[float]:
        return [self.mean, self.sigma]

class MultipleExponentialFits:
    """ No more fussing around with if statements,
    will work for arbitary numbers of exponentials """

    def __init__(self,
                loss_function : LossFunction = ChiSquared(),
                n_exp : int = 2,
        ):

        self.n_exp = n_exp

        self.exps  : List[Exp] = [Exp(6+6*j, 1.0/n_exp) for j in range(n_exp)]
        self.irf : Irf = Irf(5.0, 0.5)

        # The parameters for this model
        #self.params = self.generate_initial_params()

        # The bounds for the optimization (2 default bounds, change based on # expn)
        #self.bounds = self.generate_bounds()

        # Go ahead, use your favorite!
        self.loss_func = loss_function

    def __str__(self) -> str:
        return f"""Multiple Exponential fit with {self.n_exp} exponentials\n
        NOTE: these units are in BINS, not nanoseconds\n
        Exponentials: {self.exps}\n
        IRF: {self.irf}
        """

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def params(self)->tuple[float]:
        """ (Tau, frac, tau, frac ... , mean, sigma)"""
        return tuple(
            list(param for exp in self.exps for param in exp.param) +
            self.irf.param
        )

    @property
    def bounds(self)->Bounds:
        """ :returns a bounds object from the scipy optimization library that limits the possible
            values our parameters can take on to ensure realistic values are achieved (this is not really
            bias, since the bounds match physical properties, these are not arbitrary) """

        # Switch the type of bounds depending on the # of exponential
        return Bounds(
            # lower bounds
            [0, 0]*(self.n_exp + 1),
            # upper bounds
            [np.inf, 1]*self.n_exp + [100, 10],
        )
    
    @property
    def fraction_bounds(self)->Bounds:
        """ Fraction-only bounds"""
        return Bounds(
            [0.0] * self.n_exp,
            [1] * self.n_exp
        )

    @property
    def constraints(self)->list[LinearConstraint]:
        """ Exponential fractions sum to one, taus in increasing order """
        sum_exps_constraint = [LinearConstraint(
            A=np.array([0.0,1]*self.n_exp + [0.0,0.0]),
            lb=np.array([1]),
            ub=np.array([1]),
        )]

        increasing_taus_constraint = [
            LinearConstraint(
                A=np.array(
                [0,0]*(exp_num-1) + # preceding exponentials
                [1,0] + [-1,0] +
                [0,0]*(self.n_exp-exp_num-1) + # tailing exponentials
                [0,0] #IRF
                ),
                ub = np.array([0]),
            )
            for exp_num in range(1, self.n_exp)
        ]
        return sum_exps_constraint + increasing_taus_constraint

    @property
    def fraction_constraints(self)->list[LinearConstraint]:
        """ For when the taus and IRF are fixed """
        return [LinearConstraint(
            A=np.array([1.0]*self.n_exp),
            lb=np.array([1]),
            ub=np.array([1]),
        )]

    @params.setter
    def params(self, new_params):
        """ :param new_params: a list of the new parameters for the model
        (Tau, frac, tau, frac, ... , mean, sigma) 
        """

        if len(new_params) != 2*self.n_exp + 2:
            raise ValueError(f"Incorrect number of parameters (should be {self.n_exp*2 + 2})")
        # Update the parameters
        self.exps = [Exp(tau, frac) for tau, frac in zip(new_params[:-2:2], new_params[1:-2:2])]
        self.irf = Irf(new_params[-2], new_params[-1])

    @property
    def taus(self)->np.ndarray:
        return np.array([exp.tau for exp in self.exps])
    
    @property
    def fracs(self)->np.ndarray:
        return np.array([exp.frac for exp in self.exps])

    def fit(self, arrival_times : np.ndarray)->None:
        """
        Fits the `params` attribute
        of the model to the data in `arrival_times`
        """

        # Initialize the error function (has to be done inside this function so we have access to the
        # photon distribution)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Optimize over the defined error function
            optimization_results = minimize(
                self.loss_func.from_data(arrival_times),
                self.params,
                method='trust-constr',
                bounds=self.bounds,
                constraints=self.constraints,
            )

        # Update the internal parameters
        self.params = optimization_results.x

    def pdist(self, x: np.ndarray):
        """ :returns the predicted distribution for the current model """
        pdist = np.zeros(x.shape)
        for exp in self.exps:
            tau, f = exp.param
            pdist += f * emg.pdf(
                x,
                K=tau/self.irf.sigma,
                loc=self.irf.mean,
                scale=self.irf.sigma
            ) 
        # The predicted values
        return pdist
    
    def fit_fraction_for_output(self, arrival_times : np.ndarray)->np.ndarray:
        """ Fit the fractions only and RETURNS them but does not STORE them """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimization_results = minimize(
                self.loss_func.from_data_frac(self.params, arrival_times),
                [exp.frac for exp in self.exps],
                method='trust-constr',
                bounds=self.fraction_bounds,
                constraints=self.fraction_constraints,
            )

        return optimization_results.x

    def calculate_empirical_tau(self, histogram : np.ndarray)->float:
        """
        Expects a histogram of photon counts by time bin,
        or an array of histograms with the "arrival time" axis being the last axis
        """
        p : np.ndarray = histogram/histogram.sum(axis = -1, keepdims=True)
        return np.sum(p * (np.arange(p.shape[-1])-self.irf.mean), axis = -1)
