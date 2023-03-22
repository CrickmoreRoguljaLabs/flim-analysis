from scipy.stats import exponnorm as emg
from scipy.optimize import Bounds
from scipy.optimize import minimize
import numpy as np


# ----------------------------------------- Optimization Related Functions -----------------------------------------


def calculate_prediction(model_params, exp_count, x):
    """ :returns the result of applying a single or double (depending on exp_count) exponentially modified gaussian
        (with parameters = model_params) on the data x """

    # Switch depending on the number of exponential terms in the model
    if exp_count == 1:

        # Single exponentially modified gaussian => model_params = # [exponential k , IRF mean, IRF sigma]
        predictions = emg.pdf(x, K=model_params[0], loc=model_params[1], scale=model_params[2])

    else:

        # Double exponentially modified gaussian => model params = [f_n, exp k #1, IRF mean, IRF sigma, exp k#2]
        predictions = model_params[0] * emg.pdf(x, K=model_params[1], loc=model_params[2], scale=model_params[3]) + \
                      (1 - model_params[0]) * emg.pdf(x, K=model_params[4], loc=model_params[2], scale=model_params[3])

    return predictions


def mse(model_params, y, exp_count, x):
    """ :returns the mse value of the model prediction """

    # the predicted values
    predictions = calculate_prediction(model_params, exp_count, x)

    # Calculate the MSE
    return np.sqrt(((predictions - y) ** 2).sum())


def chi_squared(model_params, y, exp_count, x):
    """ :returns the chi^2 value of the model prediction """

    # the predicted values
    predictions = calculate_prediction(model_params, exp_count, x)

    # Cast y values <= 0 -> 0.0001 so we can divide without issue (slightly messy)
    y = [ind_count if ind_count > 0 else 0.0001 for ind_count in y]

    # Calculate Chi^2
    return sum(((predictions - y) ** 2) / y)


# ----------------------------------------- ExponentiallyModifiedGaussian -----------------------------------------


class ExponentiallyModifiedGaussian:

    def __init__(self, bin_width, exponential=2):

        # The number of exponential terms that can be used by this model
        self.expn_count = exponential

        # The parameters for this model
        self.params = self.generate_initial_params()

        # The bounds for the optimization (2 default bounds, change based on # expn)
        self.bounds = self.generate_bounds()

        # The size of the bins (this is taken from the flim file to help convert to nanoseconds correctly
        self.bin_width = bin_width

    def generate_bounds(self):
        """ :returns a bounds object from the scipy optimization library that limits the possible
            values our parameters can take on to ensure realistic values are achieved (this is not really
            bias, since the bounds match physical properties, these are not arbitrary) """

        # Switch the type of bounds depending on the # of exponential
        if self.expn_count == 1:

            # Bounds indexing = [exponential k , IRF mean, IRF sigma]
            return Bounds([0.0, -1, 0], [np.inf, 100, 100])

        else:

            # Bounds indexing = [f_n, exp k #1, IRF mean, IRF sigma, exp k#2]
            return Bounds([0.0, 0.0, -1, 0.0, 0.0], [1, np.inf, 100, 100, np.inf])

    def fit(self, x, y):
        """ :returns [tau_1, tau_2, average_tau, empirical_tau]
            Fit the current model (with the predefined # of exponential components) to the real photon
            histogram w.r.t the optimization function
            updates the parameters of this mode"""

        # Initialize the error function (has to be done inside this function so we have access to the
        # photon distribution)
        optimization_func = lambda params: mse(params, y, self.expn_count, x)

        # Optimize over the defined error function
        optimization_results = minimize(optimization_func, self.params, method='trust-constr', bounds=self.bounds)

        # Update the internal parameters
        self.params = optimization_results.x

        # Calculate and return the various tau values
        taus = self.calculate_tau_values(y)
        return list(taus) + [mse(self.params, y, self.expn_count, x)]  # Append the mse for error tracking

    def calculate_empirical_tau(self, y):
        """ :returns empirical tau => the mean arrival time of the photon """

        # The index of the mean of the IRF function
        irf_mean = self.params[1] if self.expn_count == 1 else self.params[2]

        # Total time weighted sum of photons
        total_sum = 0

        # Total number of photons
        total_terms = 0

        # Iterate through each of the time bins
        for i, photon_count in enumerate(y):

            # Only if this data occurs after the irf mean do we include it
            if i > irf_mean:

                # Weight the number of photons by their time of arrival
                total_sum += i * y[i]

                # increment the total number of observed photons
                total_terms += y[i]

        # Convert to nanoseconds and round to be more clean
        return round(((total_sum / total_terms) - irf_mean) / 1000.0 * self.bin_width, 4)

    def calculate_tau_values(self, y):
        """ :returns 4 tau_values in the form of a list [tau_1, tau_2, average_tau, empirical_tau]
            in the case that there is no tau_2 => tau_2 = -1, average_tau = tau_1 """

        # The calculation changes based on the number of exponentials
        if self.expn_count == 1:

            # Using the scipy optimization library we know tau = K * sigma
            tau_1 = float(self.params[[0]] * self.params[2] * self.bin_width / 1000)  # convert to nanoseconds
            tau_2 = -1  # Filler value
            average_tau = tau_1  # average is the single datapoint

        else:

            # Using the scipy optimization library we know tau = K * sigma
            tau_1 = float(self.params[[1]] * self.params[3] * self.bin_width / 1000)  # convert to nanoseconds
            tau_2 = float(self.params[[4]] * self.params[3] * self.bin_width / 1000)  # convert to nanoseconds
            average_tau = self.params[0] * tau_1 + (1 - self.params[0]) * tau_2

        return [tau_1, tau_2, average_tau, self.calculate_empirical_tau(y)]

    def predict(self, x):
        """ :returns the prediction of this model (calculated by applying the function to the input x values and using
            the saved parameters to make a guess ** this should only be called after fitting )"""

        # Ugly way to write this but I want to save from importing the ind function into other classes
        return calculate_prediction(self.params, self.expn_count, x)

    def generate_initial_params(self):
        """ Given the number of exponentials we generate a placeholder parameters (this will be updated to be the
            results of the optimization (from the fit function) and then be used in the predict function to draw the
            curve """

        # We change the size of the parameter holder based ont he number of exponetials
        if self.expn_count == 1:

            # In this case we are a monoexponetial so we can drop the coefficent and second term
            return [1, 0, 1]  # [exponential k , IRF mean, IRF sigma]
        else:

            # TODO why is this so sensitive to initial f_n guesses?
            # In this case we have two exponentials => 2 k's and now we also need a proportion coefficient
            return [0.75, 1, 0, 1, 1]  # [f_n, exp k #1, IRF mean, IRF sigma, exp k#2]
