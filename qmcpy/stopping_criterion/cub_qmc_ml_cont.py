from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MLQMCData
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
from numpy import *
from numpy.linalg import lstsq
from scipy.stats import norm
from time import time
import warnings


class CubQMCMLCont(StoppingCriterion):
    """
    Stopping criterion based on continuation multi-level quasi-Monte Carlo.

    >>> mlco = MLCallOptions(Lattice(seed=7))
    >>> sc = CubQMCML(mlco,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    10.444...
    >>> data
    Solution: 10.4445        
    MLCallOptions (Integrand Object)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
    Lattice (DiscreteDistribution Object)
        d               2^(6)
        randomize       1
        order           natural
        seed            748493
        mimics          StdUniform
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      1
        decomp_type     pca
    CubQMCML (StoppingCriterion Object)
        rmse_tol        0.019
        n_init          2^(8)
        n_max           10000000000
        replications    2^(5)
        levels_min      2^(1)
        levels_max      10
        n_tols          10
        tol_mult        1.200
        theta_init      2^(-2)
        theta           0.01
    MLQMCData (AccumulateData Object)
        levels          7
        dimensions      [ 1.  2.  4.  8. 16. 32. 64.]
        n_level         [8192.  256.  256.  256.  256.  256.  256.]
        mean_level      [1.005e+01 1.821e-01 1.048e-01 5.404e-02 2.787e-02 1.386e-02 7.084e-03]
        var_level       [2.254e-05 7.454e-05 3.118e-05 1.288e-05 3.455e-06 1.263e-06 3.503e-07]
        bias_estimate   0.007
        n_total         311296
        time_integrate  ...
    
    References:
        
        [1] M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics,
        de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf
    """

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
        replications=32., levels_min=2, levels_max=10, n_tols=10, tol_mult=100**(1/9), theta_init=0.5):
        """
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance
            alpha (float): uncertaintly level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rmse_tol (float): root mean squared error
                If supplied (not None), then absolute tolerance and alpha are ignored
                in favor of the rmse tolerance
            n_max (int): maximum number of samples
            replications (int): number of replications on each level
            levels_min (int): minimum level of refinement >= 2
            levels_max (int): maximum level of refinement >= Lmin
            n_tols (int): number of coarser tolerances to run
            tol_mult (float): coarser tolerance multiplication factor
            theta_init (float) : initial error splitting constant

        """
        self.parameters = ['rmse_tol','n_init','n_max','replications','levels_min',
            'levels_max','n_tols','tol_mult','theta_init','theta']
        # initialization
        if rmse_tol:
            self.target_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.target_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.replications = float(replications)
        self.levels_min = levels_min
        self.levels_max = levels_max
        self.theta_init = theta_init
        self.theta = theta_init
        self.n_tols = n_tols
        self.tol_mult = tol_mult
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        # Verify Compliant Construction
        allowed_levels = ['adaptive-multi']
        allowed_distribs = ["Lattice", "Sobol","Halton"]
        super(CubQMCMLCont,self).__init__(allowed_levels, allowed_distribs)

    def integrate(self):
        # Construct AccumulateData Object to House Integration Data
        self.data = MLQMCData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.levels_min, self.n_init, self.replications)
        # Loop over coarser tolerances
        for t in range(self.n_tols):
            self.rmse_tol = self.tol_mult**(self.n_tols-t-1)*self.target_tol # Set new target tolerance
            self._integrate()
        return self.data.solution,self.data

    def _integrate(self):
        """ See abstract method. """
        t_start = time()
        #self.theta = self.theta_init
        self.data.levels = int(self.levels_min+1)

        converged = False
        while not converged:
            # Ensure that we have samples on the finest level
            self.data.update_data()
            self._update_theta()

            while self._varest() > (1-self.theta)*self.rmse_tol**2:
                efficient_level = argmax(self.data.var_cost_ratio_level[:self.data.levels])
                self.data.eval_level[efficient_level] = True

                # Check if over sample budget
                total_next_samples = (self.data.replications*self.data.eval_level*self.data.n_level*2).sum()
                if (self.data.n_total + total_next_samples) > self.n_max:
                    warning_s = """
                    Alread generated %d samples.
                    Trying to generate %d new samples, which would exceed n_max = %d.
                    Stopping integration process.
                    Note that error tolerances may no longer be satisfied""" \
                    % (int(self.data.n_total), int(total_next_samples), int(self.n_max))
                    warnings.warn(warning_s, MaxSamplesWarning)
                    self.data.time_integrate += time() - t_start
                    return

                self.data.update_data()
                self._update_theta()

            # Check for convergence
            converged = self._rmse() < self.rmse_tol
            if not converged:
                if self.data.levels == self.levels_max:
                    warnings.warn(
                        'Failed to achieve weak convergence. levels == levels_max.',
                        MaxLevelsWarning)
                    converged = True
                else:
                    self.data._add_level()

        self.data.time_integrate += time() - t_start
    
    def set_tolerance(self, abs_tol=None, alpha=.01, rmse_tol=None):
        """
        See abstract method. 
        
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            alpha (float): uncertaintly level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
                Takes priority over aboluste tolerance and alpha if supplied. 
        """
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = (float(abs_tol) / norm.ppf(1-alpha/2.))

    def _update_theta(self):
        """Update error splitting parameter"""
        max_levels = len(self.data.n_level)
        A = ones((2,2))
        A[:,0] = range(max_levels-2, max_levels)
        y = ones(2)
        y[0] = log2(abs(self.data.mean_level_reps[max_levels-2].mean()))
        y[1] = log2(abs(self.data.mean_level_reps[max_levels-1].mean()))
        x = lstsq(A, y, rcond=None)[0]
        alpha = maximum(.5,-x[0])
        real_bias = 2**(x[1]+max_levels*x[0]) / (2**alpha - 1)
        self.theta = max(0.01, min(0.125, (real_bias/self.rmse_tol)**2))

    def _rmse(self):
        """Returns an estimate for the root mean square error"""
        return sqrt(self._mse())

    def _mse(self):
        """Returns an estimate for the mean square error"""
        return (1-self.theta)*self._varest() + self.theta*self.data.bias_estimate**2

    def _varest(self):
        """Returns the variance of the estimator"""
        return self.data.var_level[:self.data.levels].sum()
