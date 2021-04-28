from ._true_measure import TrueMeasure
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import TransformError,DimensionError, ParameterError
from ..discrete_distribution import Sobol
from numpy import *
from numpy.linalg import cholesky, det, inv, eigh
import numpy as np
from numpy import linalg as la
from scipy.stats import norm
from scipy.special import erfcinv


class Gaussian(TrueMeasure):
    """
    Normal Measure.
    
    >>> g = Gaussian(Sobol(2,seed=7),mean=[1,2],covariance=[[9,4],[4,5]])
    >>> g.gen_samples(4)
    array([[ 1.356,  2.568],
           [-2.301, -0.282],
           [ 8.211,  2.946],
           [-0.381,  3.591]])
    >>> g
    Gaussian (TrueMeasure Object)
        mean            [1 2]
        covariance      [[9 4]
                        [4 5]]
        decomp_type     pca
    """

    def __init__(self, sampler, mean=0., covariance=1., decomp_type='PCA', nearest_pd=False):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            mean (float): mu for Normal(mu,sigma^2)
            covariance (ndarray): sigma^2 for Normal(mu,sigma^2). 
                A float or d (dimension) vector input will be extended to covariance*eye(d)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
            nearest_pd (bool): If True, find the nearest positive definite matrix 
                to the supplied covariance matrix. If True, the 'nearestPD' method 
                (defined after this class in this file)
                will be run regardless of if the supplied covariance matrix is PD.
        """
        self.parameters = ['mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = array([[0,1]])
        self._transform = self._transform_std_uniform
        self._jacobian = self._jacobian_std_uniform
        if isinstance(sampler,DiscreteDistribution) and sampler.mimics=='StdGaussian':
            # need to use transformation from standard gaussian
            self.domain = array([[-inf,inf]])
            self._transform = self._transform_std_gaussian
            self._jacobian = self._jacobian_std_gaussian
        self.nearest_pd = nearest_pd
        self._parse_sampler(sampler)
        self.decomp_type = decomp_type.lower()
        self._set_mean_cov(mean,covariance)
        self.range = array([[-inf,inf]])
        super(Gaussian,self).__init__()
    
    def _set_mean_cov(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        if isscalar(mean):
            mean = tile(mean,self.d)
        if isscalar(covariance):
            covariance = covariance*eye(self.d)
        self.mu = array(mean)
        self.sigma = array(covariance)
        if self.sigma.shape==(self.d,):
            self.sigma = diag(self.sigma)
        if not (len(self.mu)==self.d and self.sigma.shape==(self.d,self.d)):
            raise DimensionError('''
                    mean must have length d and
                    covariance must be of shape d x d''')
        if self.nearest_pd:
            self.sigma = nearestPD(self.sigma)
        self._set_constants()
    
    def _set_constants(self):
        if self.decomp_type == 'pca':
            evals,evecs = eigh(self.sigma) # get eigenvectors and eigenvalues for
            order = argsort(-evals)
            self.a = dot(evecs[:,order],diag(sqrt(evals[order])))
        elif self.decomp_type == 'cholesky':
            self.a = cholesky(self.sigma).T
        else:
            raise ParameterError("decomp_type should be 'PCA' or 'Cholesky'")
        self.det_sigma = det(self.sigma)
        self.det_a = sqrt(self.det_sigma)
        self.inv_sigma = inv(self.sigma)  
    
    def _transform_std_uniform(self, x):
        return self.mu + norm.ppf(x)@self.a.T
    
    def _jacobian_std_uniform(self, x):
        return self.det_a/norm.pdf(norm.ppf(x)).prod(1)
    
    def _transform_std_gaussian(self, x):
        return self.mu + x@self.a.T
    
    def _jacobian_std_gaussian(self, x):
        return tile(self.det_a,x.shape[0])

    def _weight(self, x):
        const = (2*pi)**(-self.d/2) * self.det_sigma**(-1./2)
        delta = x-self.mu
        return const*exp(-((delta@self.inv_sigma)*delta).sum(1)/2)

    def _set_dimension(self, dimension):
        m = self.mu[0]
        c = self.sigma[0,0]
        expected_cov = c*eye(int(self.d))
        if not ( (self.mu==m).all() and (self.sigma==expected_cov).all() ):
            raise DimensionError('''
                    In order to change dimension of Gaussian measure
                    mean (mu) must be all the same and 
                    covariance must be a scaler times I''')
        self.d = dimension
        self.mu = tile(m,int(self.d))
        self.sigma = c*eye(int(self.d))
        self._set_constants()


def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input

    see: https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    