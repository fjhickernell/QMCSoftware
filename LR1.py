from qmcpy.integrand.LR import LR
from sys import meta_path
import numpy
from qmcpy import *
data = numpy.genfromtxt('binary.csv', dtype=float, delimiter=',', skip_header = True)
# print(data)

mdata = 4
s = data[:mdata,1:]
t = data[:mdata,0]
prior_variance = [1,1e-4,1,1]

no, dim = s.shape
# print(s.shape)

k = LR(IIDStdUniform(dim+1,seed=8), s_matrix = s, t = t, prior_variance = prior_variance)
solution, data = CubMCCLT(k, abs_tol = .001).integrate()
print(data)
print(" ")
k1 = LR(Sobol(dim+1,seed=8), s_matrix = s, t = t,  prior_variance = prior_variance)
solution1, data1 = CubQMCSobolG(k1, abs_tol = .001).integrate()
print(data1)
print(" ")
my_instance = LR(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t, prior_variance = prior_variance)
p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=1024)
y = my_instance.f(p)
print(y.mean())