[![Build Status](https://travis-ci.com/QMCSoftware/QMCSoftware.png?branch=master)](https://travis-ci.com/QMCSoftware/QMCSoftware) [![codecov](https://codecov.io/gh/QMCSoftware/QMCSoftware/branch/master/graph/badge.svg)](https://codecov.io/gh/QMCSoftware/QMCSoftware) [![Documentation Status](https://readthedocs.org/projects/qmcpy/badge/?version=latest)](https://qmcpy.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3964490.svg)](https://doi.org/10.5281/zenodo.3964490)

# Quasi-Monte Carlo Community Software

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications.

<img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/qmcpy_logo.png?raw=true" alt="QMCPy logo" height=200px width=200px/>

[Homepage](https://qmcsoftware.github.io/QMCSoftware/) ~ [GitHub](https://github.com/QMCSoftware/QMCSoftware) ~ [Read the Docs](https://qmcpy.readthedocs.io/en/latest/) ~ [PyPI](https://pypi.org/project/qmcpy/) ~ [Blogs](http://qmcpy.wordpress.com/) ~ [DockerHub](https://hub.docker.com/r/alegresor/qmcpy) ~ [Contributing](https://github.com/QMCSoftware/QMCSoftware/blob/master/CONTRIBUTING.md) ~ [Issues](https://github.com/QMCSoftware/QMCSoftware/issues)

----

## Installation

~~~
pip install qmcpy
~~~

----

## The QMCPy Framework

The central package including the 5 main components as listed below. Each component is implemented as abstract classes with concrete implementations. For example, the lattice and Sobol' sequences are implemented as concrete implementations of the `DiscreteDistribution` abstract class. An overview of implemented componenets and some of the underlying mathematics is available in the [QMCPy README](https://github.com/QMCSoftware/QMCSoftware/blob/master/qmcpy/README.md).  A complete list of concrete implementations and thorough documentation can be found in the [QMCPy Read the Docs](https://qmcpy.readthedocs.io/en/latest/algorithms.html) .

- **Stopping Criterion:** determines the number of samples necessary to meet an error tolerance.
- **Integrand:** the function/process whose expected value will be approximated.
- **True Measure:** the distribution to be integrated over.
- **Discrete Distribution:** a generator of nodes/sequences, that can be either IID (for Monte Carlo) or low-discrepancy (for quasi-Monte Carlo), that mimic a standard distribution.
- **Accumulate Data:** stores and updates data used in the integration process.

----

## Quickstart

Note: If the following mathematics is not rendering try using Google Chrome and installing the [Mathjax Plugin for GitHub](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).

Will will find the exptected value of the Keister integrand [18]

$$g(\boldsymbol{x})=\pi^{d/2}\cos(||\boldsymbol{x}||)$$

integrated over a $d$ dimensional Gaussian true measure with variance $1/2$

$$\mathcal{X} \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}/2).$$

We may choose a Sobol' discrete distribution with a corresponding Sobol' cubature stopping criterion to preform quasi-Monte Carlo numerical integration.

```python
import qmcpy as qp
from numpy import pi, cos, sqrt, linalg
d = 2
s = qp.Sobol(d)
g = qp.Gaussian(s, covariance=1./2)
k = qp.CustomFun(g, lambda x: pi**(d/2)*cos(linalg.norm(x,axis=1)))
cs = qp.CubQMCSobolG(k, abs_tol=1e-3)
solution,data = cs.integrate()
print(data)
```

A more detailed quickstart can be found in our GitHub repo at `QMCSoftware/demos/quickstart.ipynb` or in [this Google Colab notebook](https://colab.research.google.com/drive/1CQweuON7jHJBMVyribvosJLW4LheQXBL?usp=sharing).

----

## Developers
 
- Sou-Cheng T. Choi
- Fred J. Hickernell
- Michael McCourt
- Jagadeeswaran Rathinavel
- Aleksei Sorokin

----

## Collaborators

- Mike Giles
- Marius Hofert
- Pierre L’Ecuyer
- Christiane Lemieux
- Dirk Nuyens
- Art Owen

----

## Citation

If you find QMCPy helpful in your work, please support us by citing the following work:

~~~
Choi, S.-C. T., Hickernell, F. J., McCourt, M., Rathinavel, J. & Sorokin, A.
QMCPy: A quasi-Monte Carlo Python Library. Working. 2020.
https://qmcsoftware.github.io/QMCSoftware/
~~~

BibTex citation available [here](https://github.com/QMCSoftware/QMCSoftware/blob/master/cite_qmcpy.bib)

----

## References

<b>[1]</b> F. Y. Kuo and D. Nuyens. "Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients - a survey of analysis and implementation," Foundations of Computational Mathematics, 16(6):1631-1696, 2016. ([springer link](https://link.springer.com/article/10.1007/s10208-016-9329-5), [arxiv link](https://arxiv.org/abs/1606.06613))

<b>[2]</b> Fred J. Hickernell, Lan Jiang, Yuewei Liu, and Art B. Owen, "Guaranteed conservative fixed width confidence intervals via Monte Carlo sampling," Monte Carlo and Quasi-Monte Carlo Methods 2012 (J. Dick, F.Y. Kuo, G. W. Peters, and I. H. Sloan, eds.), pp. 105-128, Springer-Verlag, Berlin, 2014. DOI: 10.1007/978-3-642-41095-6_5

<b>[3]</b> Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, GAIL: Guaranteed Automatic Integration Library (Version 2.3.1) [MATLAB Software], 2020. Available from [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).

<b>[4]</b> Sou-Cheng T. Choi, "MINRES-QLP Pack and Reliable Reproducible Research via Supportable Scientific Software," Journal of Open Research Software, Volume 2, Number 1, e22, pp. 1-7, 2014.

<b>[5]</b> Sou-Cheng T. Choi and Fred J. Hickernell, "IIT MATH-573 Reliable Mathematical Software" [Course Slides], Illinois Institute of Technology, Chicago, IL, 2013. Available from [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).

<b>[6]</b> Daniel S. Katz, Sou-Cheng T. Choi, Hilmar Lapp, Ketan Maheshwari, Frank Loffler, Matthew Turk, Marcus D. Hanwell, Nancy Wilkins-Diehr, James Hetherington, James Howison, Shel Swenson, Gabrielle D. Allen, Anne C. Elster, Bruce Berriman, Colin Venters, "Summary of the First Workshop On Sustainable Software for Science: Practice and Experiences (WSSSPE1)," Journal of Open Research Software, Volume 2, Number 1, e6, pp. 1-21, 2014.

<b>[7]</b> Fang, K.-T., and Wang, Y. (1994). Number-theoretic Methods in Statistics. London, UK: CHAPMAN & HALL

<b>[8]</b> Lan Jiang, Guaranteed Adaptive Monte Carlo Methods for Estimating Means of Random Variables, PhD Thesis, Illinois Institute of Technology, 2016.

<b>[9]</b> Lluis Antoni Jimenez Rugama and Fred J. Hickernell, "Adaptive multidimensional integration based on rank-1 lattices," Monte Carlo and Quasi-Monte Carlo  Methods: MCQMC, Leuven, Belgium, April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1411.1966, pp. 407-422.

<b>[10]</b> Kai-Tai Fang and Yuan Wang, Number-theoretic Methods in Statistics, Chapman & Hall, London, 1994.

<b>[11]</b> Fred J. Hickernell and Lluis Antoni Jimenez Rugama, "Reliable adaptive cubature using digital sequences," Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Leuven, Belgium, April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1410.8615 [math.NA], pp. 367-383.

<b>[12]</b> Marius Hofert and Christiane Lemieux (2019). qrng: (Randomized) Quasi-Random Number Generators. R package version 0.0-7. [https://CRAN.R-project.org/package=qrng](https://CRAN.R-project.org/package=qrng).

<b>[13]</b> Faure, Henri, and Christiane Lemieux. “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases,” Mathematics and Computers in Simulation 161 (2019): 13–22. 

<b>[14]</b> M. B. Giles. "Multi-level Monte Carlo path simulation," Operations Research, 56(3):607-617, 2008. [http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf](http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf).

<b>[15]</b> M. B. Giles. "Improved multilevel Monte Carlo convergence using the Milstein scheme," 343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008. [http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf](http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf).

<b>[16]</b> M. B. Giles and B. J. Waterhouse. "Multilevel quasi-Monte Carlo path simulation," pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics, de Gruyter, 2009. [http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf](http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf).

<b>[17]</b> Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]

<b>[18]</b> B. D. Keister, Multidimensional Quadrature Algorithms,  'Computers in Physics', *10*, pp. 119-122, 1996.

<b>[19]</b> L’Ecuyer, Pierre & Munger, David. (2015). LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules. ACM Transactions on Mathematical Software. 42. 10.1145/2754929. 

<b>[20]</b> Fischer, Gregory & Carmon, Ziv & Zauberman, Gal & L’Ecuyer, Pierre. (1999). Good Parameters and Implementations for Combined Multiple Recursive Random Number Generators. Operations Research. 47. 159-164. 10.1287/opre.47.1.159. 

<b>[21]</b> I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, Russian Acamdey of Sciences, Moscow (1992).

<b>[22]</b> Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 2011. 10.1002/wilm.10056. 

<b>[23]</b> Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

<b>[24]</b> S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, SIAM J. Sci. Comput. 30, 2635-2654 (2008).

<b>[25]</b> Paul Bratley and Bennett L. Fox. 1988. Algorithm 659: Implementing Sobol's quasirandom sequence generator. ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. DOI:https://doi.org/10.1145/42288.214372

----

## Sponsors

Illinois Tech
--------------


   <p style="height:30x">
     <a href="https://www.iit.edu">
       <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/illinois-institute-of-technology-vector-logo.jpg?raw=true"/ width="300" height="150">
     </a>
   </p>

Kamakura Corporation
---------------------


   <p style="height:30x">
     <a href="http://www.kamakuraco.com">
       <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/kamakura-corporation-vector-logo.png?raw=true" width="300" height="150"/>
     </a>
   </p>


SigOpt, Inc.
------------


   <p>
     <a href="https://sigopt.com">
       <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/SigOpt_Logo_Files/Horz/Blue/SigoOpt-Horz-Blue.jpg?raw=true" width="300" height="100"/>
     </a>
   </p>

