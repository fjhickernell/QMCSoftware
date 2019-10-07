""" Definition for abstract class StoppingCriterion """

from abc import ABC, abstractmethod

from .._util import DistributionCompatibilityError, univ_repr

class StoppingCriterion(ABC):
    """ Decide when to stop """

    def __init__(self, distribution, allowed_distribs, abs_tol, rel_tol, n_init, n_max):
        """
        Args:
            distribution (DiscreteDistribution): an instance of DiscreteDistribution
            allowed_distribs: distribution's compatible with the StoppingCriterion
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
        """
        if type(distribution).__name__ not in allowed_distribs:
            error_message = type(self).__name__ + ' only accepts distributions:'\
                            + str(allowed_distribs)
            raise DistributionCompatibilityError(error_message)
        super().__init__()
        self.abs_tol = abs_tol if abs_tol else 1e-2
        self.rel_tol = rel_tol if rel_tol else 0
        self.n_init = n_init if n_init else 1024
        self.n_max = n_max if n_max else 1e8

    @abstractmethod
    def stop_yet(self): # distribution = data or summary of data computed already
        """
        Determine when to stop
        """

    def __repr__(self):
        return univ_repr(self)

    def summarize(self):
        h1 = '%s (%s)\n'
        item_i = '%25s: %d\n'
        item_f = '%25s: %-15.4f\n'

        s = ""
        s += h1 % (type(self).__name__, 'StoppingCriterion Object')
        s += item_f % ('abs_tol', self.abs_tol)
        s += item_f % ('rel_tol', self.rel_tol)
        s += item_i % ('n_max', self.n_max)
        s += item_f % ('inflate', self.inflate)
        s += item_f % ('alpha', self.alpha)

        print(s[:-1])


# API
from .clt import CLT
from .clt_rep import CLTRep