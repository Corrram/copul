from copul.copula_sampler import CopulaSampler
from copul.families.core_copula import CoreCopula


class Copula(CoreCopula):
    def rvs(self, n=1, random_state=None, approximate=False):
        """
        Generate random variates from the copula.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate (default is 1).
        random_state : int or None, optional
            Seed for the random number generator.
        approximate : bool, optional

        Returns
        -------
        np.ndarray
            An array of shape (n, 2) containing samples from the copula.
        """
        sampler = CopulaSampler(self, random_state=random_state)
        return sampler.rvs(n, approximate)
