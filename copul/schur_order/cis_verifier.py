import logging
import numpy as np

log = logging.getLogger(__name__)


class CISVerifier:
    """
    Verifier for Comprehensive Increasing/Decreasing Stochasticity (CIS) property
    of copulas.
    Also known as stochastically increasing/decreasing copulas (CI/CD).
    """

    def __init__(self, cond_distr=1):
        """
        Initialize CISVerifier with the specified conditional distribution to check.

        Parameters:
        -----------
        cond_distr : int
            Which conditional distribution to check (1 or 2)
        """
        self.cond_distr = cond_distr

    def is_cis(self, copul, range_min=None, range_max=None):
        """
        Check if the copula satisfies CIS property.

        Parameters:
        -----------
        copul : Copula
            Copula to check
        range_min : float, optional
            Minimum value for parameter range
        range_max : float, optional
            Maximum value for parameter range

        Returns:
        --------
        tuple
            (is_ci, is_cd) - whether the copula is CI/CD
        """
        log.info(f"Checking if {type(self).__name__} copula is CI")
        range_min = -10 if range_min is None else range_min
        n_interpolate = 20
        linspace = np.linspace(0.001, 0.999, 20)

        # If no parameters, check the copula directly
        try:
            param = str(copul.params[0])
        except IndexError:
            is_ci, is_cd = self._is_copula_cis(copul, linspace)
            if is_ci:
                log.debug("CI True for param: None")
            elif is_cd:
                log.debug("CD True for param: None")
            else:
                log.debug("False for param: None")
            return is_ci, is_cd

        # Handle parameters and their ranges
        interval = copul.intervals[param]
        range_min = float(max(interval.inf, range_min))
        if interval.left_open:
            range_min += 0.01
        param_range_max = 10 if range_max is None else range_max
        param_range_max = float(min(interval.end, param_range_max))
        if interval.right_open:
            param_range_max -= 0.01

        param_range = np.linspace(range_min, param_range_max, n_interpolate)
        points = linspace

        # Initialize result variables
        final_is_ci = None
        final_is_cd = None

        for param_value in param_range:
            param_dict = {param: param_value}
            my_copul = copul(**param_dict)
            is_ci, is_cd = self._is_copula_cis(my_copul, points)

            # Store the latest result
            final_is_ci, final_is_cd = is_ci, is_cd

            if is_ci:
                log.debug(f"CI True for param: {param_value}")
            elif is_cd:
                log.debug(f"CD True for param: {param_value}")
            else:
                log.debug(f"False for param: {param_value}")

        # Return the last result
        return final_is_ci, final_is_cd

    def _is_copula_cis(self, my_copul, points):
        """
        Check if a specific copula instance satisfies CIS property.

        Parameters:
        -----------
        my_copul : Copula
            Copula instance to check
        points : numpy.ndarray
            Points to evaluate

        Returns:
        --------
        tuple
            (is_ci, is_cd) - whether the copula is CI/CD
        """
        is_cis = True
        is_cds = True

        # Get the right conditional distribution method
        if self.cond_distr == 1:
            cond_method = my_copul.cond_distr_1
        elif self.cond_distr == 2:
            cond_method = my_copul.cond_distr_2
        else:
            raise ValueError("cond_distr must be 1 or 2")

        try:
            # Try to get the symbolic function
            cond_method = cond_method().func
        except (TypeError, ValueError):
            # Method-based approach
            for v in points:
                for u, next_u in zip(points[:-1], points[1:]):
                    if self.cond_distr == 1:
                        val1 = cond_method(u, v)
                        val2 = cond_method(next_u, v)
                    else:
                        val1 = cond_method(v, u)
                        val2 = cond_method(v, next_u)

                    # CI: decreasing in u, CD: increasing in u
                    if val1 < val2 - 1e-10:
                        is_cis = False
                    if val1 > val2 + 1e-10:
                        is_cds = False

                    if not is_cis and not is_cds:
                        break

                if not is_cis and not is_cds:
                    break
        else:
            # Symbolic function approach
            for v in points:
                cond_distr_eval_u = cond_method.subs(my_copul.v, v)
                for u, next_u in zip(points[:-1], points[1:]):
                    eval_u = cond_distr_eval_u.subs(my_copul.u, u)
                    eval_next_u = cond_distr_eval_u.subs(my_copul.u, next_u)

                    # CI: decreasing in u, CD: increasing in u
                    if eval_u < eval_next_u:
                        is_cis = False
                    if eval_u > eval_next_u:
                        is_cds = False

                    if not is_cis and not is_cds:
                        break

                if not is_cis and not is_cds:
                    break

        return is_cis, is_cds
