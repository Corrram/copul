import numpy as np
import sympy
from scipy import stats

from copul.families.elliptical.elliptical_copula import EllipticalCopula

from numpy.linalg import svd
from scipy._lib import doccer

# Imports of constants from _multivariate script
from scipy.stats._multivariate import (
    _LOG_2,
    _doc_random_state,
    multi_rv_generic,
    _PSD,
    _squeeze_output,
    multi_rv_frozen,
)

from copul.sympy_wrapper import SymPyFunctionWrapper

# ==================== START OF THE LOGIC FOR MV LAPLACE =====================

_mvl_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
allow_singular : bool, optional
    Whether to allow a singular covariance matrix.  (Default: False)
"""

_mvl_doc_callparams_note = """Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.
    """

_mvl_doc_frozen_callparams = ""

_mvl_doc_frozen_callparams_note = (
    """See class definition for a detailed description of parameters."""
)

mvl_docdict_params = {
    "_mvl_doc_default_callparams": _mvl_doc_default_callparams,
    "_mvl_doc_callparams_note": _mvl_doc_callparams_note,
    "_doc_random_state": _doc_random_state,
}

mvl_docdict_noparams = {
    "_mvl_doc_default_callparams": _mvl_doc_frozen_callparams,
    "_mvl_doc_callparams_note": _mvl_doc_frozen_callparams_note,
    "_doc_random_state": _doc_random_state,
}


class multivariate_laplace_gen(multi_rv_generic):
    def __init__(self, seed=None):
        super(multivariate_laplace_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvl_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        return multivariate_laplace_frozen(
            mean, cov, allow_singular=allow_singular, seed=seed
        )

    def _process_parameters(self, dim, mean, cov):
        # Try to infer dimensionality
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be " "a scalar.")

        # Check input sizes and return full arrays for mean and cov if
        # necessary
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)

        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)

        if dim == 1:
            mean.shape = (1,)
            cov.shape = (1, 1)

        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." % dim)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = (
                    "Array 'cov' must be square if it is two dimensional,"
                    " but cov.shape = %s." % str(cov.shape)
                )
            else:
                msg = (
                    "Dimension mismatch: array 'cov' is of shape %s,"
                    " but 'mean' is a vector of length %d."
                )
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError(
                "Array 'cov' must be at most two-dimensional,"
                " but cov.ndim = %d" % cov.ndim
            )

        return dim, mean, cov

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.

        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def _logpdf(self, x, mean, prec_U, log_det_cov, rank):
        dev = x - mean
        # maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        maha = np.sum(np.abs(np.dot(dev, prec_U)), axis=-1)

        # return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)
        return -(rank * _LOG_2 + 0.5 * log_det_cov + maha)

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank)
        return _squeeze_output(out)

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank))
        return _squeeze_output(out)

    def _cdf(self, x, mean, prec_U):
        # Dev here is the nominator of expression, equals to: x - mu
        dev = x - mean
        # Equals to: (x - mu) / b
        exp_body = np.sum((np.dot(dev, prec_U)), axis=-1)
        mask_minus = exp_body < 0
        exp_body[mask_minus] *= -1  # equals to np.abs(exp_body) but faster
        # Equals to the CDF for both positive and negative branches
        cdf_val = 0.5 * np.exp(-exp_body)
        # Special analytical case when x - mu < 0
        cdf_val[mask_minus] *= -1
        cdf_val[mask_minus] += 1

        return cdf_val

    def logcdf(self, x, mean=None, cov=1, allow_singular=False):
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.log(self._cdf(x, mean, psd.U))
        return _squeeze_output(out)

    def cdf(self, x, mean=None, cov=1, allow_singular=False):
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = self._cdf(x, mean, psd.U)
        return _squeeze_output(out)

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        # Check preconditions on arguments
        mean = np.array(mean)
        cov = np.array(cov)
        if size is None:
            shape = []
        elif isinstance(size, (int, np.integer)):
            shape = [size]
        else:
            shape = size

        if len(mean.shape) != 1:
            raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")

        # Compute shape of output and create a matrix of independent
        # standard normally distributed random numbers. The matrix has rows
        # with the same length as mean and as many rows are necessary to
        # form a matrix of shape final_shape.
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])
        random_state = self._get_random_state(random_state)
        # Standard laplace
        x = random_state.laplace(loc=0.0, scale=1.0, size=final_shape).reshape(
            -1, mean.shape[0]
        )
        dim, mean, cov = self._process_parameters(None, mean, cov)
        (u, s, v) = svd(cov)

        x = np.dot(x, np.sqrt(s)[:, None] * v)
        x += mean
        return x

    def entropy(self, mean=None, cov=1):
        dim, mean, cov = self._process_parameters(None, mean, cov)
        _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
        return 0.5 * logdet


multivariate_laplace = multivariate_laplace_gen()


class multivariate_laplace_frozen(multi_rv_frozen):
    def __init__(
        self,
        mean=None,
        cov=1,
        allow_singular=False,
        seed=None,
        maxpts=None,
        abseps=1e-5,
        releps=1e-5,
    ):
        self._dist = multivariate_laplace_gen(seed)
        self.dim, self.mean, self.cov = self._dist._process_parameters(None, mean, cov)
        self.cov_info = _PSD(self.cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(
            x, self.mean, self.cov_info.U, self.cov_info.log_pdet, self.cov_info.rank
        )
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        return np.log(self.cdf(x))

    def cdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.mean, self.cov_info.U)
        return _squeeze_output(out)
        return self._dist.cdf(x)
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(
            x, self.mean, self.cov, self.maxpts, self.abseps, self.releps
        )
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.mean, self.cov, size, random_state)

    def entropy(self):
        log_pdet = self.cov_info.log_pdet  # log(rho^2) = 2*log(b)
        rank = self.cov_info.rank
        return rank * (_LOG_2 + 1) + 0.5 * log_pdet  # = log(2*b*e)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_laplace_gen and fill in default strings in class docstrings
for name in ["logpdf", "pdf", "logcdf", "cdf", "rvs"]:
    method = multivariate_laplace_gen.__dict__[name]
    method_frozen = multivariate_laplace_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__, mvl_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvl_docdict_params)

# =============================================================================


class Laplace(EllipticalCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def rvs(self, n=1):
        mu = [0, 0]
        cov = np.array(self.corr_matrix, dtype=float)
        samples = multivariate_laplace.rvs(mean=mu, cov=cov, size=n)
        u1 = stats.laplace.cdf(samples[:, 0])
        u2 = stats.laplace.cdf(samples[:, 1])
        return np.array([u1, u2]).T

    def cdf(self) -> SymPyFunctionWrapper:
        pass

    def pdf(self) -> SymPyFunctionWrapper:
        pass
