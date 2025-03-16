import copy
import sympy
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from copul.copula_sampler import CopulaSampler
from copul.families.copula_graphs import CopulaGraphs
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Copula:
    """
    A unified Copula class that combines functionality previously split between
    CoreCopula and Copula classes.
    """
    params = []
    intervals = {}
    log_cut_off = 4
    _cdf = None
    _free_symbols = {}

    def __str__(self):
        return self.__class__.__name__

    def __init__(self, dimension, *args, **kwargs):
        """
        Initialize a Copula.

        Parameters
        ----------
        dimension : int
            Dimension of the copula.
        *args : tuple
            Positional arguments for parameters.
        **kwargs : dict
            Keyword arguments for parameters.
        """
        self.u_symbols = sympy.symbols(f"u1:{dimension + 1}")
        self.dim = dimension
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(self, k, v)
        self.params = [param for param in self.params if str(param) not in kwargs]
        self.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }

    def __call__(self, *args, **kwargs):
        """
        Create a new Copula instance with updated parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for parameters.
        **kwargs : dict
            Keyword arguments for parameters.

        Returns
        -------
        Copula
            A new Copula instance with the updated parameters.
        """
        new_copula = copy.copy(self)
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(new_copula, k, v)
        new_copula.params = [param for param in self.params if str(param) not in kwargs]
        new_copula.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }
        return new_copula

    def _set_params(self, args, kwargs):
        """
        Set parameters from args and kwargs.

        Parameters
        ----------
        args : tuple
            Positional arguments for parameters.
        kwargs : dict
            Keyword arguments for parameters.
        """
        if args and len(args) <= len(self.params):
            for i in range(len(args)):
                kwargs[str(self.params[i])] = args[i]
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    @property
    def parameters(self):
        """
        Get the parameters of the copula.

        Returns
        -------
        dict
            Dictionary of parameter intervals.
        """
        return self.intervals

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True if the copula is absolutely continuous, False otherwise.
        """
        # Implementations should override this method
        raise NotImplementedError("This method must be implemented in a subclass")

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric.

        Returns
        -------
        bool
            True if the copula is symmetric, False otherwise.
        """
        # Implementations should override this method
        raise NotImplementedError("This method must be implemented in a subclass")

    def _are_class_vars(self, kwargs):
        """
        Check if keys in kwargs are class variables.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to check.

        Raises
        ------
        AssertionError
            If any key in kwargs is not a class variable.
        """
        class_vars = set(dir(self))
        assert set(kwargs).issubset(
            class_vars
        ), f"keys: {set(kwargs)}, free symbols: {class_vars}"

    def slice_interval(self, param, interval_start=None, interval_end=None):
        """
        Slice the interval of a parameter.

        Parameters
        ----------
        param : str or sympy.Symbol
            The parameter to slice.
        interval_start : float, optional
            Start of the interval.
        interval_end : float, optional
            End of the interval.
        """
        if not isinstance(param, str):
            param = str(param)
        left_open = self.intervals[param].left_open
        right_open = self.intervals[param].right_open
        if interval_start is None:
            interval_start = self.intervals[param].inf
        else:
            left_open = False
        if interval_end is None:
            interval_end = self.intervals[param].sup
        else:
            right_open = False
        self.intervals[param] = sympy.Interval(
            interval_start, interval_end, left_open, right_open
        )

    @property
    def cdf(self, *args, **kwargs):
        """
        Get the cumulative distribution function of the copula.

        Returns
        -------
        CDFWrapper
            Wrapper for the CDF.
        """
        expr = self._cdf
        for key, value in self._free_symbols.items():
            expr = expr.subs(value, getattr(self, key))
        return CDFWrapper(expr)(*args, **kwargs)

    def cond_distr(self, i, u=None):
        """
        Get the conditional distribution of the i-th variable.

        Parameters
        ----------
        i : int
            Index of the variable (1-based).
        u : array_like, optional
            Values at which to evaluate the conditional distribution.

        Returns
        -------
        SymPyFuncWrapper or float
            The conditional distribution function or its value at u.
        """
        assert i in range(1, self.dim + 1)
        result = SymPyFuncWrapper(sympy.diff(self.cdf, self.u_symbols[i - 1]))
        if u is None:
            return result
        return result(*u)

    def cond_distr_1(self, u=None):
        """
        Get the conditional distribution of the first variable.

        Parameters
        ----------
        u : array_like, optional
            Values at which to evaluate the conditional distribution.

        Returns
        -------
        SymPyFuncWrapper or float
            The conditional distribution function or its value at u.
        """
        result = SymPyFuncWrapper(sympy.diff(self.cdf, self.u_symbols[0]))
        if u is None:
            return result
        return result(*u)

    def cond_distr_2(self, u=None):
        """
        Get the conditional distribution of the second variable.

        Parameters
        ----------
        u : array_like, optional
            Values at which to evaluate the conditional distribution.

        Returns
        -------
        SymPyFuncWrapper or float
            The conditional distribution function or its value at u.
        """
        result = SymPyFuncWrapper(sympy.diff(self.cdf, self.u_symbols[1]))
        if u is None:
            return result
        return result(*u)

    def pdf(self, u=None):
        """
        Get the probability density function of the copula.

        Parameters
        ----------
        u : array_like, optional
            Values at which to evaluate the PDF.

        Returns
        -------
        SymPyFuncWrapper or float
            The PDF function or its value at u.
        """
        term = self.cdf
        for u_symbol in self.u_symbols:
            term = sympy.diff(term, u_symbol)
        pdf = SymPyFuncWrapper(term)
        return pdf(u) if u is not None else pdf

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
            Whether to use approximate sampling.

        Returns
        -------
        np.ndarray
            An array of shape (n, dim) containing samples from the copula.
        """
        sampler = CopulaSampler(self, random_state=random_state)
        return sampler.rvs(n, approximate)

    def scatter_plot(
        self, n=1_000, approximate=False, figsize=(10, 8), alpha=0.6, colormap="viridis"
    ):
        """
        Create a scatter plot of random variates from the copula.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate (default is 1,000).
        approximate : bool, optional
            Whether to use explicit sampling from the conditional distributions or
            approximate sampling with a checkerboard copula
        figsize : tuple, optional
            Figure size as (width, height) in inches.
        alpha : float, optional
            Transparency of points (0 to 1).
        colormap : str, optional
            Colormap to use for 3D plots.

        Returns
        -------
        None
        """
        if self.dim == 2:
            data_ = self.rvs(n, approximate=approximate)
            plt.figure(figsize=figsize)
            plt.scatter(data_[:, 0], data_[:, 1], s=rcParams["lines.markersize"] ** 2)
            title = CopulaGraphs(self).get_copula_title()
            plt.title(title)
            plt.xlabel("u")
            plt.ylabel("v")
            plt.grid(True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
            plt.close()
        elif self.dim == 3:
            # Generate samples
            data = self.rvs(n, approximate=approximate)

            # Create 3D figure and axes
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            # Create color mapping based on the third dimension for better visualization
            colors = data[:, 2]

            # Plot the 3D scatter points
            scatter = ax.scatter(
                data[:, 0],  # x-coordinates (first margin)
                data[:, 1],  # y-coordinates (second margin)
                data[:, 2],  # z-coordinates (third margin)
                c=colors,  # color by third dimension
                cmap=colormap,
                s=rcParams["lines.markersize"] ** 2,
                alpha=alpha,
            )

            # Add a color bar to show the mapping
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label("w value")

            # Set title and labels
            title = CopulaGraphs(self).get_copula_title()
            ax.set_title(title)
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_zlabel("w")

            # Set axis limits to the copula domain [0,1]³
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

            # Add gridlines
            ax.grid(True)

            # Add a view angle that shows the 3D structure well
            ax.view_init(elev=30, azim=45)

            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            # For higher dimensions, display scatter plot matrix
            data = self.rvs(n, approximate=approximate)

            # Create scatter plot matrix
            fig, axs = plt.subplots(
                self.dim,
                self.dim,
                figsize=(3 * self.dim, 3 * self.dim),
            )

            # Get copula title
            title = CopulaGraphs(self).get_copula_title()
            fig.suptitle(title, fontsize=16)

            # Variable names
            var_names = [f"u{i+1}" for i in range(self.dim)]

            # Fill the scatter plot matrix
            for i in range(self.dim):
                for j in range(self.dim):
                    if i == j:
                        # Histogram on the diagonal
                        axs[i, j].hist(data[:, i], bins=20, alpha=0.7)
                    else:
                        # Scatter plot on off-diagonal
                        axs[i, j].scatter(
                            data[:, j],
                            data[:, i],
                            s=rcParams["lines.markersize"],
                            alpha=0.5,
                        )

                    # Set labels only on the outer plots
                    if i == self.dim - 1:
                        axs[i, j].set_xlabel(var_names[j])
                    if j == 0:
                        axs[i, j].set_ylabel(var_names[i])

                    # Set limits
                    axs[i, j].set_xlim(0, 1)
                    axs[i, j].set_ylim(0, 1)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
            plt.show()
            plt.close()