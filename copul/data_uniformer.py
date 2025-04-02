import numpy as np
from scipy import stats


class DataUniformer:
    """Class to transform data to uniform margins using empirical CDF.

    Transforms multivariate data to have uniform margins on [0,1] by
    converting values to ranks and then scaling appropriately.
    """

    def __init__(self):
        pass

    def uniform(self, data, touch_boundaries=False):
        """Transform data to uniform margins (ranks scaled to [0,1]).

        Parameters
        ----------
        data : numpy.ndarray or list
            Array of shape (n_samples, n_features) to be transformed
        touch_boundaries : bool, optional
            If False (default), the transformed values lie strictly in (0,1).
            If True, the transformed values will exactly include 0.0 and 1.0
            for the min and max of each column.

        Returns
        -------
        numpy.ndarray
            Transformed data with values in [0,1].
        """
        # Ensure data is a numpy array
        data = np.asarray(data, dtype=np.float64)

        # Fast path for 1D arrays
        if data.ndim == 1:
            return self._transform_column(data, touch_boundaries)

        # Multi-dimensional case
        n_samples, n_features = data.shape

        # Preallocate output array
        transformed_data = np.empty_like(data)

        # For large datasets, use parallel processing (optional)
        if n_samples * n_features > 100000 and n_features > 1:
            try:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=-1)(
                    delayed(self._transform_column)(data[:, j], touch_boundaries)
                    for j in range(n_features)
                )
                for j, result in enumerate(results):
                    transformed_data[:, j] = result
                return transformed_data
            except ImportError:
                pass  # fallback to serial

        # Serial transformation
        for j in range(n_features):
            transformed_data[:, j] = self._transform_column(data[:, j], touch_boundaries)

        return transformed_data

    def _transform_column(self, column, touch_boundaries=False):
        """Transform a single column to uniform margins.

        Parameters
        ----------
        column : numpy.ndarray (1D)
            Column to transform
        touch_boundaries : bool
            See `uniform` docstring

        Returns
        -------
        numpy.ndarray
            Transformed column with values in [0,1].
        """
        n_samples = len(column)

        # If there's only one sample, choose a sensible default
        if n_samples == 1:
            if touch_boundaries:
                # Could choose 0.0 or 1.0, but 0.0 is typical to "touch" the boundary
                return np.array([0.0], dtype=np.float64)
            else:
                # Typically we stay in (0,1), 0.5 is a neutral midpoint
                return np.array([0.5], dtype=np.float64)

        # Compute ranks using 'average' to handle ties gracefully
        ranks = stats.rankdata(column, method="average")

        if touch_boundaries:
            # Map ranks to [0,1]: 
            # smallest rank (1) -> 0.0, largest rank (n_samples) -> 1.0
            return (ranks - 1.0) / (n_samples - 1.0)
        else:
            # Original behavior: map ranks to (0,1) => (1/(n+1), ..., n/(n+1))
            return ranks / (n_samples + 1.0)
