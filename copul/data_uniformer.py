import numpy as np
from scipy import stats


class DataUniformer:
    """Class to transform data to uniform margins using empirical CDF.

    Transforms multivariate data to have uniform margins on [0,1] by
    converting values to ranks and then scaling appropriately.
    """

    def __init__(self):
        pass

    def uniform(self, data):
        """Transform data to uniform margins (ranks scaled to [0,1]).

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape (n_samples, n_features) to be transformed

        Returns
        -------
        numpy.ndarray
            Transformed data with values in [0,1] for each feature
        """
        # Ensure data is a numpy array
        data = np.asarray(data, dtype=np.float64)

        # Fast path for 1D arrays
        if data.ndim == 1:
            return self._transform_column(data)

        # Multi-dimensional case
        n_samples, n_features = data.shape

        # Preallocate output array - slightly faster than zeros_like
        transformed_data = np.empty_like(data)

        # For large datasets, use parallel processing
        if n_samples * n_features > 100000 and n_features > 1:
            try:
                from joblib import Parallel, delayed

                # Process columns in parallel
                results = Parallel(n_jobs=-1)(
                    delayed(self._transform_column)(data[:, j])
                    for j in range(n_features)
                )
                for j, result in enumerate(results):
                    transformed_data[:, j] = result
                return transformed_data
            except ImportError:
                # Fall back to serial processing if joblib not available
                pass

        # Standard serial processing
        for j in range(n_features):
            transformed_data[:, j] = self._transform_column(data[:, j])

        return transformed_data

    def _transform_column(self, column):
        """Transform a single column to uniform margins.

        Parameters
        ----------
        column : numpy.ndarray
            1D array to transform

        Returns
        -------
        numpy.ndarray
            Transformed column with values in [0,1]
        """
        n_samples = len(column)

        # Use 'average' method for ties to get more accurate copula representation
        # This is better than 'ordinal' for statistical properties
        ranks = stats.rankdata(column, method="average")

        # Scale ranks to [0,1] - division by (n+1) ensures values are inside (0,1)
        # and not exactly at the boundaries, which is important for copulas
        return ranks / (n_samples + 1)
