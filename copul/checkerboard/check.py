import numpy as np

from copul.foci import codec


class Check:
    def __init__(self, matr):
        """
        Initialize a checkerboard (piecewise-uniform) copula with the given weight matrix.

        Parameters
        ----------
        matr : array-like
            d-dimensional array of nonnegative weights (one per cell).
            They will be normalized so that the total sum is 1.
        """
        # Convert to array
        matr = np.asarray(matr, dtype=float)
        matr_sum = matr.sum()
        if matr_sum > 0:
            self.matr = matr / matr_sum
        else:
            self.matr = np.zeros_like(matr, dtype=float)
        self.dim = matr.ndim

    def __str__(self):
        return f"CheckerboardCopula({self.matr.shape})"

    @classmethod
    def from_data(cls, data, num_bins=None, kappa=1 / 3, **kwargs):
        """
        Create a checkerboard copula from raw data **very quickly** by:
          1. Sorting each dimension to compute ordinal ranks.
          2. Mapping those ranks directly to bin indices.
          3. Building the d-dimensional histogram in a single pass.

        This is an O(d * n log n) algorithm (due to sorting each dimension).
        It avoids building an intermediate uniform array, and it does NOT
        do average ranking for ties (which speeds things up significantly).

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            The raw data. Each row is a sample.
        num_bins : int or list/array of int, optional
            Number of bins in each dimension. If None, defaults to
            ~ n^(1/(2*d)) per dimension.
        kappa : float, optional
            Exponent for the number of bins. Default is 1/3.
            This is a heuristic to control the number of bins
            vs number of observations per bin.
            Recommended range is 1.0 to 2.0.

        Returns
        -------
        Check
            A new Check instance (the histogram is normalized in __init__).
        """
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("Data must have at least one sample and one feature.")

        # Heuristic default for the number of bins
        if num_bins is None:
            # Similar to sqrt rule in each dimension => n^(1/(2*d))
            bin_count = np.floor(n_samples ** (2 * kappa / n_features))
            num_bins = np.full(n_features, bin_count, dtype=int)
        elif isinstance(num_bins, int):
            num_bins = np.full(n_features, num_bins, dtype=int)
        else:
            num_bins = np.asarray(num_bins, dtype=int)
            if len(num_bins) != n_features:
                raise ValueError(
                    f"num_bins must be scalar or have length {n_features}."
                )

        # Prepare array to hold bin indices: (n_samples, n_features)
        bin_indices = np.empty_like(data, dtype=np.int64)

        # 1) For each dimension:
        #    - Sort data to get the ascending order.
        #    - The rank of the smallest point is 0, next is 1, etc. (ordinal rank)
        #    - Convert that rank to a bin index by scale+floor.
        #    - In other words, bin_index = floor( (rank + 1)/(n+1) * num_bins[d] ).
        #      For speed, we usually just do rank / n, ignoring +1.
        #      But if you want to match standard pseudo-obs more closely, include +1.

        for dim in range(n_features):
            order = np.argsort(data[:, dim], kind="quicksort")

            # Ranks go from 0..(n_samples-1)
            # Put them back into bin_indices in their original position
            bin_indices[order, dim] = np.arange(n_samples, dtype=np.int64)

            # Convert ranks to bin indices.
            # Option A: classic pseudo-observations => (rank+1)/(n+1)
            # Option B: simpler => rank/n
            # We'll do Option A to be consistent with typical definitions:
            # bin_index = floor((rank+1)/(n+1) * num_bins[dim])
            bin_indices[:, dim] = np.floor(
                (bin_indices[:, dim] + 1) / (n_samples + 1) * num_bins[dim]
            ).astype(np.int64)

            # Safely clamp at [0, num_bins[dim]-1]
            np.clip(bin_indices[:, dim], 0, num_bins[dim] - 1, out=bin_indices[:, dim])

        # 2) Construct the d-dimensional histogram in a single pass
        hist_shape = tuple(num_bins)
        hist_flat = np.zeros(np.prod(hist_shape), dtype=np.float64)

        # Flatten each row's multi-d bin index into a single integer index
        raveled_idx = np.ravel_multi_index(bin_indices.T, hist_shape)
        # Increment
        np.add.at(hist_flat, raveled_idx, 1)

        # Reshape back to an ND histogram
        hist = hist_flat.reshape(hist_shape)

        # 3) Create the copula object (normalizes in __init__)
        return cls(hist)

    def lambda_L(self):
        """Lower tail dependence (usually 0 for a checkerboard copula)."""
        return 0

    def lambda_U(self):
        """Upper tail dependence (usually 0 for a checkerboard copula)."""
        return 0

    def chatterjees_xi(self, n=100_000, seed=None, i=1, samples=None):
        i0 = i - 1  # Convert to zero-based index
        if samples is None:
            samples = self.rvs(n, random_state=seed)
        x = samples[:, i0]
        # exclude i0-th column
        z = samples[:, np.arange(self.dim) != i0]
        xi = codec(x, z)
        return xi
