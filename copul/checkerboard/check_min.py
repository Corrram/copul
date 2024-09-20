import copy
import itertools

import numpy as np

from copul.checkerboard.check_pi import CheckPi
from copul.exceptions import PropertyUnavailableException


class CheckMin(CheckPi):

    def cdf(self, *args):
        if len(args) != len(self.dim):
            raise ValueError(
                "Number of arguments must be equal to the dimension of the copula"
            )

        indices = []
        overlaps = []

        # Compute the indices and overlaps for each argument
        for i in range(len(args)):
            arg = args[i]
            if arg <= 0:
                return 0  # If the argument is out of bounds, return 0
            elif arg >= 1:  # If the argument exceeds 1, set it to the last index
                indices.append(self.dim[i])
                overlaps.append(0)
            else:
                shape = self.dim[i]
                index = int((arg * shape) // 1)  # Calculate the integer index
                indices.append(index)
                overlap = arg * shape - index  # Calculate the overlap for interpolation
                overlaps.append(overlap)

        # Create slices based on the computed indices
        slices = [slice(i) for i in indices]
        total_integral = self.matr[tuple(slices)].sum()

        # Now we calculate contributions from bordering hypercubes
        for i in range(self.d):
            if overlaps[i] > 0:
                border_slices = copy.deepcopy(slices)
                if indices[i] + 1 < self.dim[i]:  # Ensure we don't go out of bounds
                    border_slices[i] = indices[i]
                    border_contrib = overlaps[i] * self.matr[tuple(border_slices)].sum()
                    total_integral += border_contrib

        # Cross terms for 2D, 3D, ..., up to d-dimensional overlaps
        for r in range(2, self.d + 1):  # Start from 2D interactions up to d-dimensional
            for dims in itertools.combinations(range(self.d), r):
                border_slices = copy.deepcopy(slices)
                overlap_min = 1
                for dim in dims:
                    if overlaps[dim] > 0 and indices[dim] + 1 < self.dim[dim]:
                        border_slices[dim] = indices[dim]
                        overlap_min = min(overlap_min, overlaps[dim])
                    else:
                        overlap_min = (
                            0  # If any dimension does not overlap, skip this term
                        )
                        break
                if overlap_min > 0:
                    total_integral += (
                        overlap_min * self.matr[tuple(border_slices)].sum()
                    )

        return total_integral

    def cond_distr(self, i: int, u):
        if i > len(self.dim):
            msg = "Must condition on a dimension that exists"
            raise ValueError(msg)
        i -= 1
        index = []
        for j in range(self.d):
            idx = int(u[j] * self.dim[j] // 1)
            index.append(idx)
        indices = [slice(j) for j in index]
        indices[i] = index[i]
        total_integral = self.matr[*indices].sum()
        overlap = u[i] * self.matr.shape[1] - index[i]
        adjusted_us = [u[j] - index[j] / self.dim[j] for j in range(self.d)]
        if overlap > 0 and all(adjusted_us[i] <= adjusted_us[j] for j in range(self.d)):
            total_integral += self.matr[*index]
        result = total_integral * self.dim[i]
        return result

    def rvs(self, n=1):
        sel_ele, sel_idx = self._weighted_random_selection(self.matr, n)
        sample = np.random.rand(n)
        samples = [sample / self.dim[i] for i in range(self.d)]
        add_random = np.array(samples).T
        data_points = np.array(
            [[idx[i] / self.dim[i] for i in range(len(self.dim))] for idx in sel_idx]
        )
        data_points += add_random
        return data_points

    @property
    def pdf(self):
        msg = "PDF does not exist for CheckMin"
        raise PropertyUnavailableException(msg)

    def lambda_L(self):
        return 1

    def lambda_U(self):
        return 1
