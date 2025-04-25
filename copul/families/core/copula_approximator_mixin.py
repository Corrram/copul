import numpy as np
from typing import Sequence, TypeAlias, Union, Tuple

# Assuming the provided ShuffleOfMin class is in a file named shuffle_of_min.py
# Adjust the import path if necessary
from copul.checkerboard.shuffle_min import ShuffleOfMin
from copul.checkerboard.checkerboarder import Checkerboarder


class CopulaApproximatorMixin:
    """
    A mixin class providing methods to approximate a copula using various
    checkerboard-based structures.
    """
    # Assume the class this mixin is added to has a 'dim' attribute
    # representing the dimensionality of the copula.
    dim: int

    def to_check_pi(self, grid_size: Union[Tuple[int, ...], int] = 100):
        """
        Convert the copula to a CheckPi object.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
            If int, assumes equal size in all dimensions.

        Returns
        -------
        CheckPi
            A CheckPi object representing the copula approximation.
        """
        # Assuming CheckPi corresponds to BivCheckPi or a generalized version
        # handled by Checkerboarder based on self.dim
        checkerboard_type = "CheckPi" # Keep it generic, let Checkerboarder decide Biv/Multi
        if self.dim == 2:
             checkerboard_type = "BivCheckPi"
        # Add logic here if a multivariate CheckPi exists and needs a different type name

        return self.to_checkerboard(grid_size, checkerboard_type=checkerboard_type)

    def to_check_min(self, grid_size: Union[Tuple[int, ...], int] = 100):
        """
        Convert the copula to a CheckMin object.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
            If int, assumes equal size in all dimensions.


        Returns
        -------
        CheckMin
            A CheckMin object representing the copula approximation.
        """
        # Assuming CheckMin corresponds to BivCheckMin or a generalized version
        checkerboard_type = "CheckMin"
        if self.dim == 2:
            checkerboard_type = "BivCheckMin"
        # Add logic here if a multivariate CheckMin exists

        return self.to_checkerboard(grid_size, checkerboard_type=checkerboard_type)

    def to_check_w(self, grid_size: Union[Tuple[int, ...], int] = 100):
        """
        Convert the copula to a CheckW object.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
            If int, assumes equal size in all dimensions.

        Returns
        -------
        CheckW
            A CheckW object representing the copula approximation.
        """
        # Assuming CheckW corresponds to BivCheckW or a generalized version
        checkerboard_type = "CheckW"
        if self.dim == 2:
            checkerboard_type = "BivCheckW"
        # Add logic here if a multivariate CheckW exists

        return self.to_checkerboard(grid_size, checkerboard_type=checkerboard_type)

    def to_checkerboard(
        self, grid_size: Union[Tuple[int, ...], int] = 100, checkerboard_type: str = "BivCheckPi"
    ):
        """
        Generic method to convert the copula to a specified checkerboard type.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
            If int, assumes equal size n, resulting in an n x n x ... grid.
        checkerboard_type : str, optional
            Type of checkerboard copula (default is "BivCheckPi"). This string
            is passed to the Checkerboarder to determine which copula type
            to instantiate and return.

        Returns
        -------
        object
            An instance of the specified checkerboard copula type, approximating
            the original copula.
        """
        # Ensure self has a 'dim' attribute
        if not hasattr(self, 'dim'):
            raise AttributeError("The object using CopulaApproximatorMixin must have a 'dim' attribute.")

        checkerboarder = Checkerboarder(grid_size, self.dim, checkerboard_type)
        # The checkerboarder uses the original copula (self) to calculate
        # the parameters for the approximating checkerboard copula.
        return checkerboarder.get_checkerboard_copula(self)

    def to_bernstein(self, grid_size: Union[Tuple[int, ...], int] = 10):
        """
        Convert the copula to a Bernstein copula approximation.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Order of the Bernstein polynomial basis (default is 10).
            If int, assumes equal order m in all dimensions.

        Returns
        -------
        BernsteinCopula
            A BernsteinCopula object representing the copula approximation.
        """
        # Bernstein copula is another type handled by the checkerboard framework
        return self.to_checkerboard(grid_size, checkerboard_type="Bernstein")

    def to_shuffle_of_min(self, grid_size: int = 100) -> ShuffleOfMin:
        """
        Approximate the copula using a ShuffleOfMin copula.

        This method uses the checkerboard framework to find the permutation 'pi'
        for a ShuffleOfMin copula of order 'n' (determined by grid_size) that
        best approximates the mass distribution of the original bivariate copula.

        Parameters
        ----------
        grid_size : int, optional
            The order 'n' of the resulting ShuffleOfMin copula, which defines
            the number of segments used in the approximation (default is 100).
            This corresponds to an n x n grid for the underlying checkerboard calculation.

        Returns
        -------
        ShuffleOfMin
            A ShuffleOfMin object approximating the original copula.

        Raises
        ------
        ValueError
            If the original copula is not bivariate (self.dim != 2), as
            ShuffleOfMin is defined for two dimensions.
        AttributeError
            If the object using the mixin doesn't have a 'dim' attribute.
        """
        # Ensure self has a 'dim' attribute before checking its value
        if not hasattr(self, 'dim'):
            raise AttributeError("The object using CopulaApproximatorMixin must have a 'dim' attribute.")

        # ShuffleOfMin is strictly bivariate based on the provided class definition
        if self.dim != 2:
            raise ValueError(f"ShuffleOfMin approximation requires a bivariate copula (dim=2), but got dim={self.dim}.")

        # Pass grid_size (which is 'n' for ShuffleOfMin) and the type string.
        # The Checkerboarder implementation is assumed to know how to handle
        # the "ShuffleOfMin" type, likely using the grid_size as the order 'n'
        # and determining the optimal permutation 'pi'.
        # Note: ShuffleOfMin only needs a single integer 'n' for its order,
        # which corresponds to an n x n checkerboard grid.
        return self.to_checkerboard(grid_size, checkerboard_type="ShuffleOfMin")
