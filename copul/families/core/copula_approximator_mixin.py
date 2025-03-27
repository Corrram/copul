from copul import Checkerboarder


class CopulaApproximatorMixin:
    def to_check_pi(self, grid_size: tuple | int = 100):
        """
        Convert the copula to a CheckPi object.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
        checkerboard_type : str, optional
            Type of checkerboard copula (default is "BivCheckPi").

        Returns
        -------
        CheckPi
            A CheckPi object representing the copula.
        """
        return self.to_checkerboard(grid_size, checkerboard_type="CheckPi")

    def to_check_min(self, grid_size: tuple | int = 100):
        """
        Convert the copula to a CheckMin object.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
        checkerboard_type : str, optional
            Type of checkerboard copula (default is "BivCheckMin").

        Returns
        -------
        CheckMin
            A CheckMin object representing the copula.
        """
        return self.to_checkerboard(grid_size, checkerboard_type="CheckMin")

    def to_check_w(self, grid_size: tuple | int = 100):
        """
        Convert the copula to a CheckW object.

        Parameters
        ----------
        grid_size : tuple or int, optional
            Size of the grid for the checkerboard (default is 100).
        checkerboard_type : str, optional
            Type of checkerboard copula (default is "BivCheckW").

        Returns
        -------
        CheckW
            A CheckW object representing the copula.
        """
        return self.to_checkerboard(grid_size, checkerboard_type="CheckW")

    def to_checkerboard(
        self, grid_size: tuple | int = 100, checkerboard_type: str = "BivCheckPi"
    ):
        checkerboarder = Checkerboarder(grid_size, self.dim, checkerboard_type)
        return checkerboarder.get_checkerboard_copula(self)
