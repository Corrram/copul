import matplotlib

from copul import Nelsen2
from copul.schur_order.schur_visualizer import visualize_rearranged

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_visualize_rearranged_n2():
    visualize_rearranged(Nelsen2, [1.4, 2], 0.1)
