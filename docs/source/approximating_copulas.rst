Approximating Copulas
===========================

This page demonstrates how to approximate and visualize copulas using the `copul` package through simple examples and visualizations.

.. highlight:: bash
.. code-block:: bash

    pip install copul

.. highlight:: python
.. code-block:: python

    >>> import copul as cp



**Bernstein Copula Approximation**

We approximate a Gaussian copula using Bernstein copulas of sizes :math:`3 \times 3` and :math:`20 \times 20`.

.. code-block:: python

    gauss = cp.Gaussian(0.5)
    gauss.plot_pdf()
    gauss.to_bernstein(3).plot_pdf()
    gauss.to_bernstein(20).plot_pdf()

.. image:: _static/images/gauss_cop.png
   :alt: Gaussian copula
   :width: 220px

.. image:: _static/images/gauss_bernstein_3.png
   :alt: Bernstein (3x3) approximation
   :width: 220px

.. image:: _static/images/gauss_bernstein_20.png
   :alt: Bernstein (20x20) approximation
   :width: 220px



**Checkerboard Copula Approximation**

Approximate a Clayton copula using a :math:`5 \times 5` checkerboard copula.

.. code-block:: python

    clayton = cp.Clayton(1)
    clayton.plot_pdf(zlim=(0, 3))
    clayton.to_checkerboard(5).plot_pdf()

.. image:: _static/images/clayton_1.png
   :alt: Clayton copula (theta=1)
   :width: 320px

.. image:: _static/images/clayton_checkerboard_5.png
   :alt: Checkerboard (5x5) approximation
   :width: 320px



**Check--Min vs Checkerboard**

Compare scatter plots of samples from a check--min copula and a checkerboard copula constructed from the same matrix.

.. code-block:: python

    matr = [[0, 1, 9], [1, 9, 0], [9, 0, 1]]
    cp.BivCheckPi(matr).scatter_plot(2_000)
    cp.BivCheckMin(matr).scatter_plot(2_000)

.. image:: _static/images/checkerboard_scatter.png
   :alt: BivCheckPi sample
   :width: 300px

.. image:: _static/images/check_min_scatter.png
   :alt: BivCheckMin sample
   :width: 300px

**Estimation of Chatterjee's xi**

Estimate Chatterjee's xi using checkerboard and check--min approximations for the Ali-Mikhail-Haq copula with parameter 0.8.

.. code-block:: python

    import matplotlib.pyplot as plt

    amh = cp.AliMikhailHaq(0.8)
    grid_sizes = range(3, 10)

    plt.plot(grid_sizes, [amh.to_check_pi(i).xi() for i in grid_sizes], label="CheckPi xi")
    plt.plot(grid_sizes, [amh.to_check_min(i).xi() for i in grid_sizes], label="CheckMin xi")
    plt.axhline(y=amh.xi(), color='r', linestyle='--', label="True xi")
    plt.legend()
    plt.grid()
    plt.show()

.. image:: _static/images/amh_xi.png
   :alt: Xi estimation plot
   :width: 500px



**Shuffle-of-Min Copula**

Visualize a shuffle-of-min copula defined by a permutation of the first 10 integers.

.. code-block:: python

    perm = (8, 4, 2, 1, 3, 9, 5, 10, 6, 7)
    cp.ShuffleOfMin(perm).scatter_plot(2_000)
    cp.ShuffleOfMin(perm).plot_cdf()

.. image:: _static/images/shuffle_of_min_10.png
   :alt: Shuffle-of-min sample
   :width: 300px

.. image:: _static/images/shuffle_of_min_cdf_10.png
   :alt: Shuffle-of-min CDF
   :width: 300px