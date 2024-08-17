.. copul documentation master file, created by
   sphinx-quickstart on Thu Aug  8 14:14:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

copul documentation
===================

**copul** is a package designed for mathematical computation and visualization of bivariate copula families.
It accompanies the `Dependence properties of bivariate copula families <https://www.degruyter.com/document/doi/10.1515/demo-2024-0002/html>`_ article released in the Dependence Modeling journal and in particular covers implementations of
the following copulas:

- **Archimedean** copula families including Clayton, Gumbel, Frank, Joe, Ali-Mikhail-Haq, and more
- **Extreme-value** copulas families including Hüsler-Reiss, Galambos, Marshall-Olkin, etc.
- **Elliptical** copula families: Gaussian, Student's t, and Laplace.
- Unclassified copula families like the Farlie-Gumbel-Morgenstern, Plackett or Raftery copula families.
- Independence copula, lower and upper Fréchet bounds
- **Checkerboard** copulas

-------------------

**Perform mathematical computations with copul**

.. highlight:: python
.. code-block:: python

    >>> import copul as cp
    >>> clayton_cdf = cp.Clayton().cdf()
    >>> clayton_cdf
    Max(0, -1 + v**(-theta) + u**(-theta))**(-1/theta)
    >>> clayton_cdf(u=0.5, v=0.5, theta=0.3)
    0.28176656750662277
    >>> cp.AliMikhailHaq(0).pdf()
    1

-------------------

**Generate scatter plots of a copula**

.. highlight:: python
.. code-block:: python

    cp.GenestGhoudi(4).scatter_plot()
    cp.GenestGhoudi(8).scatter_plot()

.. image:: _static/images/GenestGhoudi_scatter_plot_4.png
   :alt: alternate text
   :width: 300px
   :align: left

.. image:: _static/images/GenestGhoudi_scatter_plot_8.png
   :alt: alternate text
   :width: 300px
   :align: right

**Visualize cumulative distribution functions of copulas**

.. highlight:: python
.. code-block:: python

    cp.UpperFrechet().plot_cdf()
    cp.LowerFrechet().plot_cdf()

.. image:: _static/images/UpperFrechet.png
   :alt: alternate text
   :width: 300px
   :align: left

.. image:: _static/images/LowerFrechet.png
   :alt: alternate text
   :width: 280px
   :align: right

.. raw:: html

   <br style="clear: both;"><br>

**Visualize conditional distributions of copulas**::

    plackett = cp.Plackett(0.1)
    plackett.plot(plackett.cond_distr_1, plackett.cond_distr_2)

.. image:: _static/images/cond_distr_pl_1.png
   :alt: alternate text
   :width: 320px
   :align: left

.. image:: _static/images/cond_distr_pl_2.png
    :alt: alternate text
    :width: 320px
    :align: right

**Visualize probability density functions of copulas**

.. highlight:: python
.. code-block:: python

    cp.HueslerReiss(0.3).plot_pdf()

.. image:: _static/images/hr_delta0point3_pdf.png
    :alt: alternate text
    :width: 500px
    :align: center

-------------------

**Use matrices for Checkerboard copulas**

Checkerboard copulas are copulas that have probability density functions, which are constant on rectangles and can be represented by matrices.

.. highlight:: python
.. code-block:: python

    >>> import copul as cp
    >>> matr = [[0, 9, 1], [1, 0, 9], [9, 1, 0]]
    >>> ccop = cp.CheckerboardCopula(matr)
    >>> ccop.cdf(0.2, 1)
    0.2
    >>> ccop.pdf(0.2, 1)
    0.03333333333333333
    >>> ccop.scatter_plot()

.. image:: _static/images/Checkerboard_scatter_plot.png
   :alt: alternate text
   :width: 400px
   :align: center

.. highlight:: python
.. code-block:: python

    >>> ccop.plot_pdf()

.. image:: _static/images/Checkerboard_pdf.png
    :alt: alternate text
    :width: 500px
    :align: center

-------------------

**Plot generator and inverse generator functions for Archimedean copulas**

Archimedean copulas are characterized by a generator function, which is available in the package.

.. highlight:: python
.. code-block:: python

    >>> nelsen7 = cp.Nelsen7()
    >>> nelsen7.generator
    -log(t*theta - theta + 1)
    >>> cp.Nelsen7(0.5).plot_generator()

.. image:: _static/images/Nelsen7_generator.png
    :alt: alternate text
    :width: 500px
    :align: center

**Plot pickands dependence functions for extreme-value copulas**

Extreme-value copulas are characterized by a pickands dependence function, which is also available in the package.

.. highlight:: python
.. code-block:: python

    >>> galambos = copul.Galambos()
    >>> galambos.pickands
    1 - 1/((1 - t)**(-delta) + t**(-delta))**(1/delta)
    >>> galambos.plot_pickands(delta=[0.5, 1, 2])

.. image:: _static/images/Galambos_pickands.png
    :alt: alternate text
    :width: 500px
    :align: center


-------------------

**Plot Spearman's rho, Kendall's tau and Chatterjee's xi**

.. highlight:: python
.. code-block:: python

    cp.Frank().plot_chatterjee(100_000, 30)


.. image:: _static/images/frank_xi_rho_tau.png
    :alt: alternate text
    :width: 500px
    :align: center


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
