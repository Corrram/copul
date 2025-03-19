.. copul documentation master file, created by
   sphinx-quickstart on Thu Aug  8 14:14:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

copul documentation
===================

**copul** is a package designed for mathematical computation and visualization of bivariate copula families.
It accompanies the `Dependence properties of bivariate copula families <https://www.degruyter.com/document/doi/10.1515/demo-2024-0002/html>`_ article released in the Dependence Modeling journal and in particular covers implementations of 35+ copula families, see :doc:`overview` for an overview.

.. - **Archimedean** copula families: Clayton, Ali-Mikhail-Haq, Gumbel-Hougaard, etc.
   - **Extreme-value** copula families: Galambos, Marshall-Olkin, ...
   - **Elliptical** copula families: Gaussian, Student's t, or Laplace.
   - Unclassified copula families: Farlie-Gumbel-Morgenstern, Plackett, Raftery, etc.

   For a full reference list of implemented copula families, their properties, and available methods, see :doc:`overview`.

-------------------

**Import the package**

.. highlight:: python
.. code-block:: python

    >>> import copul as cp

**List callable copula families**

.. highlight:: python
.. code-block:: python

   >>> cp.families
   ['Clayton', 'Nelsen2', 'AliMikhailHaq', 'GumbelHougaard', 'Frank', 'Joe', 'Nelsen7', 'Nelsen8', 'GumbelBarnett', 'Nelsen10', 'Nelsen11', 'Nelsen12', 'Nelsen13', 'Nelsen14', 'GenestGhoudi', 'Nelsen16', 'Nelsen17', 'Nelsen18', 'Nelsen19', 'Nelsen20', 'Nelsen21', 'Nelsen22', 'JoeEV', 'BB5', 'CuadrasAuge', 'Galambos', 'GumbelHougaard', 'HueslerReiss', 'Tawn', 'tEV', 'MarshallOlkin', 'Gaussian', 'StudentT', 'Laplace', 'B11', 'CheckerboardCopula', 'FarlieGumbelMorgenstern', 'Frechet', 'IndependenceCopula', 'LowerFrechet', 'Mardia', 'Plackett', 'Raftery', 'UpperFrechet']


**Call a copula family and view its parameters**

.. highlight:: python
.. code-block:: python

   >>> clayton = cp.Clayton()
   >>> cp.Clayton().parameters
   {'theta': Interval(-1, oo)}
   >>> cp.MarshallOlkin().parameters
   {'alpha_1': Interval(0, 1), 'alpha_2': Interval(0, 1)}

-------------------

**Simulate data points from a copula**

.. highlight:: python
.. code-block:: python

    >>> cp.Clayton(0.5).rvs(n=3)
    array([[0.33620426, 0.34329421],
           [0.34242024, 0.21372513],
           [0.5785887 , 0.94612088]])

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

.. raw:: html

   <br style="clear: both;"><br>

-------------------

**Cumlative distribution functions**

.. highlight:: python
.. code-block:: python

    >>> cp.Clayton().cdf  # cumulative distribution function
    Max(0, -1 + v**(-theta) + u**(-theta))**(-1/theta)
    >>> cp.Clayton(theta=0.5).cdf  # specify a parameter
    Max(0, u**(-0.5) + v**(-0.5) - 1)**(-2.0)
    >>> cp.Clayton(0.5).cdf  # the parameter can be passed as a positional argument
    Max(0, u**(-0.5) + v**(-0.5) - 1)**(-2.0)
    >>> cp.Clayton(0.5).cdf(0.3, 1)
    0.

.. highlight:: python
.. code-block:: python

    cp.UpperFrechet().plot_cdf()
    cp.LowerFrechet().plot_cdf()

.. image:: _static/images/LowerFrechet.png
   :alt: alternate text
   :width: 280px
   :align: left

.. image:: _static/images/UpperFrechet.png
   :alt: alternate text
   :width: 300px
   :align: right


.. raw:: html

   <br style="clear: both;"><br>

**Visualize conditional distributions of copulas**

.. highlight:: python
.. code-block:: python

   >>> cp.Plackett().cond_distr_1()
   (theta - (-2*theta*v*(theta - 1) + (2*theta - 2)*((theta - 1)*(u + v) + 1)/2)/sqrt(-4*theta*u*v*(theta - 1) + ((theta - 1)*(u + v) + 1)**2) - 1)/(2*theta - 2)
   >>> plackett = cp.Plackett(0.1)
   >>> plackett.plot(plackett.cond_distr_1, plackett.cond_distr_2)

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

   >>> cp.HueslerReiss(0.3).pdf
   (u*v)**(-(log(v)/log(u*v) - 1)*(erf(sqrt(2)*(0.15*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 3.33333333333333)/2)/2 + 1/2) + (erf(sqrt(2)*(0.15*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 3.33333333333333)/2)/2 + 1/2)*log(v)/log(u*v))*((-((log(v)/log(u*v) - 1)*(erf(sqrt(2)*(0.15*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 3.33333333333333)/2)/2 + 1/2) - (erf(sqrt(2)*(0.15*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 3.33333333333333)/2)/2 + 1/2)*log(v)/log(u*v))*log(u*v) - (log(v) - log(u*v))*(-0.075*sqrt(2)*(log(v)/log(u*v) - 1)*(-1/(log(v)/log(u*v) - 1) + log(v)/((log(v)/log(u*v) - 1)**2*log(u*v)))*exp(-5.55555555555556*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)**2)/sqrt(pi) + 0.075*sqrt(2)*((log(v)/log(u*v) - 1)*log(u*v)**2/log(v)**2 - log(u*v)/log(v))*exp(-5.55555555555556*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)**2)*log(v)/(sqrt(pi)*log(u*v)) + erf(sqrt(2)*(0.15*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 3.33333333333333)/2)/2 - erf(sqrt(2)*(0.15*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 3.33333333333333)/2)/2))*(-((log(v)/log(u*v) - 1)*(erf(sqrt(2)*(0.15*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 3.33333333333333)/2)/2 + 1/2) - (erf(sqrt(2)*(0.15*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 3.33333333333333)/2)/2 + 1/2)*log(v)/log(u*v))*log(u*v) - (-0.075*sqrt(2)*(log(v)/log(u*v) - 1)*(-1/(log(v)/log(u*v) - 1) + log(v)/((log(v)/log(u*v) - 1)**2*log(u*v)))*exp(-5.55555555555556*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)**2)/sqrt(pi) + 0.075*sqrt(2)*((log(v)/log(u*v) - 1)*log(u*v)**2/log(v)**2 - log(u*v)/log(v))*exp(-5.55555555555556*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)**2)*log(v)/(sqrt(pi)*log(u*v)) + erf(sqrt(2)*(0.15*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 3.33333333333333)/2)/2 - erf(sqrt(2)*(0.15*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 3.33333333333333)/2)/2)*log(v))*log(u*v) + sqrt(2)*(log(v) - log(u*v))*(-0.0375*(-1 + log(v)/((log(v)/log(u*v) - 1)*log(u*v)))**2*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)*exp(-5.55555555555556*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)**2)*log(u*v)/log(v) + (-0.15 + 0.15*log(v)/((log(v)/log(u*v) - 1)*log(u*v)))*exp(-5.55555555555556*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)**2)/(log(v)/log(u*v) - 1) - (-0.075 + 0.075*log(v)/((log(v)/log(u*v) - 1)*log(u*v)))*exp(-5.55555555555556*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)**2)*log(u*v)/log(v) - (-0.075 + 0.075*log(v)/((log(v)/log(u*v) - 1)*log(u*v)))*exp(-5.55555555555556*(0.045*log(-log(v)/((log(v)/log(u*v) - 1)*log(u*v))) + 1)**2)/(log(v)/log(u*v) - 1) + (-0.15*(log(v)/log(u*v) - 1)*log(u*v)/log(v) + 0.15)*exp(-5.55555555555556*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)**2)*log(u*v)/log(v) - (-0.075*(log(v)/log(u*v) - 1)*log(u*v)/log(v) + 0.075)*exp(-5.55555555555556*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)**2)*log(u*v)/log(v) + 0.0375*(-(log(v)/log(u*v) - 1)*log(u*v)/log(v) + 1)**2*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)*exp(-5.55555555555556*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)**2)/(log(v)/log(u*v) - 1) - (-0.075*(log(v)/log(u*v) - 1)*log(u*v)/log(v) + 0.075)*exp(-5.55555555555556*(0.045*log(-(log(v)/log(u*v) - 1)*log(u*v)/log(v)) + 1)**2)/(log(v)/log(u*v) - 1))*log(v)/sqrt(pi))/(u*v*log(u*v)**3)
   >>> cp.HueslerReiss(0.3).plot_pdf()

.. image:: _static/images/hr_delta0point3_pdf.png
    :alt: alternate text
    :width: 500px
    :align: center


-------------------

**Use matrices for Checkerboard copulas**

Checkerboard copulas are copulas that have probability density functions, which are constant on rectangles and can be represented by matrices.

.. highlight:: python
.. code-block:: python

    >>> matr = [[0, 9, 1], [1, 0, 9], [9, 1, 0]]
            >>> ccop = cp.BivCheckPi(matr)
            >>> ccop.cdf(0.2, 1)
            0.2
            >>> ccop.pdf(0.2, 1)
            0.03333333333333333
            >>> ccop.scatter_plot()
        >>> ccop = cp.CheckerboardCopula(matr)
        >>> ccop.cdf(0.2, 1)
        0.2
        >>> ccop.pdf(0.2, 1)
        0.03333333333333333
        >>> ccop.scatter_plot()
        >>> ccop = cp.CheckPi(matr)
        >>> ccop.cdf(0.2, 1)
        0.2
        >>> ccop.pdf(0.2, 1)
        0.03333333333333333
        >>> ccop.scatter_plot()
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

**Spearman's rho, Kendall's tau and Chatterjee's xi**

.. highlight:: python
.. code-block:: python

    >>> cp.CuadrasAuge().rho()
    -3*delta/(delta - 4)
    >>> cp.CuadrasAuge(0.5).rho()
    0.428571428571427
    >>> cp.FarlieGumbelMorgenstern().tau()
    2*theta/9
    >>> cp.AliMikhailHaq().xi()
    -theta/6 - 0.666666666666667 + 3/theta - 2/theta**2 - 2*(1 - theta)**2*log(1 - theta)/theta**3
    >>> cp.Gaussian().plot_rank_correlations(1_000_000, 50)


.. image:: _static/images/Gaussian_rank_correlations.png
    :alt: alternate text
    :width: 500px
    :align: center

-------------------

**Archimedean copulas: Generator and inverse generator functions**

Archimedean copulas are characterized by a generator function, which is available in the package.

.. highlight:: python
.. code-block:: python

    >>> nelsen7 = cp.Nelsen7()
    >>> nelsen7.generator
    -log(t*theta - theta + 1)
    >>> nelsen7.inv_generator
    (theta*exp(y) - exp(y) + 1)*exp(-y)*Heaviside(-y - log(1 - theta))/theta
    >>> cp.Nelsen7(0.5).plot_generator()

.. image:: _static/images/Nelsen7_generator.png
    :alt: alternate text
    :width: 500px
    :align: center

**Extreme-value copulas: Pickands dependence functions**

Extreme-value copulas are characterized by a pickands dependence function, which is also available in the package.

.. highlight:: python
.. code-block:: python

    >>> galambos = cp.Galambos()
    >>> galambos.pickands
    1 - 1/((1 - t)**(-delta) + t**(-delta))**(1/delta)
    >>> galambos.plot_pickands(delta=[0.5, 1, 2])

.. image:: _static/images/Galambos_pickands.png
    :alt: alternate text
    :width: 500px
    :align: center

-------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   custom_copulas
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
