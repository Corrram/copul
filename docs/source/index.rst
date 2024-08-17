.. copul documentation master file, created by
   sphinx-quickstart on Thu Aug  8 14:14:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

copul documentation
===================

**copul** is a package designed for mathematical computation and visualization of bivariate copula families.
It accompanies the `Dependence properties of bivariate copula families <https://www.degruyter.com/document/doi/10.1515/demo-2024-0002/html>`_ article released in the Dependence Modeling journal.

-------------------

**Perform mathematical computations with copul**::

    >>> import copul as cp
    >>> clayton_cdf = cp.archimedean.Clayton().cdf()
    >>> clayton_cdf
    Max(0, (-1 + v**(-theta) + u**(-theta))**(-1/theta))
    >>> clayton_cdf(u=0.5, v=0.5, theta=0.3)
    0.281766567506623
    >>> copul.archimedean.GumbelHougaard(1).pdf()
    1

**Visualize cumulative distribution functions of copulas**::

    >>> cp.UpperFrechet().plot_cdf()

.. image:: _static/images/UpperFrechet.png
   :alt: alternate text
   :width: 300px
   :align: center

**Visualize conditional distributions of copulas**::

.. code-block:: python

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

**Visualize probability density functions of copulas**::

    >>> cp.extreme_value.HueslerReiss(0.3).plot_pdf()

.. image:: _static/images/hr_delta0point3_pdf.png
    :alt: alternate text
    :width: 500px
    :align: center

**Plot Spearman's rho, Kendall's tau and Chatterjee's xi**::

    >>> copul.archimedean.Frank().plot_chatterjee(100_000, 30)

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
