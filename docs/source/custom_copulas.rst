**Working with custom copulas**
--------------------------------

The `copul` package allows users to define custom copulas by specifying their cumulative distribution function.
This can be useful when working with copulas that are not covered by the built-in families or when customizing existing copulas.

Consider the following copula as an example:

.. math::

    C(u, v) = u  v  \exp(-\theta  \ln(u)  \ln(v))


We can define this copula in the following way:

.. highlight:: python
.. code-block:: python

    copula_family = Copula.from_cdf(cdf)
    copula = copula_family(0.1)
