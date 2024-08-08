.. copul documentation master file, created by
   sphinx-quickstart on Thu Aug  8 14:14:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

copul documentation
===================

**copul** is a package for designed for mathematical computation with copulas.

-------------------

The power of copul::

    >>> import copul
    >>> clayton_cdf = copul.archimedean.Clayton().cdf()
    >>> clayton_cdf
    Max(0, (-1 + v**(-theta) + u**(-theta))**(-1/theta))
    >>> clayton_cdf(u=0.5, v=0.5, theta=0.3)
    0.281766567506623
    >>> copul.archimedean.GumbelHougaard(theta=1).pdf()
    1


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
