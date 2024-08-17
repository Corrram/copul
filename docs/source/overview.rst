**Copula families, properties and methods**
--------------------------------------------

**Families**

The `copul` package covers implementations of the following copula families:

- **Archimedean copula families** including Clayton, Gumbel, Frank, Joe, Ali-Mikhail-Haq, and more
- **Extreme-value copulas families** like Hüsler-Reiss, Galambos, Marshall-Olkin, etc.
- **Elliptical copula families**: Gaussian, Student's t, and Laplace.
- Unclassified copula families like the Plackett or Raftery copula families.

Furthermore, the package provides the following copulas:

- Independence copula
- Lower and upper Fréchet bounds
- **Checkerboard** copulas

A list of all implemented copulas can be found in :py:mod:`copul.Families`.

.. The following examples are also available as a Jupyter notebook in the `notes/examples` folder.

**Properties**

The following properties are available for the above copula families and copulas if they exist and are known:

- ``cdf``: Cumulative distribution function
- ``pdf``: Probability density function
- ``cond_distr_1``, ``cond_distr_2``: Conditional distribution functions
- ``lambda_L``, ``lambda_U``: Lower and upper tail dependence coefficients
- ``rho``, ``tau``, ``xi``: Spearman's rho, Kendall's tau, and Chatterjee's xi
- ``generator``, ``inv_generator``: Generator and inverse generator for Archimedean copula families
- ``pickands``: Pickands dependence functions for extreme-value copula families

**Methods**

The following methods are available for the above copula families and copulas:

- ``rvs``: Generate random samples from the copula
- ``scatter_plot``: Generate a scatter plot of the copula
- ``plot_cdf``: Visualize the cumulative distribution function
- ``plot_pdf``: Visualize the probability density function
- ``plot_rank_correlations``: Visualize Spearman's rho, Kendall's tau, and Chatterjee's xi
- ``plot_generator``: Visualize the generator function
- ``plot_pickands``: Visualize the Pickands dependence function