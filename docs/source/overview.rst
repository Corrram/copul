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


**Evaluation**

All copulas share one API. Called with arguments (scalars, broadcastable arrays,
an ``(N, 2)`` array or ``u=``, ``v=`` keywords) the methods return numbers;
called without arguments they return SymPy expressions.

- ``cdf``, ``pdf``, ``logpdf``, ``survival_function``: distribution function,
  density, log-density and survival function
- ``cond_distr_1``, ``cond_distr_2``: conditional distribution functions, and
  ``cond_distr_1_inv``, ``cond_distr_2_inv``: conditional quantiles
- ``generator``, ``inv_generator``: generator and inverse generator of
  Archimedean families; ``pickands``: Pickands dependence function of
  extreme-value families

**Dependence measures**

``chatterjees_xi``, ``spearmans_rho``, ``kendalls_tau``, ``blests_nu``,
``spearmans_footrule``, ``ginis_gamma``, ``blomqvists_beta``, ``hoeffdings_d``,
``schweizer_wolff_sigma``, ``uniform_distance``, ``lp_distance``,
``blum_kiefer_rosenblatt``, ``mutual_information``, ``lambda_L`` and
``lambda_U``, each with the ``method=`` switch described in :doc:`measures`.

**Methods**

The following methods are available for the above copula families and copulas:

- ``rvs``: Generate random samples from the copula
- ``scatter_plot``: Generate a scatter plot of the copula
- ``plot_cdf``: Visualize the cumulative distribution function
- ``plot_pdf``: Visualize the probability density function
- ``plot_rank_correlations``: Visualize Spearman's rho, Kendall's tau, and Chatterjee's xi
- ``plot_generator``: Visualize the generator function
- ``plot_pickands``: Visualize the Pickands dependence function