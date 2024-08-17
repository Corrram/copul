# copul

**copul** is a package designed for mathematical computation with and visualization of bivariate copula families.

# Install

Install the copul library using pip.

```bash
pip install copul
```

# Documentation

A guide and documentation is available at [https://copul.readthedocs.io/](https://copul.readthedocs.io/).

# Copula families and copulas

The `copul` package covers implementations of the following copula families:

- **Archimedean copula families**: The 22 Archimedean copula families from the book "Nelsen - An Introduction to Copulas" including 
    * Clayton
    * Gumbel-Hougaard
    * Frank
    * Joe
    * Ali-Mikhail-Haq
    * etc.
- **Extreme-value copulas families**:
    * BB5
    * Cuadras-Augé
    * Galambos
    * Gumbel
    * Husler-Reiss
    * Joe
    * Marshall-Olkin
    * tEV
    * Tawn
- **Elliptical copula families**:
    * Gaussian
    * Student's t
    * Laplace.
- **Other copula families**:
    * Farlie-Gumbel-Morgenstern
    * Fréchet
    * Mardia
    * Plackett
    * Raftery


Furthermore, the package provides the following copulas:

- Independence copula
- Lower and upper Fréchet bounds
- Checkerboard copulas

<!--A list of all implemented copulas can be found in `copul.Families`.

> **Note**: The following examples are also available as a Jupyter notebook in the `notes/examples` folder.-->

# Copula properties

The following properties are available for the above copula families and copulas if they exist and are known:

- `cdf`: Cumulative distribution function
- `pdf`: Probability density function
- `cond_distr_1`, `cond_distr_2`: Conditional distribution functions
- `lambda_L`, `lambda_U`: Lower and upper tail dependence coefficients
- `rho`, `tau`, `xi`: Spearman's rho, Kendall's tau, and Chatterjee's xi
- `generator`, `inv_generator`: Generator and inverse generator for Archimedean copula families
- `pickands`: Pickands dependence functions for extreme-value copula families

# Copula methods

The following methods are available for the above copula families and copulas:

- `rvs`: Generate random samples from the copula
- `scatter_plot`: Generate a scatter plot of the copula
- `plot_cdf`: Visualize the cumulative distribution function
- `plot_pdf`: Visualize the probability density function
- `plot_rank_correlations`: Visualize Spearman's rho, Kendall's tau, and Chatterjee's xi
- `plot_generator`: Visualize the generator function
- `plot_pickands`: Visualize the Pickands dependence function
