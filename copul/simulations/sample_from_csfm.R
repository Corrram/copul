library("pacman")

p_load("VineCopula", "ggplot2")

sample_gaussian_csfm <- function(n, d) {
    x_means <- c(1, 3, 0, -3, -1)
    x_std <- c(3, 1, 0.1, 1, 3)
    z <- runif(n, min = 0, max = 1)
    par <- c(.95, .7, 0, -.7, -.95)
    x <- matrix(, nrow = n, ncol = 5)
    for (j in 1:d) {
        for (i in 1:n) {
            x[i, j] <- BiCopCondSim(1, z[[i]], 1, family = 1, par = par[[j]])
        }
        x[, j] <- qnorm(x[, j], mean = x_means[[j]], sd = x_std[[j]])
    }
    return(list("x" = x, "z" = z))
}

get_sum_of_x_density_plot <- function(x, var, avar = None) {
    density_plot <- ggplot(data.frame(seq_along(x), x)) +
        geom_density(aes(x = x)) +
        geom_vline(xintercept = var, col = "red", lwd = 1.5) +
        annotate(x = var, y = +Inf, label = "VaR", vjust = 2, geom = "label") +
        geom_vline(xintercept = avar, col = "red", lwd = 1.5) +
        annotate(x = avar, y = +Inf, label = "AVaR", vjust = 2, geom = "label") +
        xlab("x") +
        ylab("density of sum of X_i")
    return(density_plot)
}

create_scattered_distribution_plot <- function(x) {
    matplot(y = x, x = matrix(seq(prod(dim(x))) / nrow(x), nrow = nrow(x), byrow = F) - 1 / nrow(x) + 1)
}

produce_graphs <- function(n, d) {
    sim <- sample_gaussian_csfm(n, d)
    summed_x <- rowSums(sim$x)
    VaR <- quantile(summed_x, 0.95)
    average_VaR <- mean(quantile(summed_x, seq(0.95, 1, 0.0001)))

    scatter_plot <- create_scattered_distribution_plot(sim$x)
    ggsave(path = "../../../images", filename = "csfm_scatter_plot.png", plot = scatter_plot)
    density_plot <- get_sum_of_x_density_plot(summed_x, VaR, average_VaR)
    ggsave(path = "../../../images", filename = "csfm_density_plot.png", plot = density_plot)
}

n <- 1000  # 1000000
d <- 5
produce_graphs(n, d)

