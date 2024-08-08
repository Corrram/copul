library("pacman")
p_load("VineCopula", "ggplot2", "Pareto")

n <- 10 ### specifies the number of observations

### specify marginal distributions as quantile transformations:
marg_dis1 <- function(p){
    qPareto(p,1,2)
}

### first, consider for simplicity homogeneous marginal distributions
### sample from the sum S of a conditionally independent factor model X_1,...,X_d with homogenous marginal distributions:
sample_sum_cifm <- function(n){
    t <- seq(1/(n+1),1-1/(n+1),length.out = n)
    U_Gauss <- BiCopCondSim(n*d_Gauss,rep(t,each = d_Gauss),1, 1, rep(par_Gauss,n))
    X_Gauss <- marg_dis1(U_Gauss)
    X_matrix_Gauss <- matrix(X_Gauss, ncol = d_Gauss,byrow = TRUE)
    #contains columnwise samples for different conditional values t where column j is drawn from Gaussian copula with parameter equal to the j-th entry of par_Gauss
    U_Clayton <- BiCopCondSim(n*d_Clayton,rep(t,each = d_Clayton),1, 3, rep(par_Clayton,n))
    X_Clayton <- marg_dis1(U_Clayton)
    X_matrix_Clayton <- matrix(X_Clayton, ncol = d_Clayton,byrow = TRUE)
    U_Gumbel <- BiCopCondSim(n*d_Gumbel,rep(t,each = d_Gumbel),1, 4, rep(par_Gumbel,n))
    X_Gumbel <- marg_dis1(U_Gumbel)
    X_matrix_Gumbel <- matrix(X_Gumbel, ncol = d_Gumbel,byrow = TRUE)
    U_Frank <- BiCopCondSim(n*d_Frank,rep(t,each = d_Frank),1, 5, rep(par_Frank,n))
    X_Frank <- marg_dis1(U_Frank)
    X_matrix_Frank <- matrix(X_Frank, ncol = d_Frank,byrow = TRUE)
    X <- cbind(X_matrix_Gauss,X_matrix_Clayton,X_matrix_Gumbel,X_matrix_Frank)
    S <- rowSums(X)
    return(S)
}

### sample from the sum S of independent random variables X_1,...,X_d with respective marginal distributions
sample_sum_ind_rv <- function(n, d){
    U <- runif(d*n)
    X <- marg_dis1(U)
    X_matrix <- matrix(X,ncol=d)
    S <- rowSums(X_matrix)
    return(S)
}

### sample from the sum S of comonotonic random variables X_1,...,X_d with respective marginal distributions
sample_sum_com_rv <- function(n, d){
    U <- runif(n)
    X <- marg_dis1(U)
    X_matrix <- matrix(X,nrow= n, ncol=d, byrow = FALSE)
    S <- rowSums(X_matrix)
    return(S)
}

get_density_plots <- function(n, d, l=level){
    sum_ind <- sample_sum_ind_rv(n, d)
    sum_cifm <- sample_sum_cifm(n)
    sum_com <- sample_sum_com_rv(n, d)
    VaR_sum_ind <- quantile(sum_ind,l)
    VaR_sum_cifm <- quantile(sum_cifm,l)
    VaR_sum_com <- quantile(sum_com,l)
    AVaR_sum_ind <- mean(sum_ind[sum_ind >= VaR_sum_ind])
    AVaR_sum_cifm <- mean(sum_cifm[sum_cifm >= VaR_sum_cifm])
    AVaR_sum_com <- mean(sum_com[sum_com >= VaR_sum_com])

    upp_limq <- quantile(sum_com,(2*l+1)/3)
    upp_lim <- mean(sum_com[sum_com >= upp_limq])

    density_plot <- ggplot(data.frame(seq_along(sum_com),sum_com)) +
        geom_density(aes(x=sum_cifm),lty = 1) +
        geom_density(aes(x=sum_com),lty = 2) +
        geom_density(aes(x=sum_ind),lty = 3) +
        xlim(0,upp_lim) +

        geom_vline(xintercept=VaR_sum_cifm, col="blue", lwd=1.5) +
        annotate(x=VaR_sum_cifm,y=+Inf,label="VaR(S^f)",vjust=2,geom="label") +
        geom_vline(xintercept=AVaR_sum_cifm, col="red", lwd=1.5) +
        annotate(x=AVaR_sum_cifm,y=+Inf,label="AVaR(S^f)",vjust=2,geom="label") +

        geom_vline(xintercept=VaR_sum_com, col="blue", lwd=1, lty =2) +
        annotate(x=VaR_sum_com,y=+Inf,label="VaR(S^c)",vjust=1,geom="label") +
        geom_vline(xintercept=AVaR_sum_com, col="red", lwd=1 , lty =2) +
        annotate(x=AVaR_sum_com,y=+Inf,label="AVaR(S^c)",vjust=1,geom="label") +

        geom_vline(xintercept=VaR_sum_ind, col="blue", lwd=.5, lty =3) +
        annotate(x=VaR_sum_ind,y=+Inf,label="VaR(S^i)",vjust=3,geom="label") +
        geom_vline(xintercept=AVaR_sum_ind, col="red", lwd=.5, lty =3) +
        annotate(x=AVaR_sum_ind,y=+Inf,label="AVaR(S^i)",vjust=4,geom="label") +
        xlab("x") +
        ylab("density of sum of X_i")
    ggsave(path = "../../../images", filename = "cifm_density_plot.png", plot = density_plot)
    return(density_plot)
}

### specify bivariate copulas of the conditional independence factor model:
d_Gauss <- 2 #number of used bivariate Gauss copulas
par_Gauss <- c(.2,0.5) #with parameters as vector of length d_Gauss
d_Clayton <- 10 #number of used bivariate Clayton copulas
par_Clayton <- c(1,2,3,4,5,6,7,8,9,10) #with parameters as vector of length d_Clayton
d_Gumbel <- 7 #number of used bivariate GumbellHougaard copulas
par_Gumbel <- c(2,4,5,6,8,9,10) #with parameters as vector of length d_Gumbel
d_Frank <- 4 #number of used bivariate Frank copulas
par_Frank <- c(6,3,4,35) #with parameters as vector of length d_Frank

d <- d_Gauss + d_Clayton + d_Gumbel + d_Frank

n <- 1000  # 100000
level <- 0.95

get_density_plots(n, d, level)