# Reciprocal LASSO.
# variable section.
# Song, Q., & Liang, F. (2015). High-Dimensional Variable Selection With Reciprocal L1-
# Regularization. Journal of the American Statistical Association, 110(512), 1607–1620. http://www.jstor.org/stable/24740172
library(glmnet)

# Reciprocal LASSO function
reciprocal_lasso <- function(x, y, lambda_seq) {
  n <- nrow(x)
  p <- ncol(x)
  
  # Standardize the predictors
  x <- scale(x)
  
  # Initialize coefficients
  beta <- rep(0, p)
  
  # Gradient descent parameters
  alpha <- 0.01
  max_iter <- 1000
  tol <- 1e-6
  
  for (lambda in lambda_seq) {
    for (iter in 1:max_iter) {
      beta_old <- beta
      for (j in 1:p) {
        r <- y - x %*% beta + x[, j] * beta[j]
        beta[j] <- soft_thresholding(sum(x[, j] * r) / n, lambda / (1 + abs(beta[j])))
      }
      if (sum(abs(beta - beta_old)) < tol) break
    }
  }
  
  return(beta)
}

soft_thresholding <- function(z, gamma) {
  sign(z) * max(0, abs(z) - gamma)
}

# Example data
set.seed(123)
x <- matrix(rnorm(100 * 10), 100, 10)
y <- rnorm(100)

# Sequence of lambda values
lambda_seq <- 10^seq(2, -4, by = -0.1)

# Fit the Reciprocal LASSO model
beta_reciprocal_lasso <- reciprocal_lasso(x, y, lambda_seq)

# Fit standard LASSO model
lasso_model <- glmnet(x, y, alpha = 1, lambda = lambda_seq)
cv_lasso_model <- cv.glmnet(x, y, alpha = 1, lambda = lambda_seq)
optimal_lambda <- cv_lasso_model$lambda.min

beta_lasso <- coef(lasso_model, s = optimal_lambda)

# Make predictions
pred_lasso <- predict(lasso_model, s = optimal_lambda, newx = x)
pred_reciprocal_lasso <- x %*% beta_reciprocal_lasso

# Calculate MSE
mse_lasso <- mean((y - pred_lasso)^2)
mse_reciprocal_lasso <- mean((y - pred_reciprocal_lasso)^2)

# Display MSEs
mse_comparison <- data.frame(
  Model = c("LASSO", "Reciprocal LASSO"),
  MSE = c(mse_lasso, mse_reciprocal_lasso)
)

print(mse_comparison)
