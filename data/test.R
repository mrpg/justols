library(sandwich)
library(lmtest)

d <- read.csv("test.csv")
m <- lm(outcome ~ x1 + x2 + x3, data = d)

ct <- coeftest(m, vcov = vcovHC(m, type = "HC3"))
print(ct)

cat("\n")
cat("R-squared:", summary(m)$r.squared, "\n")
cat("Adj R-squared:", summary(m)$adj.r.squared, "\n")

# LOO R-squared (PRESS-based)
h <- hatvalues(m)
press <- sum((residuals(m) / (1 - h))^2)
ss_tot <- sum((d$outcome - mean(d$outcome))^2)
cat("LOO R-squared:", 1 - press / ss_tot, "\n")
cat("PRESS:", press, "\n")

# Condition number of X
X <- model.matrix(m)
sv <- svd(X)$d
cat("Condition number:", max(sv) / min(sv), "\n")

# Max leverage
cat("Max leverage:", max(h), "\n")

# Robust Wald F-test using HC3 covariance
V <- vcovHC(m, type = "HC3")
beta_slope <- coef(m)[-1]
V_slope <- V[-1, -1]
wald <- as.numeric(t(beta_slope) %*% solve(V_slope) %*% beta_slope)
q <- length(beta_slope)
F_robust <- wald / q
cat("Robust F-stat:", F_robust, "\n")
cat("Robust F p-value:", 1 - pf(F_robust, q, m$df.residual), "\n")
