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
