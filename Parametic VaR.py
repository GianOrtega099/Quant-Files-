from matplotlib.pylab import norm
import numpy as np
from scipy.stats import norm

returns = np.array({0.01, -0.02, 0.15, -0.005, 0.03, -0.01, 0.02}) # your daily portfolio returns as decimals
returns = np.array([0.01, -0.02, 0.15, -0.005, 0.03, -0.01, 0.02])
mean = np.mean(returns)

std = np.std(returns, ddof=1)  # sample standard deviation
confidence_level = 0.95

z_score = norm.ppf(1 - confidence_level)

var_return = mean + z_score * std
portfolio_value = 500000
var_dollar = abs(var_return) * portfolio_value

print(f"Parametric VaR (95%) as return: {var_return:.4f}")
print(f"Parametric VaR (95%) in dollars: ${var_dollar:.2f}")
std = np.std(returns, ddof=1) # sample standard deviation
confidence_level = 0.95

# norm.ppf gives the z-score for a given probability
# norm.ppf gives the z-score for a given probability
z_score = norm.ppf(1 - confidence_level)  # about -1.645 for 95%

var_return = mean + z_score * std
portfolio_value = 500000  # example portfolio size in dollars
var_dollar = abs(var_return) * portfolio_value

print(f"Parametric VaR (95%) as return: {var_return:.4f}")
print(f"Parametric VaR (95%) in dollars: ${var_dollar:.2f}")