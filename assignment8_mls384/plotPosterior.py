import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
n = 4  # Total flips
y = 3  # Number of heads observed

theta_range = np.linspace(0, 1, 100)

posterior_values = binom.pmf(y, n, theta_range) * (n + 1)

plt.figure(figsize=(8, 6))
plt.plot(theta_range, posterior_values, label='Posterior after 1 head flip')
plt.fill_between(theta_range, posterior_values, alpha=0.3)
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
plt.title('Posterior Distribution for $\\theta$ after 4 Flips (Head, Head, Tails, Head)')
plt.legend()
plt.show()

