import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters for the binomial distribution
n = 4  # Number of trials
p = 0.75  # Probability of success

# Range of possible outcomes
x = np.arange(0, n + 1)

# Binomial distribution PMF
pmf_values = binom.pmf(x, n, p)

posterior = binom.pmf(x, n, p) * (n+1)

# Plotting the binomial distribution
plt.figure(figsize=(10, 6))
plt.stem(x, posterior, basefmt=" ")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.title(f"Binomial Distribution (n={n}, p={p})")
plt.show()

