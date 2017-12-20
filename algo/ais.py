"""
Annealed Importance Sampling (AIS
=================================
Neal, Radford M. "Annealed importance sampling." Statistics and computing 11.2 (2001): 125-139.
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def f_0(x):
    """
    Target distribution: \propto N(-5, 2)
    """
    return np.exp(-(x+5)**2/2/2)


def f_n(x):
    """
    Proposal distribution: \propto N(0, 1)
    """
    return np.exp(-x**2/2)


def f_j(x, beta):
    """
    Intermediate distribution: interpolation between f_0 and f_n
    """
    return f_0(x)**beta * f_n(x)**(1-beta)


def T(x, f, n_steps=10):
    """
    Transition distribution: T(x'|x) using n-steps Metropolis sampler
    """
    for t in range(n_steps):
        # Proposal
        x_prime = x + np.random.randn()

        # Acceptance prob
        a = f(x_prime) / f(x)

        if np.random.rand() < a:
            x = x_prime

    return x


# Proposal distribution: 1/Z * f_n
p_n = st.norm(0, 1)


x = np.arange(-10, 5, 0.1)

n_inter = 50  # num of intermediate dists
betas = np.linspace(0, 1, n_inter)


# Plot sequences of distribution starting from q(x) to p(x)
for beta in betas:
    plt.plot(x, f_j(x, beta))

plt.show()


# Sampling
n_samples = 100
samples = np.zeros(n_samples)
weights = np.zeros(n_samples)

for t in range(n_samples):
    # Sample initial point from q(x)
    x = p_n.rvs()
    w = 1

    for n in range(1, len(betas)):
        # Transition
        x = T(x, lambda x: f_j(x, betas[n]), n_steps=5)

        # Compute weight in log space:
        # w *= f_{n-1}(x_{n-1}) / f_n(x_{n-1})
        w += np.log(f_j(x, betas[n])) - np.log(f_j(x, betas[n-1]))

    samples[t] = x
    weights[t] = np.exp(w)


# Compute expectation
a = 1/np.sum(weights) * np.sum(weights * samples)
# Print: should be close to -5, the mean of p_0
print('Expectation of p_0: {:.4f}'.format(a))
