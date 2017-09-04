"""
Stochastic Gradient Langevin Dynamics (Welling, Teh, ICML 2011)
---------------------------------------------------------------
Example of simple SGLD sampling the posterior of param of 1D linear regression.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


N = 100
n = 50  # minibatch size
true_theta = 1.5

# Noisy data
X = np.linspace(0, 10, num=N)
y = true_theta * X + np.random.randn(N)  # Simple linear regression w/ noise

plt.scatter(X, y)
plt.show()

eps = 0.1  # step size
theta = -60   # initial param

N_iter = 10000

samples = np.zeros(N_iter)
samples[0] = theta

for t in range(1, N_iter):
    mb_idxs = np.random.randint(0, N, size=n)
    x_mb, y_mb = X[mb_idxs], y[mb_idxs]

    eps_t = eps / t  # Annealed learning rate
    eta = np.random.normal(0, np.sqrt(eps_t))  # Noise

    grad_logprior = theta  # prior = N(w | 0, 1)
    grad_loglik = np.sum(y_mb - theta*x_mb)  # lik = N(y | wx, 1)
    grad_logpost = grad_logprior + N/n * grad_loglik  # N/n: scale correction

    delta = eps_t/2 * grad_logpost + eta
    theta = theta + delta

    samples[t] = theta

burnin = 100  # burn-in samples to be discarded
samples = samples[burnin:]

theta_expected = np.mean(samples)

print('Posterior mean: {:.4f}'.format(theta_expected))

sns.distplot(samples)
plt.show()

y_pred = theta_expected*X

plt.scatter(X, y)
plt.plot(X, y_pred, color='red', alpha=0.5)
plt.show()
