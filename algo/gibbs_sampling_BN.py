import numpy as np


"""
Bayes Net
---------

(S) --> (T) <-- (R) --> (J)

p(S,R,J,T) = p(S)p(R)p(J|R)p(T|S,R)
"""

# CPD Tables
p_r = np.array([0.8, 0.2])
p_s = np.array([0.9, 0.1])

p_j_given_r = np.array([
    [0.8, 0.2],  # p(j|r=0)
    [0, 1]  # p(j|r=1)
])


p_t_given_r_s = np.array([
    [[1, 0], [0.1, 0.9]],  # p(t|r=0,s)
    [[0, 1], [0, 1]],  # p(t|r=1,s)
])

supp = [0, 1]  # Support of all dists.


# Evidence
t_evid = 1


"""
Forward Sampling
"""


# Forward sampling, discard sample not satisfying evidence
n_sample = 1000
samples = []
i = 0

while True:
    r = np.random.choice(supp, p=p_r)
    s = np.random.choice(supp, p=p_s)
    j = np.random.choice(supp, p=p_j_given_r[r])
    t = np.random.choice(supp, p=p_t_given_r_s[r, s])

    if t == t_evid:
        samples.append([r, s, j, t])

    if len(samples) >= n_sample:
        break

    i += 1

print('======================================================================')
print('Total samples needed to get {} valid samples: {}'.format(n_sample, i))


"""
Gibbs Sampling
"""

r, s, j, t = 0, 1, 0, 1  # Initial states
samples_gibbs = []

gibbs_iters = 25000
burnin = 1000
thinning = 20

for it in range(gibbs_iters):
    # Pick index random variables
    i = np.random.randint(3)

    if i == 0:  # r
        # Sample from full conditional of r: p(r)p(j|r)p(t|r,s)/Z
        p_1 = p_r[r]*p_j_given_r[r, j]*p_t_given_r_s[r, s, t]
        Z = p_1 + p_r[1-r]*p_j_given_r[1-r, j]*p_t_given_r_s[1-r, s, t]
        p_1 = p_1 / Z
        p_0 = 1-p_1

        p = [p_0, p_1] if r == 1 else [p_1, p_0]

        r = np.random.choice(supp, p=p)
    elif i == 1:  # s
        # Sample from: p(s)p(t|r,s)/Z
        p_1 = p_s[s]*p_t_given_r_s[r, s, t]/(p_s[s]*p_t_given_r_s[r, s, t] + p_s[1-s]*p_t_given_r_s[r, 1-s, t])
        p_0 = 1-p_1

        p = [p_0, p_1] if s == 1 else [p_1, p_0]

        s = np.random.choice(supp, p=p)
    elif i == 2:  # j
        # Sample from: p(j|r)
        j = np.random.choice(supp, p=p_j_given_r[r])

    if it >= burnin:
        if it % thinning == 0:
            samples_gibbs.append([r, s, j, t])

print('======================================================================')
print('Gibbs sampling with {} iterations, burnin: {} and thinning: {}'.format(gibbs_iters, burnin, thinning))
print('Total samples from Gibbs sampling: {}'.format(len(samples_gibbs)))

print('======================================================================')
samples = np.array(samples)
samples_gibbs = np.array(samples_gibbs)

# Look for samples with s = 1
# Ancestral
p_s1_t1 = np.mean(samples[:, 1] == 1)
print('Ancestral sampling: p(s=1|t=1) = {:.4f}'.format(p_s1_t1))

# Gibbs
p_s1_t1_gibbs = np.mean(samples_gibbs[:, 1] == 1)
print('Gibbs sampling: p(s=1|t=1) = {:.4f}'.format(p_s1_t1_gibbs))
print('======================================================================')
