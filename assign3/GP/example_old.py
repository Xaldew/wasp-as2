import GPy
import numpy as np
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save

# Test data
n = 100
Xtest = np.linspace(0, 10, n).reshape(-1,1)

# Define kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return np.exp(-0.5 * (1/param) * sqdist)

param = 0.5
K_ss = kernel(Xtest, Xtest, param)

# Get Cholesky decomposition
L = np.linalg.cholesky(K_ss + 1e-9*np.eye(n))

# Sample 3 sets from the prior
f_prior = np.dot(L, np.random.normal(size=(n, 3)))

fig1 = plt.figure(1)
plt.clf()
plt.plot(Xtest, f_prior)
plt.axis([0, 10, -5,5])


# Training data
noise = 0
Xtrain = np.array([3, 4, 6, 7]).reshape(-1, 1)
m = len(Xtrain)
ytrain = np.sin(Xtrain) + noise*np.random.normal(size=(m, 1))

# Apply kernel to the training data
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.000005*np.eye(len(Xtrain)))

# Compute mean at our test points
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain))

#Compute standard dev so that we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2).reshape(-1, 1)

# Draw some samples from posterior at the test points
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, 3)))

# For groundtruth
d = np.linspace(0, 10, 1000)

fig2 = plt.figure(2)
plt.clf()
plt.plot(Xtrain, ytrain, 'bs', ms=8)
plt.plot(Xtest, f_post)
plt.gca().fill_between(Xtest.reshape(n), (mu-2*stdv).reshape(n), (mu+2*stdv).reshape(n), color="#dddddd")
plt.plot(Xtest, mu, 'r--', lw=2)
#plt.plot(d, np.sin(d), 'k--')
plt.axis([0, 10, -5, 5])
plt.show()

tikz_save("example_prior.tex", 
            figure=fig1, 
            figureheight='\\figureheight', 
            figurewidth='\\figurewidth')

tikz_save("example_post.tex", 
            figure=fig2, 
            figureheight='\\figureheight', 
            figurewidth='\\figurewidth')






