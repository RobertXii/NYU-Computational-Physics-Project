import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np
import pandas as pd

data = pd.read_csv('survey.csv')
x = data['age'].to_numpy()
y = data['recognized_it'].to_numpy()
x_sort = np.argsort(x)
x = x[x_sort]
y = y[x_sort]
#
# plt.scatter(xs, ys)
# plt.xlabel('age')
# plt.ylabel('0 or 1')
# plt.show()

def logistic_function(x, params):
    beta_0 = params[0]
    beta_1 = params[1]
    y = 1. / (1. + np.exp(-(beta_0 + beta_1 * x)))
    return(y)

def log_likelihood(params, *args):
        x = args[0]
        y = args[1]
        p = logistic_function(x, params=params)
        p = np.clip(p, 1e-16, 1 - 1e-16)
        t = y * np.log10(p/(1-p)) + np.log10(1-p)
        summ = np.sum(np.array(t), axis = -1)
        return -summ

gradient = np.gradient(log_likelihood)

def error(hess_inv, resvariance):
    covariance = hess_inv * resvariance
    return np.sqrt(np.diag(covariance))

#initialization
inti = np.array([0.1, 0.1])

o = optimize.minimize(log_likelihood, inti, args=(x, y))

hess_inv = o.hess_inv
var = o.fun/(len(y)-len(inti))
er = error(hess_inv, var)

print('parameters at optimal: ', o.x, '; error: ', er)

beta_0, beta_1 = o.x
yy = logistic_function(x, [beta_0, beta_1])

plt.scatter(x, y, label="data")
plt.plot(x, yy, label="logistic model", color='black')
plt.legend()
plt.xlabel('age')
plt.ylabel('1 or 0')
plt.savefig("logistic_model.png")
plt.show()