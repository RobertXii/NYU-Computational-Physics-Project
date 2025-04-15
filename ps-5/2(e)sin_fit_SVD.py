import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

data = pd.read_csv('signal.dat', delimiter='|', skipinitialspace=True)
time= data.iloc[:, 1]
signal = data.iloc[:, 2]

plt.plot(time, signal,  '.')
plt.xlabel('time')
plt.ylabel('signal')

N = 9
T = (np.max(time) - np.min(time))  # time length
A = np.zeros((len(time), 2 * N + 1))
A[:, 0] = 1.  # a0 term
for i in range(1, N + 1):
    A[:, 2 * i - 1] = np.cos(2 * np.pi * time / (T / i))  # b_i
    A[:, 2 * i] = np.sin(2 * np.pi * time / (T / i))  # a_i
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
w_inverse = np.zeros(np.shape(w))
for i in range(len(w)):
    if w[i] != 0:
        w_inverse[i] = 1. / w[i]
ainv = vt.transpose().dot(np.diag(w_inverse)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c)
condition_number = np.max(w) / np.min(w)
print(condition_number)

plt.plot(time, signal, '.', label='data')
plt.plot(time, ym, '.', label='model')
plt.xlabel('time')
plt.ylabel('signal')
plt.legend()
plt.title("Sinusoidal Fit, N=10")
plt.show()