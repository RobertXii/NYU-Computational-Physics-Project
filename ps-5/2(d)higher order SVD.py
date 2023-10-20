import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('signal.dat', delimiter='|', skipinitialspace=True)
time= data.iloc[:, 1]
max_time = np.max(time)
time = time/max_time #normalize
signal = data.iloc[:, 2]
max_signal = np.max(signal)
signal = signal/max_signal

plt.plot(time * max_time, signal * max_signal,  '.')
plt.xlabel('time')
plt.ylabel('signal')
#plt.show()

#SVD
order = 100
A = np.zeros((len(time), order+1))
A[:, 0] = 1.
for i in range(order):
    A[:, i] = time ** i
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
w_inverse = np.zeros(np.shape(w))
for i in range(len(w)):
    if w[i] != 0:
        w_inverse[i] = 1. / w[i]
ainv = vt.transpose().dot(np.diag(w_inverse)).dot(u.transpose())
condition_number = np.max(w)/np.min(w)
print(condition_number)
c = ainv.dot(signal)
ym = A.dot(c)
plt.plot(time * max_time , ym * max_signal, '.', label = '9th order')
plt.plot(time * max_time, (ym-signal) * max_signal, '.', label = 'residual')
plt.title('higher order SVD fitting')
plt.show()