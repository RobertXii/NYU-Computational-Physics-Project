# Newman excercise 9.8 p 439
# The Schrodinger equation and the Crank-Nicolson method

import numpy as np
import banded
import matplotlib.pyplot as plt


L = 1e-8
n = 1000
a = L/n
x0 = L/2
m = 9.109e-31
sigma = 1e-10
k = 5e10
h = 1e-18
h_bar = 1.05457e-34

a1 = 1 + h * 1j * h_bar / (2 * m * a ** 2)
a2 = -h * 1j * h_bar / (4 * m * a ** 2)
b1 = 1 - h * 1j * h_bar / (2 * m * a ** 2)
b2 = h * 1j * h_bar / (4 * m * a ** 2)

# A = np.zeros((n, n), dtype=np.complex_)
# np.fill_diagonal(A, a1)
# np.fill_diagonal(A[:, 1:], a2)
# np.fill_diagonal(A[1:, :], a2)

A = np.zeros((3, n), dtype=np.complex_)
for i in range(n):
    A[1, i] = a1
for i in range(1, n):
    A[0, i] = a2
    A[2, i-1] = a2

B = np.zeros((n, n), dtype=np.complex_)
np.fill_diagonal(B, b1)
np.fill_diagonal(B[:, 1:], b2)
np.fill_diagonal(B[1:, :], b2)


x = np.linspace(0, L, n)  # grid (1000,)
init_states = np.exp(-((x - L / 2.0) ** 2) / (2.0 * sigma ** 2)) * np.exp(1j * k * x)  # initial psi

steps = 1000  # time steps
q = np.zeros((steps, n), dtype='complex_')
q[0, :] = init_states
for i in np.arange(steps - 1):
    v = np.dot(B, q[i, :])
    q[i + 1, :] = banded.banded(A, v, 1, 1)
# print(np.real(q[1,:]))
# print(A)

t = h * steps
xlist = np.array(range(1, n+1)) * a
plt.plot(xlist, np.real(q[-1,:]))
plt.ylim(-1,1)
plt.xlabel("x [m]")
plt.ylabel("$|\psi|$")
plt.title(f"states at time = {t}")
plt.savefig('spatial_distribution_plot500.png')
plt.show()
