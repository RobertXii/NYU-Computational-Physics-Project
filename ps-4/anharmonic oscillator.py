import numpy as np
from gaussxw import gaussxw
import matplotlib.pyplot as plt

N= 20

# Calculate the sample points and weights, then map them # to the required integration domain
x,w = gaussxw(N)

def Period(b):
    a = 0 #lower bound
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w
    E = b**4

    y = 1 / np.sqrt(E - xp**4)
    s = sum(y * wp)

    return s * np.sqrt(8 * 1)

a = np.linspace(0.001, 2, 1000)
periods = [Period(i) for i in a]

plt.plot(a, periods)
plt.xlabel('initial position (a)')
plt.ylabel('period/s')
plt.savefig('anharmoic oscillator.png')
plt.show()
