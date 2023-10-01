import numpy as np
from gaussxw import gaussxw
import matplotlib.pyplot as plt

N= 20

# Calculate the sample points and weights, then map them # to the required integration domain
x,w = gaussxw(N)


def Period(bb):
    a = 0
    b = bb
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w
    E = bb**4

    y = 1 / np.sqrt(E - xp**4)
    s = sum(y * wp)

    return s * np.sqrt(8 * 1)

a = np.linspace(0.001, 2, 1000)
periods = [Period(i) for i in a]
print(periods)

plt.plot(a, periods)
plt.xlabel('initial position')
plt.ylabel('period/s')
plt.savefig('anharmoic oscillator.png')
plt.show()
