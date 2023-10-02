import numpy as np
from gaussxw import gaussxw
import math as math


def Hermite_polynomial(n,x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return (2*x*Hermite_polynomial(n-1,x)) - (2*(n-1)*Hermite_polynomial(n-2,x))

def quantum_well(n, x):
    return (1 / (np.sqrt(2 ** n * math.factorial(n) * np.sqrt(np.pi)))) * np.e ** (-x ** 2 / 2) * Hermite_polynomial(n, x)


N = 100
# Calculate the sample points and weights, then map them # to the required integration domain
x, w = gaussxw(N)

boundary = 10
a = -boundary
b = boundary
xp = 0.5 * (b - a) * x + 0.5 * (b + a)
wp = 0.5 * (b - a) * w
y = []

for i in xp:
    yp = i**2 * quantum_well(5,i)**2
    y.append(yp)

s = sum(y * wp)

print(np.sqrt(s))

