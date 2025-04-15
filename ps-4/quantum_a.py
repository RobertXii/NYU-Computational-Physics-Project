import math as math
import numpy as np
import matplotlib.pyplot as plt

def Hermite_polynomial(n,x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return (2*x*Hermite_polynomial(n-1,x)) - (2*(n-1)*Hermite_polynomial(n-2,x))

def quantum_well(n, x):
    return (1 / (np.sqrt(2 ** n * math.factorial(n) * np.sqrt(np.pi)))) * np.e ** ( -x ** 2 / 2) * Hermite_polynomial(n, x)

k = 1000
n = 4 #number of solutions
m = k
x_values = np.linspace(-4, 4, k)
matrix = np.zeros((n, m)) #solutions

for i in range(n):
    o = 0
    for j in x_values:
        solution = quantum_well(i, j)
        matrix[i][o] = solution
        o += 1
    plt.plot(x_values, matrix[i])

plt.xlabel('x')
plt.ylabel('Probability density')
plt.savefig('quantum_a.png')
plt.show()