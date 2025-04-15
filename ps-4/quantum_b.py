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

k = 300
x_values = np.linspace(-10, 10, k)
i = 30
solution_30 = []
for j in x_values:
    solution = quantum_well(i, j)
    solution_30.append(solution)

plt.plot(x_values, solution_30)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.savefig('quantum_b_n=30.png')
plt.show()