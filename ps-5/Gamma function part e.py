import matplotlib.pyplot as plt
import numpy as np
from gaussxw import gaussxwab
import math as math

def Gamma(a,z):
    return ((a - 1) / ((1 - z) ** 2)) * (np.exp((a - 1) * np.log(z * (a - 1) / (1 - z)) - z * (a - 1)/(1 - z)))

N = 50
# Calculate the sample points and weights, then map them # to the required integration domain
x, w = gaussxwab(N,0,1)

def gamma(a):
    s = 0.0
    for k in range(N):
        s += w[k]*Gamma(a, x[k])
    return s

print(f"Gamma(1.5) = {gamma(1.5)}")
print(f"Gamma(3)= {gamma(3)}, 2! = 2")
print(f"Gamma(6)= {gamma(6)}, 5! = 120")
print(f"Gamma(10)= {gamma(10)}, 9! = 362880")

