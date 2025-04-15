import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
#question (a)
#define function y=x(x-1)
def function(x):
    y = x*(x-1)
    return y

#define and calculate derivative using at x=1 using definition
def approxderivative(x,delta):
    approxderivative = (function(x+delta)-function(x))/delta
    return approxderivative
delta = 0.01
x = 1
approxderivative = approxderivative(x,delta)

print(f"derivative of x(x-1) at x=1 using definition with delta = {delta}: derivative = {approxderivative} ")

# Define the symbolic variable
x = sp.symbols('x')
# Define the function
f = x*(x-1)
# Calculate the derivative
f_prime = sp.diff(f, x)
anaderivative = f_prime.subs(x, 1)
print(f"derivative of x(x-1) at x=1 analytically: derivative = {anaderivative} ")

#calculate difference
difference = anaderivative - approxderivative
print(f"the difference is {difference} ")

#question(b)
delta = [10**(-2*i) for i in range(1,8)]
my_function = lambda x: x*(x-1)
value = [my_function(1) for x in delta]
value_dx = [my_function(x+1) for x in delta]
approxderivative= (np.array(value_dx)-np.array(value))/np.array(delta)
difference = approxderivative - 1
#print(difference)
print(f"for different delta:{delta}, the difference between actual value is {difference} ")
