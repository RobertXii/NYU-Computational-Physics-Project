import sympy as sp

def f(x):
    return x**4 - 2*x + 1

# Define the limits of integration and the number of slices
a = 0  # Lower limit
b = 2  # Upper limit
N = 20  # Number of slices
h = (b-a)/N

s1 = 0.5*f (a) + 0.5*f(b)
s2 = s1
#calculate 20 slices
for k in range(1,N):
    s1 += f(a+k*h)
s1 *= h

#calculate 10 slices
for k in range(1,10):
    s2 += f(a+k*h*2)
s2 *= h*2

#calculating error s2 using (5.28):
error_theroy = abs((s1-s2)/3)

#calcualting error using integration
# Define the variable and the function
x = sp.symbols('x')
f = x**4 - 2*x + 1
# Calculate the definite integral (optional)
a = 0  # Lower limit
b = 2  # Upper limit
definite_integral = sp.integrate(f, (x, a, b))
error_intergral = abs(s1-definite_integral)

print(f"20 steps trapezoidal: {s1:.4f}")
print(f"10 steps trapezoidal: {s2:.4f}")
print(f"error between 20 steps trapezoidal versus actual difference: {error_theroy:.4f}")
print(f"error between 20 steps trapezoidal versus actual difference: {error_intergral:.4f}")