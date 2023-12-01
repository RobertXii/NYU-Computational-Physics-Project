import numpy as np
import matplotlib.pyplot as plt

start = 0.0
end = 50.0
step = 100000
h = (end-start)/step

def fx(x,y,z):
    return 10 * (y - x)

def fy(x,y,z):
    return 28 * x - y - x * z

def fz(x,y,z):
    return x * y- 8/3 * z

def f(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return np.array([fx(x,y,z), fy(x,y,z), fz(x,y,z)], float)

time = np.arange(start, end, h)
xarray = []
yarray = []
zarray = []
r = np.array([0,1,0], float)

for i in time:
    xarray.append(r[0])
    yarray.append(r[1])
    zarray.append(r[2])
    k1 = h * f(r)
    k2 = h * f(r + 0.5 * k1 + 0.5 * h)
    k3 = h * f(r + 0.5 * k2 + 0.5 * h)
    k4 = h * f(r + k3 + h)
    r += (k1 + 2 * k2 + 2 * k3 + k4)/6


plt.plot(time, yarray)
plt.xlabel('t')
plt.ylabel('y')
plt.title('y(t)')
plt.savefig("y(t).png")
plt.show()

plt.plot(xarray, zarray)
plt.xlabel('x')
plt.ylabel('z')
plt.title('z vs x')
plt.savefig("z vs x.png")
plt.show()

