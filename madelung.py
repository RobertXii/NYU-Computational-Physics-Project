import math
import numpy as np
import time

def madelung_for_loop(L):
    r = math.floor(L ** (1 / 3)/2)# find the half of the length of each side of the cube
    x = 0
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                if i == 0 and j == 0 and k == 0:
                    continue
                x += (-1) ** (i + j + k) / math.sqrt(i**2 + j**2 + k**2)
    return x

def madelung_no_for_loop(L):
    r = math.floor(L ** (1 / 3)/2)# find the half of the length of each side of the cube
    # Generate arrays of indices using numpy
    i, j, k = np.meshgrid(np.arange(-r, r+1), np.arange(-r,r+1), np.arange(-r,r+1))

    # Mask the (0, 0, 0) point
    mask = (i != 0) | (j != 0) | (k != 0)

    distance = np.sqrt(i**2 + j**2 + k**2)
    sign = (-1.0) ** (i + j + k)

    # Calculate the Madelung constant
    x = np.sum(sign[mask] / distance[mask])

    return x


L = 10000000 # set the number of atoms for the calculation
start_for = time.time()
x = madelung_for_loop(L)
end_for = time.time()
time_for = end_for-start_for

start_no_for = time.time()
y = madelung_no_for_loop(L)
end_no_for = time.time()
time_no_for = end_no_for-start_no_for


print(f"Madelung constant for NaCl with L ={L}: {x} using for loop, it takes {time_for}s")
print(f"Madelung constant for NaCl with L ={L}: {y} without using for loop, it takes {time_no_for}s")

#Madelung constant for NaCl with L =10000000: 1.7529352948146202 using for loop, it takes 9.706836938858032s
#Madelung constant for NaCl with L =10000000: 1.7529352948110846 without using for loop, it takes 0.31986522674560547s
