import numpy as np
import matplotlib.pyplot as plt
import hydro

nt = 1000

hydro.initialize()

for i in range(nt):
    hydro.evolve(i)

def plot():
    for i in range(nt):
        if i % 50 == 0:
            plt.plot(range(hydro.nx), np.load('data/array'+str(i)+'.npy'))

plot()

plt.show()