import hydro2
import numpy as np
import matplotlib.pyplot as plt

nt = 500
hydro2.initialize()

for i in range(nt):
    hydro2.evolve(i)

def plot():
    for i in range(nt):
        if i % 50 == 0:
            plt.plot(range(hydro2.nx), np.load('data-part2/array'+str(i)+'.npy'))
    plt.show()

plot()
