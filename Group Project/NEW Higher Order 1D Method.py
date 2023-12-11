import hydro2
import numpy as np
import matplotlib.pyplot as plt

nt = 1500
hydro2.initialize()

for i in range(nt):
    hydro2.evolve(i)

def plot():
    for i in range(nt):
        if i % 50 == 0:
            plt.plot(range(hydro2.nx), np.load('data-part2/array'+str(i)+'.npy'))
            plt.title("Density Distribution Over Time At Higher Order")
            plt.xlabel('Position (arb. unit)')
            plt.ylabel('Density (arb. unit)')
    plt.show()

plot()
