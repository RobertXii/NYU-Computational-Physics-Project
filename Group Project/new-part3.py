import hydro3
import numpy as np
import matplotlib.pyplot as plt

nt = 300
hydro3.initialize()

for i in range(nt):
    hydro3.evolve(i)

def plot():
    for i in range(nt):
        if i % 50 == 0:
            plt.plot(range(hydro2.nx), np.load('data-part2/array'+str(i)+'.npy'))
            plt.title("Density Distribution at t=")
            # plt.xlabel('x-Position (arb. unit)')
            # plt.ylabel('y-Position (arb. unit)')
            # plt.zlabel('Density (arb. unit)')

    plt.show()

# plot()