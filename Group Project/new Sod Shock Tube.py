import numpy as np
import matplotlib.pyplot as plt
import hydro
import time

# Record the start time
start_time = time.time()
nt = 1000

hydro.initialize()

for i in range(nt):
    hydro.evolve(i)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time using lower order method: {elapsed_time} seconds")

def plot():
    for i in range(nt):
        if i % 50 == 0:
            plt.plot(range(hydro.nx), np.load('data/array'+str(i)+'.npy'))
            plt.title("Density Distribution over time")
            plt.xlabel('Position (arb. unit)')
            plt.ylabel('Density (arb. unit)')

plot()

plt.show()