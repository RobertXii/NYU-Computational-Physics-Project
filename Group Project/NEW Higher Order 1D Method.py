import hydro2
import numpy as np
import matplotlib.pyplot as plt
import time

# Record the start time
start_time = time.time()
nt = 250
hydro2.initialize()

for i in range(nt):
    hydro2.evolve(i)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time using lower higher method: {elapsed_time} seconds")

def plot():
    for i in range(nt):
        if i % 50 == 0:
            plt.plot(range(hydro2.nx), np.load('data-part2/array'+str(i)+'.npy'))
            plt.title("Pressure Distribution Over Time At Higher Order")
            plt.xlabel('Position (arb. unit)')
            plt.ylabel('Pressure (arb. unit)')
    plt.show()

plot()
