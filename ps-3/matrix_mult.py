from numpy import zeros
import time
import numpy as np
import matplotlib.pyplot as plt

# Set sizes of matrices that we want to perform multiplication on.
my_values = np.array([10, 30, 50,70,90])
num_matrices = len(my_values)

# Create an empty array to store elapsed times
elapsed_time = np.empty(num_matrices)
elapsed_time_dot = np.empty(num_matrices)


for i in range(num_matrices):
    N = my_values[i]
    A = zeros([N, N], float)
    B = zeros([N, N], float)
    C = zeros([N, N], float)

    start_time = time.time()
    for row in range(N):
        for col in range(N):
            for k in range(N):
                C[row, col] += A[row, k] * B[k, col]
    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)
    elapsed_time[i] = elapsed

    ##using dot method
    start_timeDOT = time.time()

    # Use the dot method for matrix multiplication
    C = np.dot(A, B)

    end_timeDOT = time.time()
    elapsedDOT = end_timeDOT - start_timeDOT
    elapsed_time_dot[i] = elapsedDOT

print(elapsed_time)
print(elapsed_time_dot)

# Create the plot
plt.plot(my_values, elapsed_time,label='explicit' )
plt.plot(my_values, elapsed_time_dot,label='dot' )

# Create a NumPy array
x = np.linspace(0, 90, 1000)  # Create an array of 100 points
x_cubed = 0.0000005*x**3  # Compute the values of x^3
# Add the plot for the function x^3
plt.plot(x, x_cubed, label='0.0000005*x^3', color='red')
# Add labels and title
plt.xlabel('size of matrix')
plt.ylabel('elapsed time(s)')
plt.title('size vs elapsed time using explicit function')
# Add a legend
plt.legend()

# Show the plot
plt.show()