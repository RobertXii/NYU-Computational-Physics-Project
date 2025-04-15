import numpy as np
import matplotlib.pyplot as plt

# Set the dimensions and resolution of the image
N = 1000  # Number of pixels for width and height
xmin, xmax = -2.0, 2.0  # X-axis range
ymin, ymax = -2.0, 2.0  # Y-axis range

# Create a grid of complex numbers
x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
C = np.array([[complex(a, b) for a in x] for b in y])

# Create an array to store the Mandelbrot set
mandelbrot = np.zeros((N, N), dtype=int)

# Set the maximum number of iterations
max_iter = 100

# Calculate the Mandelbrot set
z = np.zeros((N, N), dtype=complex)
for k in range(max_iter):
    mask = np.abs(z) <= 2.0
    mandelbrot += mask  # Increment iterations for points still in the set
    z[mask] = z[mask] * z[mask] + C[mask]

# Create a density plot
plt.imshow(mandelbrot, extent=(xmin, xmax, ymin, ymax), cmap='hot', aspect='auto')
plt.colorbar()
plt.title('Mandelbrot Set')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('mandelbrot.png')
plt.show()
