import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the Gaussian distribution
mean = 0
std_dev = 3

# Define the range of x values
x = np.linspace(-10, 10, 100)

# Calculate the Gaussian function
pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std_dev**2))

# Normalize
normalized_pdf = pdf / np.trapz(pdf, x)

# Create the plot
plt.plot(x, normalized_pdf, label=f"Mean = {mean}, Std Dev = {std_dev}")
plt.title('Normalized Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('gaussian.png')
