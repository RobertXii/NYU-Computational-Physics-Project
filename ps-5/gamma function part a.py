import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values from 0 to 5
x = np.linspace(0, 5, 100)
a_values = [2, 3, 4]

# Create a plot for each value of a
for a in a_values:
    y = x**(a - 1) * np.exp(-x)
    plt.plot(x, y, label=f'a = {a}')

# Set plot labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('integrand function for Different Values of a = 2,3,4')

# Add a legend to differentiate the curves
plt.legend()
plt.savefig('integrand function.png')
plt.show()

