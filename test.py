
import numpy as np
import matplotlib.pyplot as plt
from opfunu.cec_based.cec2014 import F12014, F112014

# Define the bounds and number of dimensions
ndim = 2
bounds = [-100, 100]

# Initialize the function with the specified bounds and dimensions
f = F112014(ndim=ndim)

# Generate a grid of x and y values
x = np.linspace(bounds[0], bounds[1], 100)
y = np.linspace(bounds[0], bounds[1], 100)
X, Y = np.meshgrid(x, y)

# evaluate the function at each point on the grid
Z = np.array(
    [f.evaluate([xi, yi]) for xi, yi in zip(X.flatten(), Y.flatten())]
)
Z = Z.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
