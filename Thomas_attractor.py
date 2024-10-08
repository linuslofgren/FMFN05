import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # Correct import for Line3DCollection
from matplotlib import cm

# Define the Thomas system equations
def thomas_flow(state, t, b):
    x, y, z = state
    dxdt = -b * x + np.sin(y)
    dydt = -b * y + np.sin(z)
    dzdt = -b * z + np.sin(x)
    return [dxdt, dydt, dzdt]

def sinc_function(x):
    return np.sinc(x/np.pi)

def equation(x,b):
    return sinc_function(x)-b

def fixed_points(b, x_min, x_max, num_points=1000):
    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = sinc_function(x_vals) - b
    
    roots = []
    if b == 1:
        return np.array([0])
    else:
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  
                root_result = root_scalar(equation, args=(b,), bracket=[x_vals[i-1], x_vals[i]], method='brentq')
                if root_result.converged:
                    roots.append(root_result.root)
        return roots


# Parameters
b = 0.2  # A smaller value of b typically leads to chaotic behavior
t_max = 1000
dt = 0.01
t = np.linspace(0, t_max, 100000)
x_min = -50
x_max = 50

fixed_point = fixed_points(b, x_min, x_max)
print(fixed_point)

# Initial condition
initial_state = [0.1, 0.1, -0.1]  # Chaotic behavior typically arises with small perturbations

# Solve the system of equations
solution = odeint(thomas_flow, initial_state, t, args=(b,))

# Extract the solution into x, y, z
x, y, z = solution.T

# Create the gradient based on the time or index
colors = np.linspace(0, 1, len(x))  # Gradient range from 0 to 1

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create segments and color them using the color map
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(colors.min(), colors.max())
lc = Line3DCollection(segments, cmap=cm.plasma, norm=norm)
lc.set_array(colors)
lc.set_linewidth(0.5)
ax.add_collection(lc)

# Plot fixed points as scatter
ax.scatter(fixed_point, fixed_point, fixed_point, color='red')

# Set axis labels and title
ax.set_title('Thomas Flow with Gradient Color')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjust the view limits (optional)
"""ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.set_zlim([-15, 15])"""

# Show colorbar
cbar = fig.colorbar(lc, ax=ax)
cbar.set_label('Time Gradient')

# Display the plot
plt.show()
