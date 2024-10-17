import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation



b_vals = np.linspace(0, 0.3, 400)
# a = 0.25
# c = 2

# def f(t, state):
#     x, y, z = state
#     return (a*x-y-z, x, x*x-c*z)


def f(state, t, b):
    x, y, z = state
    return (np.sin(y)-b*x, np.sin(z)-b*y, np.sin(x)-b*z)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
ax.set_title("3D Plot of (x, y, z)")


from itertools import product

def g_traj(b):
    ax.cla()
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-12, 12)
    for xi, yi, zi in product(range(-2,2), range(-2,2), range(0,2)):
        state = [xi/10, yi/10, zi/10]
        soln = odeint(f, state, t=np.linspace(0,100,1000), args=(b,) ,rtol=1e-6)

        x = soln.T[0]
        y = soln.T[1]
        z = soln.T[2]
        
        ax.plot(x, y, z, label=f"Trajectory in 3D space b={b}", color="b", alpha=0.1)
    plt.title(f"Trajectory in 3D space b={b}")

def update(frame):
    g_traj(b_vals[frame])


ani = FuncAnimation(fig, update, frames=len(b_vals), interval=1000 / 4) 

ani.save('thomas3d.mp4', writer='ffmpeg', fps=24)

plt.show()