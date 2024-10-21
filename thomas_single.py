import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



b_vals = np.linspace(0, 0.21, 400)
b = 0.2
# a = 0.25
# c = 2

# def f(t, state):
#     x, y, z = state
#     return (a*x-y-z, x, x*x-c*z)


def f(state, t, b):
    x, y, z = state
    return (np.sin(y)-b*x, np.sin(z)-b*y, np.sin(x)-b*z)

def jacobian(x, y, z, b):
    return np.array([
        [-b, np.cos(y), 0],
        [0, -b, np.cos(z)],
        [np.cos(x), 0, -b]
    ])

def jac(a, b):
    x, y, z = a
    return jacobian(x, y, z, b)

def fun(a, b):
    return f(a, 0, b)

from scipy import optimize

soln = optimize.root(fun, [2.5,2.5,2.5], jac=jac, args=(b,))
zero = soln.x

from scipy.linalg import eigh

w, v = eigh(jac(zero, b))
print(w, v)


fig = plt.figure(figsize=(20, 8))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
ax.set_title("3D Plot of (x, y, z)")

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
from itertools import product

a = range(-10,10, 4)

# for eig in v:
#     plt.quiver(*zero, *eig, arrow_length_ratio=1.0)

for xi, yi, zi in product(a, a, a):
    state = [xi, yi, zi]
    # state=[0.1, 0.1, -0.1]
    soln = odeint(f, state, t=np.linspace(0,100,1000), args=(b,) ,rtol=1e-6)

    x = soln.T[0]
    y = soln.T[1]
    z = soln.T[2]

    ax.plot(x, y, z, label=f"Trajectory in 3D space b={b}", color="orange", alpha=0.1)
    plt.title(f"Trajectory in 3D space b={b}")
plt.tight_layout()
plt.show()
# plt.savefig("thomas_flower", dpi=200)