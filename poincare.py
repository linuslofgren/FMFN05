import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



b_vals = np.linspace(0, 0.21, 400)
b = 0.2




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
ax_pon = fig.add_axes([0, 0, 1, 1])

fig_thomas = plt.figure(figsize=(20, 8))
# ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax = fig_thomas.add_axes([0, 0, 1, 1], projection='3d')
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
# # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

from itertools import product


# for eig in v:
#     plt.quiver(*zero, *eig, arrow_length_ratio=1.0)
plane = np.array([1, 1, 1])
D = -6
if plane[0] != 0:
    v1 = (-plane[1], plane[0], 0)
else:
    v1 = (0, -plane[2], plane[1])
v2 = np.cross(plane, v1)
O = np.array([0,0,0]) if D==0 else np.array([0, 0, -D/plane[2]])
poncare_points = []
xx, yy = np.meshgrid(range(-4, 4), range(-4, 4))
zz = (-plane[0]*xx-plane[1]*yy-D)/plane[2]
ax.plot_surface(xx, yy, zz, alpha=0.2)
ax.quiver(xx[0,0], yy[0,0], zz[0,0], v1[0], v1[1], v1[2])
ax.quiver(xx[0,0], yy[0,0], zz[0,0], v2[0], v2[1], v2[2])
ax.set_aspect('equal', adjustable='box')

a = range(-10,10, 4)
for xi, yi, zi in tqdm(product(a, a, a), total=len(a)**3):
    # xi = 0.2
    # yi = 0.1
    # zi = -0.1
    state = [xi, yi, zi]
    # state=[0.1, 0.1, -0.1]
    soln = odeint(f, state, t=np.linspace(0,100,1000), args=(b,) ,rtol=1e-6)

    x = soln.T[0]
    y = soln.T[1]
    z = soln.T[2]
    ax.plot(x, y, z, label=f"Trajectory in 3D space b={b}", color="orange", alpha=0.1)
    for v in zip(x, y, z):
        d = (np.dot(v, plane)+D)/np.linalg.norm(plane)
        if(np.abs(d) < 0.1):
            # plt.scatter(v[0], v[1], v[2], c="k", s=1)
            P = np.array([v[0]-plane[0]*np.abs(d),v[1]-plane[1]*np.abs(d),v[2]-plane[2]*np.abs(d)])
            u1 = np.dot(P-O, v1)/np.linalg.norm(v1)
            u2 = np.dot(P-O, v2)/np.linalg.norm(v2)
            poncare_points.append([u1, u2])
poincare = np.array(poncare_points)
if len(poincare):
    ax_pon.scatter(poincare[:,0], poincare[:,1], c="k", s=0.6)
ax_pon.set_aspect('equal')
# ax.plot(x, y, z, label=f"Trajectory in 3D space b={b}", color="orange", alpha=0.1)
# plt.title(f"Trajectory in 3D space b={b}")
# plt.tight_layout()
plt.show()
# plt.savefig("thomas_flower", dpi=200)