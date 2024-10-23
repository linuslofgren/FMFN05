import numpy as np
from scipy import optimize
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

def f(state, b):
    x, y, z = state
    return np.array([
        np.sin(y) - b*x,
        np.sin(z) - b*y,
        np.sin(x) - b*z
    ])

def jac(state, b):
    x, y, z = state
    return np.array([
        [-b, np.cos(y), 0],
        [0, -b, np.cos(z)],
        [np.cos(x), 0, -b]
    ])

@np.vectorize(signature='(n)->()')
def kaplan_yorke(l):
    l = np.sort(l)[::-1]
    j = 0
    while np.sum(l[:j+1]) >= 0 and j < l.size - 1:
         j+=1
    return j+np.sum(l[:j])/np.abs(l[j])

def rk4(state, f, dt):
    k1 = f(state)
    k2 = f(state+k1*dt/2.0)
    k3 = f(state+k2*dt/2.0)
    k4 = f(state+k3*dt)
    return state + (k1+2.0*k2+2.0*k3+k4)*dt/6.0

def calculate_lyaponov(f, jac, dim, bs, dt=0.1, n_compute=1000, n_preload=500):
    lyaponov_exponents = np.zeros((bs.size, 3))
    for i, b in tqdm(enumerate(bs), total=bs.size):
        x0 = [1, 1.1, 1.2] # np.random.random((3))

        # Basis vectors for divergence calculation
        W = np.eye(3)

        state = x0
        lambdas = np.zeros(dim)
        dt = 0.1

        # Evolve the system from initial conditions closer to the attractor
        for _ in range(n_preload):
            state = rk4(state, partial(f, b=b), dt=dt)

        # Keep track of total time to normalize lyaponov exponents
        t_total = 0

        for _ in range(n_compute):
            t_total += dt
            # How will each basis vector evolve? Evolve W
            W = rk4(W, lambda M: jac(state, b) @ M, dt=dt)
            # Evolve the state
            state = rk4(state, partial(f, b=b), dt=dt)

            # equation (8) in https://carretero.sdsu.edu/teaching/M-638/lectures/ProgTheorPhys_1990-Geist-875-93.pdf
            # lambda = lim_{t->inf} 1/t ln[sigma^2(t)]
            # @MISC {4465988,
            #     TITLE = {How to understand QR decomposition? Compare the power method, QR decomposition for finding eigenvalues and Lyapunov exponents.},
            #     AUTHOR = {Lutz Lehmann (https://math.stackexchange.com/users/115115/lutz-lehmann)},
            #     HOWPUBLISHED = {Mathematics Stack Exchange},
            #     NOTE = {URL:https://math.stackexchange.com/q/4465988 (version: 2022-06-05)},
            #     EPRINT = {https://math.stackexchange.com/q/4465988},
            #     URL = {https://math.stackexchange.com/q/4465988}
            # }
            W, R = np.linalg.qr(W)
            for j in range(dim):
                # This is not trivial
                lambdas[j] += np.log(np.abs(R[j, j]))

        lambdas = lambdas/(t_total)
        lyaponov_exponents[i] = lambdas

    return lyaponov_exponents

dim = 3
bs = np.linspace(0.0, 1.1, 100)
lyaponov_exponents = calculate_lyaponov(f, jac, 3, bs)
box_dim = kaplan_yorke(lyaponov_exponents)

fig, (ax1, ax2,) = plt.subplots(2, sharex=True, dpi=100, figsize=(14, 7))

ax1.scatter(bs, box_dim, s=0.01, c="k")
ax1.axhline(1, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax1.axhline(2, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax1.set_ylabel("$D_B$")
ax1.set_title("Box dimension")

ax2.plot(bs, lyaponov_exponents, "k", linewidth=0.1)
ax2.axhline(0, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax2.invert_xaxis()
ax2.set_xlabel("b")
ax2.set_ylabel(r"$\lambda$")
ax2.set_title("Lyaponov exponents")

plt.show()
