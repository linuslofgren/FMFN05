import numpy as np
import matplotlib.pyplot as plt

print("Hello2")
N = 10000
R = np.linspace(3.56, 3.573, N)
P = 10
xj = np.zeros((P, N))
for i, r in enumerate(R):
    for p in range(P):
        x = np.random.random()
        for _ in range(200):
            x = r*x*(1-x)
        xj[p][i] = x
plt.plot(R, xj.T, ls='', marker='.', markersize=1, alpha=0.4, color="gray")
plt.show()


# N = 10
# R = np.linspace(2.95, 3.025, N)
# j = 100

# def show_one(r):
#     xj = np.zeros(j)
#     x = np.random.random()
#     for i in range(j):
#         x = r*x*(1-x)
#         xj[i] = x
#     [p] = plt.plot(xj, linestyle='-', alpha=0.2)
#     plt.plot(xj, marker='.', linestyle='', color=p.get_color())

# # show_one(3.14)
# for r in R:
#     show_one(r)
plt.show()
