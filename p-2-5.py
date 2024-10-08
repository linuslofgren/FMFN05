import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
r = 3.2
logi = r*x*(1-x)

def logi_f(x):
    return r*x*(1-x)

x_j = 0.1

x_js = []
for _ in range(200):
    old_x_j = x_j
    old_l = logi_f(x_j)

    x_j = r*x_j*(1-x_j)

    x_js.append(x_j)
    plt.arrow(old_x_j, old_x_j, 0, logi_f(old_x_j)-old_x_j, head_width=0.01)
    plt.arrow(old_x_j, logi_f(old_x_j), x_j-old_x_j, 0, head_width=0.01)

# plt.plot(x_js, gs)
print(x_js)
plt.plot(x, logi)
plt.plot(x, x)
plt.show()