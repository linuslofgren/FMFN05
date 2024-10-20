# Libraires
import numpy as np
import DynamicalSystem
from tqdm import tqdm
# Compute maximal 1-LCE
def mLCE(system : DynamicalSystem, n_forward : int, n_compute : int, keep : bool):
    '''
    Compute the maximal 1-LCE.
        Parameters:
            system (DynamicalSystem): Dynamical system for which we want to compute the mLCE.
            n_forward (int): Number of steps before starting the mLCE computation. 
            n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
            keep (bool): If True return a numpy array of dimension (n_compute,) containing the evolution of mLCE.
        Returns:
            mLCE (float): Maximum 1-LCE.
            history (numpy.ndarray): Evolution of mLCE during the computation.
    '''
    # Forward the system before the computation of mLCE
    system.forward(n_forward, False)
    
    # Compute the mLCE
    mLCE = 0.
    w = np.random.rand(system.dim)
    w = w / np.linalg.norm(w)
    if keep:
        history = np.zeros(n_compute)
        for i in tqdm(range(1, n_compute + 1)):
            w = system.next_LTM(w)
            system.forward(1, False)
            mLCE += np.log(np.linalg.norm(w))
            history[i-1] = mLCE / (i * system.dt)
            w = w / np.linalg.norm(w)
        mLCE = mLCE / (n_compute * system.dt)
        return mLCE, history
    else:
        for _ in range(n_compute):
            w = system.next_LTM(w)
            system.forward(1, False)
            mLCE += np.log(np.linalg.norm(w))
            w = w / np.linalg.norm(w)
        mLCE = mLCE / (n_compute * system.dt)
        return mLCE

# Compute LCE
def LCE(system : DynamicalSystem, p : int, n_forward : int, n_compute : int, keep : bool):
    '''
    Compute LCE.
        Parameters:
            system (DynamicalSystem): Dynamical system for which we want to compute the LCE.
            p (int): Number of LCE to compute.
            n_forward (int): Number of steps before starting the LCE computation. 
            n_compute (int): Number of steps to compute the LCE, can be adjusted using keep_evolution.
            keep (bool): If True return a numpy array of dimension (n_compute,p) containing the evolution of LCE.
        Returns:
            LCE (numpy.ndarray): Lyapunov Charateristic Exponents.
            history (numpy.ndarray): Evolution of LCE during the computation.
    '''
    # Forward the system before the computation of LCE
    system.forward(n_forward, False)

    # Computation of LCE
    W = np.eye(system.dim)[:,:p]
    LCE = np.zeros(p)
    if keep:
        history = np.zeros((n_compute, p))
        for i in range(1, n_compute + 1):
            W = system.next_LTM(W)
            system.forward(1, True)
            W, R = np.linalg.qr(W)
            for j in range(p):
                LCE[j] += np.log(np.abs(R[j,j]))
                history[i-1,j] = LCE[j] / (i * system.dt)
        LCE = LCE / (n_compute * system.dt)
        return LCE, history
    else:
        for _ in range(n_compute):
            W = system.next_LTM(W)
            system.forward(1, True)
            W, R = np.linalg.qr(W)
            for j in range(p):
                LCE[j] += np.log(np.abs(R[j,j]))
        LCE = LCE / (n_compute * system.dt)
        return LCE

# Compute CLV
def CLV(system : DynamicalSystem, p : int, n_forward : int, n_A : int, n_B : int, n_C : int, traj : bool, check = False):
    '''
    Compute CLV.
        Parameters:
            system (DynamicalSystem): Dynamical system for which we want to compute the mLCE.
            p (int): Number of CLV to compute.
            n_forward (int): Number of steps before starting the CLV computation. 
            n_A (int): Number of steps for the orthogonal matrice Q to converge to BLV.
            n_B (int): Number of time steps for which Phi and R matrices are stored and for which CLV are computed.
            n_C (int): Number of steps for which R matrices are stored in order to converge A to A-. 
            traj (bool): If True return a numpy array of dimension (n_B,system.dim) containing system's trajectory at the times CLV are computed.
        Returns:
            CLV (List): List of numpy.array containing CLV computed during n_B time steps.
            history (numpy.ndarray): Trajectory of the system during the computation of CLV.
    '''
    # Forward the system before the computation of CLV
    system.forward(n_forward, False)

    # Make W converge to Phi
    W = np.eye(system.dim)[:,:p]
    for _ in range(n_A):
        W = system.next_LTM(W)
        W, _ = np.linalg.qr(W)
        system.forward(1, False)
    
    # We continue but now Q and R are stored to compute CLV later
    Phi_list, R_list1 = [W], []
    if traj:
        history = np.zeros((n_B+1, system.dim))
        history[0,:] = system.x
    if check:
        copy = system.copy()
    for i in range(n_B):
        W = system.next_LTM(W)
        W, R = np.linalg.qr(W)
        Phi_list.append(W)
        R_list1.append(R)
        system.forward(1, False)
        if traj:
            history[i+1,:] = system.x
    
    # Now we only store R to compute A- later
    R_list2 = []
    for _ in range(n_C):
        W = system.next_LTM(W)
        W, R = np.linalg.qr(W)
        R_list2.append(R)
        system.forward(1, False)
    
    # Generate A make it converge to A-
    A = np.triu(np.random.rand(p,p))
    for R in reversed(R_list2):
        C = np.diag(1. / np.linalg.norm(A, axis = 0))
        B = A @ C
        A = np.linalg.solve(R, B)
    del R_list2

    # Compute CLV
    CLV = [Phi_list[-1] @ A]
    for Q, R in zip(reversed(Phi_list[:-1]), reversed(R_list1)):
        C = np.diag(1. / np.linalg.norm(A, axis = 0))
        B = A @ C
        A = np.linalg.solve(R, B)
        CLV_t = Q @ A
        CLV.append(CLV_t / np.linalg.norm(CLV_t, axis = 0))
    del R_list1
    del Phi_list
    CLV.reverse()

    if traj:
        if check:
            return CLV, history, copy
        else:
            return CLV, history
    else:
        if check:
            return CLV, copy
        else:
            return CLV

# Compute adjoints of CLV
def ADJ(CLV : list):
    '''
    Compute adjoints vectors of CLV.
        Parameters:
            CLV (list): List of np.ndarray containing CLV at each time step: [CLV(t1), ...,CLV(tn)].
        Returns:
            ADJ (List): List of numpy.array containing adjoints of CLV at each time step (each column corresponds to an adjoint).
    '''
    ADJ = []
    for n in range(len(CLV)):
        try:
            ADJ_t = np.linalg.solve(np.transpose(CLV[n]), np.eye(CLV[n].shape[0]))
            ADJ.append(ADJ_t / np.linalg.norm(ADJ_t, axis = 0))
        except:
            ADJ_t = np.zeros_like(CLV[n])
            for j in range(ADJ_t.shape[1]):
                columns = [i for i in range(ADJ_t.shape[1])]
                columns.remove(j)
                A = np.transpose(CLV[n][:,columns])
                _, _, Vh = np.linalg.svd(A)
                theta_j = Vh[-1] / np.linalg.norm(Vh[-1])
                ADJ_t[:,j] = theta_j
            ADJ.append(ADJ_t)
    return ADJ


def f(state, t, b):
    x, y, z = state
    return np.array((np.sin(y)-b*x, np.sin(z)-b*y, np.sin(x)-b*z))

def jac(state, t, b):
    x, y, z = state
    return np.array([
        [-b, np.cos(y), 0],
        [0, -b, np.cos(z)],
        [np.cos(x), 0, -b]
    ])


def kaplan_yorke(l):
    l = np.sort(l)[::-1]
    j = 0
    while np.sum(l[:j+1]) >= 0 and j < l.size - 1:
        j+=1
    return j+np.sum(l[:j])/np.abs(l[j])

def kaplan_yorke_old(lyapunov_exponents):
    lyapunov_exponents.sort(reverse=True)
    sum_of_exponents = 0
    j = 0
    for _ in range(len(lyapunov_exponents)):
        sum_of_exponents += lyapunov_exponents[_]

        if sum_of_exponents >=0:
            j +=1
        else:
            pass
    
    sum_to_j = 0
    for _ in range(j):
        sum_to_j += lyapunov_exponents[_]
    
    if j == len(lyapunov_exponents):
        KY_dimension = j
    else:
        KY_dimension = j+sum_to_j/np.abs(lyapunov_exponents[j])
    print(KY_dimension)
    return KY_dimension


from scipy import optimize
GENERATE = False

if GENERATE:
    bs = np.linspace(0.0, 1.1, 10000)
    kdim = np.zeros_like(bs)
    lyaps = np.zeros((bs.size, 3))
    lyaps_var = np.zeros((bs.size, 3))
    for i, b in tqdm(enumerate(bs), total=bs.size):
        # print("b=", b)
        t=0
        soln = optimize.root(f, [2.5,2.5,2.5], jac=jac, args=(t,b,))
        zero = soln.x
        delta = np.random.random((3))/1000
        x0 = np.array(zero + delta)
        print(x0)
        t0 = 0.0
        dt = 0.1
        continuous_system = DynamicalSystem.ContinuousDS(x0, t0, f, jac, dt, b=b)
        mLCEr, history = LCE(continuous_system, 3, 500, 700, True)
        # print(history)

        # print(mLCEr)

        lyaps[i] = mLCEr
        lyaps_var[i] = np.var(history[-10:, :], axis=0)

        dim = kaplan_yorke(mLCEr.tolist())

        # print(dim)
        kdim[i] = dim

    # data = np.array([bs, kdim, lyaps])

    with open("lyap_data_accuracy_10000_subset_x0_n.npy", "wb") as f:
        np.save(f, bs)
        np.save(f, kdim)
        np.save(f, lyaps)
        np.save(f, lyaps_var)

else:
    with open("lyap_data_accuracy_10000_subset_x0_n.npy", "rb") as f:
        bs = np.load(f)
        kdim = np.load(f)
        lyaps = np.load(f)
        lyaps_var = np.load(f)

import matplotlib.pyplot as plt
    # 2.431108234844971

# print(lyaps)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.axhline(1, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax1.axhline(2, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax2.axhline(0, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax1.scatter(bs, kdim, s=0.01, c="k")
# ax1.plot(bs, kdim, c="k", linewidth=0.1)
print(lyaps.shape)
window_size = 100
rolling_avg = np.zeros_like(lyaps)  # New array for storing rolling averages
for i in range(3):
    rolling_avg[:, i] = np.convolve(lyaps[:, i], np.ones(window_size) / window_size, "same")
ax2.plot(bs, lyaps, "k", linewidth=0.1)

rolling_kap = np.zeros_like(rolling_avg)
print(rolling_avg.shape[0])
for i in range(rolling_avg.shape[0]):
    rolling_kap[i] = kaplan_yorke(rolling_avg[i])

ax3.axhline(1, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax3.axhline(2, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax4.axhline(0, 0, 1, c="gray", linestyle="--", linewidth=0.4)
ax3.plot(bs, rolling_kap, "k", linewidth=0.4)
ax4.plot(bs, rolling_avg, "k", linewidth=0.4)
ax1.invert_xaxis()
ax2.invert_xaxis()
plt.show()