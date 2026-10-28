import numpy as np
import matplotlib.pyplot as plt


def k_space(k_min, k_max, n_points):
    """
    Generate a k-space array for transport calculations.
    Parameters:
    k_min (float): Minimum value of k.
    k_max (float): Maximum value of k.
    n_points (int): Number of points in the k-space.
    """

    k = np.exp(np.linspace(np.log(k_min), np.log(k_max), n_points))
    dlnk = np.log(k[1]) - np.log(k[0])
    return k, dlnk

def initial_condition(A, k, p):
    """
    Generate the initial condition for the transport calculation.
    Parameters:
    A (float): Amplitude of the initial condition.
    k (array): k-space array.
    p (float): spectral power.
    """

    return A * (k**(p))

def dndk(n, k, dlnk):
    """
    Calculate the derivative of n with respect to k.
    Parameters:
    n (array): Array of n values.
    dk (float): Step size in k-space.
    """
    #dn_dk = np.zeros_like(n)
    #dn_dk[1:-1] = (n[2:] - n[:-2]) / (2 * dlnk)

    # Handle the boundaries using forward and backward differences
    #dn_dk[0] = (n[1] - n[0]) / dlnk
    #dn_dk[-1] = (n[-1]- n[-2]) / dlnk

    dn_dlnk = np.gradient(n, dlnk, edge_order=2)

    return dn_dlnk/k

def dndk_upwind(n, k, dlnk):
    """
    Calculate the derivative of n with respect to k using upwind scheme.
    Parameters:
    n (array): Array of n values.
    dk (float): Step size in k-space.
    """

    dn_dk = np.zeros_like(n)
    dn_dk[1:] = (n[1:] - n[:-1]) / dlnk

    return dn_dk/k

def apply_standard_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6):
    """
    Apply standard boundary conditions to the array n.
    Parameters:
    n (array): Array of n values.
    k (array): k-space array.
    """

    n = n.copy()
    n[:buf_left] = n_left_fix

    n[-buf_right:] = n[-buf_right-1]

    return n

def apply_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6, fit_points = 2):

    """
    Applying boundary conditions, where the first 5
    points remain fixed and the last 5 are fitted with a log log 
    linear fit.
    """

    n = n.copy()

    n[:buf_left] = n_left_fix

    idx_fit = slice(-(buf_right + fit_points), -buf_right)
    #log_k_fit = np.log(k[idx_fit])
    k_fit = k[idx_fit]
    #log_n_fit = np.log(n[idx_fit])
    n_fit = n[idx_fit]

    #slope, intercept = np.polyfit(log_k_fit, log_n_fit, 1)
    slope, intercept = np.polyfit(k_fit, n_fit, 1)

    #log_k_tail = np.log(k[-buf_right:])
    k_tail = k[-buf_right:]
    #n[-buf_right:] = np.exp(slope * log_k_tail + intercept)
    n[-buf_right:] = np.exp(slope * k_tail + intercept)

    return n

def rhs(t, n, k, n_left_fix, dlnk , c):
    
    #n = apply_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6, fit_points = 2)
    n = apply_standard_bc(n, k, n_left_fix, buf_left= 6, buf_right=6)
    #dn = dndk(n, k, dlnk)
    dn = dndk_upwind(n, k, dlnk)
    dndt = c*dn
    dndt[:6] = 0.0

    return dndt

def rk4_step(rhs, t, n, dt, n_left_fix, k, dlnk, c):
    k1 = rhs(t, n, k, n_left_fix,  dlnk, c)
    k2 = rhs(t + dt/2, n + dt/2 * k1, k, n_left_fix, dlnk, c )
    k3 = rhs(t + dt/2, n + dt/2 * k2, k, n_left_fix, dlnk, c )
    k4 = rhs(t + dt,   n + dt   * k3, k, n_left_fix, dlnk, c )

    return n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrate(rhs, n0, t_span, dt, n_left_fix, k, dlnk, c):
    t0, tf = t_span
    n_step = int(np.ceil((tf-t0)/dt))

    t = t0
    n = n0.copy()

    ts = [t]
    ns = [n.copy()]

    for _ in range(n_step):
        n = rk4_step(rhs, t, n, dt, n_left_fix, k, dlnk, c)

        t += dt
        ts.append(t)
        ns.append(n.copy())

    return np.array(ts), np.array(ns)

def plot_results(ts, ns, n_snap, k, save_path = None, show = True):

    n_snapshots = n_snap
    idx = np.linspace(0, len(ts) - 1, n_snapshots).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    fig, ax = plt.subplots()
    for i, j in enumerate(idx):
        ax.plot(k, ns[j], color=colors[i], label=f"t = {ts[j]:.2f}")

    ax.set_title("Numerical integration of the transport equation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$n_k$")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.4)

    if save_path:
        plt.savefig(save_path, dpi = 150)
    if show:
        plt.show()
    plt.close()
