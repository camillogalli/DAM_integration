import numpy as np
import matplotlib.pyplot as plt

def k_space(k_min, k_max, n_points):

    k = np.exp(np.linspace(np.log(k_min), np.log(k_max), n_points))
    dlnk = np.log(k[1]) - np.log(k[0])
    return k, dlnk

def initial_condition(A, k, p):

    return A * (k**(p))

def dndk_upwind(n, k, dlnk):

    dn_dlnk = np.zeros_like(n)
    dn_dlnk[1:] = (n[1:] - n[:-1]) / dlnk

    return dn_dlnk/k

def d2ndk2_center(n, k, dlnk):

    d2n_dlnk2 = np.zeros_like(n)
    dn_dlnk = np.zeros_like(n)

    dn_dlnk = dndk_upwind(n, k, dlnk)*k
    d2n_dlnk2[1:-1] = (n[2:] - 2 * n[1:-1] + n[:-2]) / (dlnk**2)

    d2n_dk2 = (d2n_dlnk2 - dn_dlnk) / (k**2)

    return d2n_dk2


def apply_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6, fit_points = 2):

    n = n.copy()

    n[:buf_left] = n_left_fix
    idx_fit = slice(-(buf_right + fit_points), -buf_right)
    log_k_fit = np.log(k[idx_fit])
    log_n_fit = np.log(n[idx_fit])

    slope, intercept = np.polyfit(log_k_fit, log_n_fit, 1)
    
    log_k_tail = np.log(k[-buf_right:])
    n[-buf_right:] = np.exp(slope * log_k_tail + intercept)

    return n

def rhs(t, n, k, n_left_fix, dlnk , eta1, eta2):

    n = apply_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6, fit_points = 4)

    dndk = dndk_upwind(n, k, dlnk)
    d2ndk2 = d2ndk2_center(n, k, dlnk)
    
    N = np.trapezoid(n, k)

    omega_k = k**0.5
    domega_dk = 0.5 * (k**(-0.5))

    Phi = -eta1 * k * dndk / n + eta2 * (k**2) * (dndk / n)**2

    dndt_action = - (N**3) * (
        -eta1 * (1/n) * dndk + eta1 * (k / (n**2)) * (dndk**2) - eta1 * (k/n) * d2ndk2 + eta2 * (2*k/(n**2)) * (dndk**2)
          - 2 * eta2 * ((k**2)/(n**3)) * (dndk**3) + 2 * eta2 * ((k**2)/(n**2)) * dndk * d2ndk2
        )
    dndt = omega_k * dndt_action - (N**3) * domega_dk * Phi
    dndt[:6] = 0.0
    dndt[-1:] = 0.0

    return dndt

def rk4_step(rhs, t, n, dt, k, n_left_fix, dlnk , eta1, eta2):
    k1 = rhs(t, n, k, n_left_fix, dlnk , eta1, eta2)
    k2 = rhs(t + dt/2, n + dt/2 * k1, k, n_left_fix, dlnk , eta1, eta2)
    k3 = rhs(t + dt/2, n + dt/2 * k2, k, n_left_fix, dlnk , eta1, eta2)
    k4 = rhs(t + dt,   n + dt   * k3, k, n_left_fix, dlnk , eta1, eta2)

    return n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def crank_nicolson_step(rhs, t, n, dt, k, n_left_fix, dlnk , eta1, eta2, n_iter = 4):
    f_n = rhs(t, n, k, n_left_fix, dlnk , eta1, eta2)             

    n_next = n + dt * f_n                    

    for _ in range(n_iter):
        f_next = rhs(t + dt, n_next, k, n_left_fix, dlnk , eta1, eta2)  
        n_next = n + (dt/2) * (f_n + f_next)  

    return n_next

def integrate(rhs, n0, t_span, dt, k, n_left_fix, dlnk , eta1, eta2):
    t0, tf = t_span
    n_step = int(np.ceil((tf-t0)/dt))

    t = t0
    n = n0.copy()

    ts = [t]
    ns = [n.copy()]

    for _ in range(n_step):
        #n = rk4_step(rhs, t, n, dt, k, n_left_fix, dlnk , eta1, eta2)
        n = crank_nicolson_step(rhs, t, n, dt, k, n_left_fix, dlnk , eta1, eta2, n_iter = 4)
        n = np.maximum(n, 1e-2)

        t += dt
        ts.append(t)
        ns.append(n.copy())

    return np.array(ts), np.array(ns)


def plot_results(ts, ns, n_snap, k, save_path = None, show = True):

    n_snapshots = n_snap
    n_kz = initial_condition(1, k, -5/6)
    idx = np.linspace(0, len(ts) - 1, n_snapshots).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    fig, ax = plt.subplots()
    for i, j in enumerate(idx):
        t = ts[j]
        ax.plot(k, ns[j], color=colors[i], label=f"t = {ts[j]:.2f}")
    ax.plot(k, n_kz, color = "black", label = "Soluzione di Kolmogorov - Zakharov", ls = "--")

    ax.set_title("Numerical integration of our MMT differential model")
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