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

    k = np.linspace(k_min, k_max, n_points)
    dk = k[1] - k[0]
    return k, dk

def initial_condition(A, k, p):
    """
    Generate the initial condition for the transport calculation.
    Parameters:
    A (float): Amplitude of the initial condition.
    k (array): k-space array.
    p (float): spectral power.
    """

    return A * (k**(p))

def dndk(n, dk):
    """
    Calculate the derivative of n with respect to k.
    Parameters:
    n (array): Array of n values.
    dk (float): Step size in k-space.
    """
    dn_dk = np.zeros_like(n)
    dn_dk[1:-1] = (n[2:] - n[:-2]) / (2 * dk)

    # Handle the boundaries using forward and backward differences
    dn_dk[0] = (n[1] - n[0]) / dk
    dn_dk[-1] = (n[-1]- n[-2]) / dk

    return dn_dk

def apply_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6, fit_points = 2):

    """
    Applying boundary conditions, where the first 5
    points remain fixed and the last 5 are fitted with a log log 
    linear fit.
    """

    n = n.copy()

    n[:buf_left] = n_left_fix

    idx_fit = slice(-(buf_right + fit_points), -buf_right)
    log_k_fit = np.log(k[idx_fit])
    log_n_fit = np.log(n[idx_fit])

    slope, intercept = np.polyfit(log_k_fit, log_n_fit, 1)

    log_k_tail = np.log(k[-buf_right:])
    n[-buf_right:] = np.exp(slope * log_k_tail + intercept)

    return n

def rhs(n, k, n_left_fix, dk , c):
    
    n = apply_bc(n, k, n_left_fix, buf_left = 6, buf_right = 6, fit_points = 2)
    dn = dndk(n, dk)
    dndt = c*dn
    dndt[:6] = 0.0

    return dndt

def rk4_step(rhs, t, n, dt, *rhs_args):
    k1 = rhs(t, n, *rhs_args)
    k2 = rhs(t + dt/2, n + dt/2 * k1, *rhs_args)
    k3 = rhs(t + dt/2, n + dt/2 * k2, *rhs_args)
    k4 = rhs(t + dt,   n + dt   * k3, *rhs_args)

    return n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrate(rhs, n0, t_span, dt, *rhs_args):
    t0, tf = t_span
    n_step = int(np.ceil((tf-t0)/dt))

    t = t0
    n = n0.copy()

    ts = [t]
    ns = [n.copy()]

    for _ in range(n_step):
        n = rk4_step(rhs, t, n, dt, *rhs_args)
        t += dt
        ts.append(t)
        ns.append(n.copy())

    return np.array(ts), np.array(ns)

