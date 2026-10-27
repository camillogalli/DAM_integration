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