import numpy as np
from DAM import integrate, plot_results, k_space, initial_condition, rhs

eta1 = 100
eta2 = 100

k, dlnk = k_space(1e-3, 10, 100)
k_min = np.min(k)
k_max = np.max(k)

n_k = initial_condition(1.0, k, -7/10)

n_left_fix = n_k[:6].copy()

N = np.trapezoid(n_k, k)

ts, ns = integrate(rhs, n_k, (0.0, 1.0),  1e-6, k, n_left_fix, dlnk, eta1, eta2, k_min, k_max)

plot_results(ts, ns, n_snap = 5, k = k, save_path = "outputs_DAM/DAM_results.png", show = False)
