import numpy as np
from DAM import integrate, plot_results, k_space, initial_condition, rhs

eta1 = 216
eta2 = 162

k, dlnk = k_space(1e-3, 10, 100)

n_k = initial_condition(1.0, k, -7/10)

n_left_fix = n_k[:6].copy()

ts, ns = integrate(rhs, n_k, (0.0, 0.1),  1e-6, k, n_left_fix, dlnk, eta1, eta2)

plot_results(ts, ns, n_snap = 5, k = k, save_path = "outputs_DAM/DAM_results.png", show = False)
