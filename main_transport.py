from transport import k_space, initial_condition, rhs, apply_bc, dndk, integrate, plot_results

c = -1.0

k, dlnk = k_space(1e-3, 10, 100)
n_k = initial_condition(1.0, k, -5/6)
rhs_args = (k, dlnk, c)
n_left_fix = n_k[:6].copy()

ts, ns = integrate(rhs, n_k, (0.0, 1.0),  1e-5, n_left_fix, k, dlnk, c)

plot_results(ts, ns, n_snap = 5, k = k, save_path = "outputs_transport/transport_results.png", show = False)


