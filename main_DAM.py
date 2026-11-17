import numpy as np
from DAM import integrate, plot_results, k_space, initial_condition, rhs, integrate_chunked, apply_bc, initial_condition_gaussian
from scipy.integrate import solve_ivp

eta1 = 216
eta2 = 162
p = -2/3

k, dlnk = k_space(1e-3, 1, 300)
k0 = k[15] 
n_k = initial_condition(1.0, k, -2/3)
n_k_gaus = initial_condition_gaussian(1.0, k, k0, 0.5, 0.3, -2/3)
n_left_fix = n_k[:6].copy() 

print("Test valutazione singola del rhs:")
r0 = rhs(0, n_k, k, n_left_fix, dlnk, eta1, eta2)
print("max |rhs| iniziale:", np.max(np.abs(r0)))
print("rhs finito ovunque:", np.all(np.isfinite(r0)))

#t_eval = np.linspace(0.0, 1.0, 1000001)

#ns = solve_ivp(rhs, (0, t_eval[-1]), n_k, args=(k, n_left_fix, dlnk, eta1, eta2),
#                 method="BDF", t_eval=t_eval, rtol=1e-6, atol=1e-10)

ts, ns = integrate_chunked(
    rhs, n_k, (0.0, 1), dt_chunk=1e-7,
    apply_bc_func=apply_bc,
    bc_args=(k, n_left_fix, 6, 6, 8),
    rhs_args=(k, n_left_fix, dlnk, eta1, eta2),
)

tsg, nsg = integrate_chunked(
    rhs, n_k_gaus, (0.0, 1), dt_chunk=1e-7,
    apply_bc_func=apply_bc,
    bc_args=(k, n_left_fix, 6, 6, 8),
    rhs_args=(k, n_left_fix, dlnk, eta1, eta2),
)

#print("positiva ovunque:", np.all(ns.y > 0))
print("positiva ovunque:", np.all(ns > 0))
#print("finito ovunque:", np.all(np.isfinite(ns.y)))
print("finito ovunque:", np.all(np.isfinite(ns)))

ln_k = np.log(k)
#p_final = np.gradient(np.log(np.maximum(ns.y[:, -1], 1e-30)), ln_k)
#p_final = np.mean(p_final[135:165])

p_final = np.gradient(np.log(np.maximum(ns[-1], 1e-30)), ln_k)
p_final = np.mean(p_final[135:165])

print("errore relativo sull'ultimo esponente di power law: ", np.abs(p_final + 5/6))

#ts, ns = integrate(rhs, n_k, (0.0, 0.1),  1e-6, k, n_left_fix, dlnk, eta1, eta2)

plot_results(ts, ns, n_snap = 5, k = k, save_path = "outputs_DAM/DAM_results.png", show = False)
plot_results(tsg, nsg, n_snap = 5, k = k, save_path = "outputs_DAM/DAM_results_gaussian_bump.png", show = False)
#plot_results(ns.t, ns.y.T, n_snap = 7, k = k, save_path = "outputs_DAM/DAM_results.png", show = False)
