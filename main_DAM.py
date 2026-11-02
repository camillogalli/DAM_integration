import numpy as np
from DAM import integrate, plot_results, k_space, initial_condition, rhs
from scipy.integrate import solve_ivp

eta1 = 216
eta2 = 162

k, dlnk = k_space(1e-3, 10, 300)
n_k = initial_condition(1.0, k, -2/3)
n_left_fix = n_k[:6].copy()

print("Test valutazione singola del rhs:")
r0 = rhs(0, n_k, k, n_left_fix, dlnk, eta1, eta2)
print("max |rhs| iniziale:", np.max(np.abs(r0)))
print("rhs finito ovunque:", np.all(np.isfinite(r0)))

ts = np.linspace(0.0, 1.0, 100001)

ns = solve_ivp(rhs, (0, ts[-1]), n_k, args=(k, n_left_fix, dlnk, eta1, eta2),
                 method="BDF", t_eval=ts, rtol=1e-6, atol=1e-10)

print("positiva ovunque:", np.all(ns.y > 0))
print("finito ovunque:", np.all(np.isfinite(ns.y)))

ln_k = np.log(k)
p_final = np.gradient(np.log(np.maximum(ns.y[:, -1], 1e-300)), ln_k)
p_final = np.mean(p_final[135:165])

print("errore relativo sull'ultimo esponente di power law: ", np.abs(p_final - 5/6))

#ts, ns = integrate(rhs, n_k, (0.0, 0.1),  1e-6, k, n_left_fix, dlnk, eta1, eta2)

#plot_results(ts, ns, n_snap = 5, k = k, save_path = "outputs_DAM/DAM_results.png", show = False)
plot_results(ns.t, ns.y.T, n_snap = 5, k = k, save_path = "outputs_DAM/DAM_results.png", show = False)
