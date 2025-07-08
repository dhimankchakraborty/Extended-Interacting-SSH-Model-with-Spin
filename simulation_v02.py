import numpy as np
import copy
import scipy as sp
from itertools import combinations
import matplotlib.pyplot as plt
from numba import njit, prange
from functions import *
from scipy.sparse.linalg import eigsh
import time




tot_sites = 12
N_e_up = tot_sites // 2
N_e_down = N_e_up
J_11 = 0.1
J_1 = 1
J_33 = 0
J_3 = 0
U = 10


start_tot_time = time.time()
basis_set = generator_basis_set(tot_sites, N_e_up, N_e_down)
dim = len(basis_set)
# print(basis_set)


H_sparse = generate_hamiltonian_matrix_sparse(basis_set, dim, tot_sites, U, J_1, J_11, J_3, J_33)
H_sparse = H_sparse.tocsc()  # Convert for efficient arithmetic

start = time.process_time()
e_val, e_vec = eigsh(H_sparse, k=1, which="SA")
end = time.process_time()
print(end - start)
ground_state = e_vec[:, 0]


spin_correlation_matrix = generate_spin_correlation(ground_state, basis_set, dim, tot_sites)
pos_sp_wf = normalize(position_space_particle_probability(ground_state, basis_set, dim, tot_sites))
end_tot_time = time.time()
print(end_tot_time - start_tot_time)
# print(ground_state)
# print(pos_sp_wf)
# print()



# for i, e in enumerate(e_val):
#     if np.abs(e) < 0.01:
#         print(f'{i + 1} ---- {e}')


plt.matshow(spin_correlation_matrix)
plt.title(f"Ground-State Spin Correlation\n$J_{{11}}$: {J_11}, $J_1$: {J_1}, $J_3$: {J_3}, $J_{{33}}$: {J_33} & $U$: {U}\nNo of Electron :: Spin-up:{N_e_up}, Spin-down:{N_e_down}")
plt.grid()
plt.colorbar()
plt.show()


plt.plot(pos_sp_wf[0], label="Spin-up")
plt.plot(pos_sp_wf[1], label="Spin-down")
plt.title(f"Ground-State Wavefunction vs Site\n$J_{{11}}$: {J_11}, $J_1$: {J_1}, $J_3$: {J_3}, $J_{{33}}$: {J_33} & $U$: {U}\nNo of Electron :: Spin-up:{N_e_up}, Spin-down:{N_e_down}")
plt.xlabel("Site")
plt.ylabel("Wavefunction")
plt.ylim(-1, 1)
plt.grid()
plt.legend(loc='upper center')
plt.show()


plt.plot(pos_sp_wf[0]**2, label="Spin-up")
plt.plot(pos_sp_wf[1]**2, label="Spin-down")
plt.title(f"Ground-State Wavefunction vs Site\n$J_{{11}}$: {J_11}, $J_1$: {J_1}, $J_3$: {J_3}, $J_{{33}}$: {J_33} & $U$: {U}")
plt.xlabel("Site")
plt.ylabel("Wavefunction")
plt.ylim(0, 1)
plt.grid()
plt.legend(loc='upper center')
plt.show()






