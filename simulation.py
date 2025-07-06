import numpy as np
import copy
import scipy as sp
from itertools import combinations
import matplotlib.pyplot as plt
from numba import njit, prange
from functions import *




tot_sites = 8
N_e_up = tot_sites // 2
N_e_down = N_e_up
J_11 = 0.8
J_1 = 0.1
J_33 = 0.1
J_3 = 0.1
U = 10


basis_set = generator_basis_set(tot_sites, N_e_up, N_e_down)
dim = len(basis_set)

H = generate_hamiltonian_matrix(basis_set, dim, tot_sites, U, J_1, J_11, J_3, J_33)


e_val, e_vec = np.linalg.eigh(H)


ground_state = e_vec[:, 0]

spin_correlation_matrix = generate_spin_correlation(ground_state, basis_set, dim, tot_sites)

plt.matshow(spin_correlation_matrix)
plt.title(f"Ground-State Spin Correlation\n$J_{{11}}$: {J_11}, $J_1$: {J_1}, $J_3$: {J_3}, $J_{{33}}$: {J_33} & $U$: {U}")
plt.grid()
plt.colorbar()
plt.show()


pos_sp_wf = generate_pos_sp_wf(ground_state, basis_set, tot_sites)
plt.plot(pos_sp_wf[0], label="Spin-up")
plt.plot(pos_sp_wf[1], label="Spin-down")
plt.title(f"Ground-State Wavefunction vs Site\n$J_{{11}}$: {J_11}, $J_1$: {J_1}, $J_3$: {J_3}, $J_{{33}}$: {J_33} & $U$: {U}")
plt.xlabel("Site")
plt.ylabel("Wavefunction")
plt.grid()
plt.legend(loc='upper center')
plt.show()






