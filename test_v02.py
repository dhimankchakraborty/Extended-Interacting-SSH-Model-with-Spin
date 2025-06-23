import numpy as np
from functions import *


tot_sites = 4
N_e_up = 2
N_e_down = 2
J_11 = 1
J_1 = 2
J_33 = 3
J_3 = 4

basis_set = basis_set_generator(tot_sites, N_e_up, N_e_down)

hamiltonian = hamiltonian_matrix_generator(basis_set, tot_sites, J_11, J_1, J_33, J_3)

print(hamiltonian)
