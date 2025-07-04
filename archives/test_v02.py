import numpy as np
from archives.functions_v01 import *


tot_sites = 4
N_e_up = 2
N_e_down = 2
J_11 = 1
J_1 = 2
J_33 = 3
J_3 = 4

basis_set = basis_set_generator(tot_sites, N_e_up, N_e_down)
for state in basis_set:
    print(state)

hamiltonian = hamiltonian_matrix_generator(basis_set, tot_sites, J_11, J_1, J_33, J_3)

for row in hamiltonian:
    for element in row:
        if element == 0:
            print("    ", end="|")
        else:
            print(element, end="|")
    print()
