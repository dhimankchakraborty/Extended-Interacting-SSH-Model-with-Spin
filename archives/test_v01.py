import numpy as np
from archives.functions_v01 import *


tot_sites = 8
N_e_up = 4
N_e_down = 4

basis_set = basis_set_generator(tot_sites, N_e_up, N_e_down)

print(len(basis_set))
# # print(basis_set)
# for i, state in enumerate(basis_set):
#     print(f"{i} ---- {state}")

