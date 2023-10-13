import teaserpp_python
import numpy as np
from os.path import join
import os
import sys


solver_params = teaserpp_python.RobustRegistrationSolver.Params()
solver_params.cbar2 = 1
solver_params.noise_bound = 0.03
solver_params.estimate_scaling = False
solver_params.rotation_estimation_algorithm = (
    teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
)
solver_params.rotation_gnc_factor = 1.4
solver_params.rotation_max_iterations = 50
solver_params.rotation_cost_threshold = 1e-1
solver_params = solver_params
#print("TEASER++ Parameters are:", solver_params)
#teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)
solver = teaserpp_python.RobustRegistrationSolver(solver_params)

def solve_r(src, target):
    solver.solve(src, dst)
    solution = solver.getSolution()
    solver.reset(solver_params)
    print("* " * 10)

for ii in range(10):
    src = np.load("src.npy").astype(np.float64)
    dst = np.load("target.npy").astype(np.float64)
    solve_r(src, dst)