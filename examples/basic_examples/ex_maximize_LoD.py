
from python_csdl_backend import Simulator
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL
import csdl
import numpy as np
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


num_nodes = 1
M = 0
AoA = 2 
Re =  1e6

run_model = csdl.Model()
run_model.create_input('mach_number', M * np.ones((num_nodes, )))
run_model.create_input('angle_of_attack', AoA) #  * np.ones((num_nodes, )))
run_model.create_input('reynolds_number', Re * np.ones((num_nodes, )))

run_model.add_design_variable('mach_number', lower=0., upper=0.6)
run_model.add_design_variable('angle_of_attack', scaler=3e-2, lower=2, upper=10)
run_model.add_design_variable('reynolds_number', scaler=0.5e-6, lower=1e5, upper=1e6)


airfoil_model_1 = AirfoilModelCSDL(
    num_nodes=num_nodes,
    compute_control_points=False,
    airfoil_name='NACA_4412',
)
run_model.add(airfoil_model_1, 'airfoil_model_1')


Cl = run_model.declare_variable('Cl', shape=(num_nodes, ))
Cd = run_model.declare_variable('Cd', shape=(num_nodes, ))

run_model.register_output('LoD', Cl/Cd)

run_model.add_objective('LoD', scaler=-1e-2)
sim = Simulator(run_model)
sim.run()

prob = CSDLProblem(problem_name='airfoil_ml_test', simulator=sim)
optimizer = SLSQP(
    prob,
    maxiter=100, 
    ftol=1e-12,
)
optimizer.solve()
optimizer.print_results()

print(sim['Cd'].flatten())
print(sim['Cl'].flatten())
print(sim['mach_number'])
print(sim['angle_of_attack'])
print(sim['reynolds_number'])

print(sim['LoD'])


# sim.check_partials()
# exit()

# print('\n')
# control_points = sim['control_points'].flatten()
# cpts_camber = np.hstack(([0], control_points[0:16], [0]))
# cpts_thickness = np.hstack(([0], control_points[16:], [0]))


# B, B_star = compute_b_spline_mat(301)
# full_camber = np.matmul(B, cpts_camber)
# full_thickness = np.matmul(B, cpts_thickness)
# full_upper = full_camber + 0.5 * full_thickness
# full_lower = full_camber - 0.5 * full_thickness

# plt.plot(x_interp, full_upper, color='blue')
# plt.plot(x_interp, full_lower, color='blue')


# print(x_lower.shape)
# print(upper.shape)
# plt.scatter(x_upper, upper, color='red')
# plt.scatter(x_lower, lower, color='red')

# plt.show()
# exit()

# np.save('sample_airfoils/control_points/boeing_vertol_vr_12', sim['control_points'])
