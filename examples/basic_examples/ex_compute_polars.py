
from python_csdl_backend import Simulator
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL
import csdl
import numpy as np
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from lsdo_airfoil.utils.compute_b_spline_mat import compute_b_spline_mat
from lsdo_airfoil import AIRFOIL_COORDINATES_FOLDER
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False


# Import raw coordinates of a sample airfoil
test_airfoil_coord = np.loadtxt(AIRFOIL_COORDINATES_FOLDER / 'boeing_vertol_vr_12.txt')
upper = test_airfoil_coord[0:43, 1]
lower = test_airfoil_coord[43:, 1]
x_upper = test_airfoil_coord[0:43, 0]
x_lower = test_airfoil_coord[43:, 0]

# Interpolate the airfoil
x_range = np.linspace(0,1,301)
i_vec = np.arange(0, len(x_range))
x_interp = 1 - np.cos(np.pi/(2 * (len(x_range)-1)) * i_vec)
upper_interp = np.interp(x_interp, x_upper, upper)
lower_interp = np.interp(x_interp, x_lower, lower)

# plot airfoil
plt.figure(1)
plt.scatter(x_upper, upper, color='r', label='Raw coordinates')
plt.scatter(x_lower, lower, color='r')

plt.plot(x_interp, upper_interp, color='b', label='Interpolated coordinates')
plt.plot(x_interp, lower_interp, color='b')
plt.axis('equal')
plt.legend()


num_nodes = 50
M = 0.2
AoA = np.linspace(-5, 20, num_nodes) 
Re =  1e6


run_model = csdl.Model()
run_model.create_input('mach_number', M * np.ones((num_nodes, )))
run_model.create_input('angle_of_attack', AoA)
run_model.create_input('reynolds_number', Re * np.ones((num_nodes, )))

run_model.add_design_variable('mach_number', lower=0., upper=0.6)
run_model.add_design_variable('angle_of_attack', scaler=3e-2, lower=2, upper=10)
run_model.add_design_variable('reynolds_number', scaler=0.5e-6, lower=1e5, upper=1e6)


run_model.create_input('airfoil_upper', val=upper_interp)
run_model.create_input('airfoil_lower', val=lower_interp)



airfoil_model = AirfoilModelCSDL(
    num_nodes=num_nodes,
    compute_control_points=True,
    airfoil_name='boeing_vertol_vr_12',
    airfoil_raw_shape=(301, )
)
run_model.add(airfoil_model, 'airfoil_model')

sim = Simulator(run_model)
sim.run()

cl = sim['Cl'].flatten()
cd = sim['Cd'].flatten()

fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(10, 8))
axs[0, 0].plot(AoA, cl)
axs[0, 0].set_xlabel('angle of attack')
axs[0, 0].set_ylabel('Cl')
axs[0, 0].grid()

axs[0, 1].plot(AoA, cd)
axs[0, 1].set_xlabel('angle of attack')
axs[0, 1].set_ylabel('Cd')
axs[0, 1].grid()

axs[1, 0].plot(AoA, cl/cd)
axs[1, 0].set_xlabel('angle of attack')
axs[1, 0].set_ylabel('Cl/Cd')
axs[1, 0].grid()

axs[1, 1].plot(cd, cl)
axs[1, 1].set_xlabel('Cd')
axs[1, 1].set_ylabel('Cl')
axs[1, 1].grid()

fig.tight_layout()


plt.show()

