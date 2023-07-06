import torch
import numpy as np
from torch import nn
from scipy.interpolate import RegularGridInterpolator
import copy
from lsdo_airfoil import CONTROL_POINTS_FOLDER


scaler_valued_nn_model = nn.Sequential(
            nn.Linear(35, 180), nn.ReLU(),
            nn.Linear(180, 160), nn.SELU(),
            nn.Linear(160, 140), nn.SELU(),
            nn.Linear(140, 120), nn.SELU(),
            nn.Linear(120, 100), nn.GELU(),
            nn.Linear(100, 80), nn.LeakyReLU(),
            nn.Linear(80, 1),
)


X_min_numpy = np.array([
                -3.87714524e-03, -6.21114345e-03, -3.65010835e-03, -5.48448414e-04,
                1.04316720e-03, -3.44090629e-04,  -1.91351119e-03, -2.22159643e-03,
                -2.74770800e-03, -4.31647711e-03, -5.94483502e-03, -9.50526167e-03,
                -1.18035562e-02, -1.22448076e-02, -8.20550136e-03, -3.83688067e-03,
                -2.66918913e-04,  5.67624951e-03,  1.92390252e-02,  2.55834870e-02,
                3.14692594e-02,  3.43126804e-02,   3.81270386e-02,  4.34582904e-02,
                4.47864607e-02,  4.02273424e-02,   3.80498208e-02,  2.97566336e-02,
                2.03249976e-02,  1.10881981e-02,   4.26956685e-03, -5.45227900e-04,
                -8.00000000e+00,  1.00000000e+05,  0.00000000e+00]
            )

X_max_numpy = np.array([
                1.64971128e-03, 5.25048282e-03, 1.47131169e-02, 3.03167850e-02,
                4.73764949e-02, 6.15255609e-02, 7.35139325e-02, 8.21573734e-02,
                8.81158486e-02, 9.02919322e-02, 8.93072858e-02, 8.19384754e-02,
                7.00145736e-02, 5.29626682e-02, 3.25598940e-02, 1.39800459e-02,
                1.76265929e-02, 4.02436182e-02, 7.17813671e-02, 1.06165685e-01,
                1.40150547e-01, 1.67483926e-01, 1.88060194e-01, 2.04852015e-01,
                2.15405628e-01, 2.26217642e-01, 2.21330181e-01, 1.99092031e-01,
                1.54896125e-01, 1.00657888e-01, 4.71989214e-02, 1.23437112e-02,
                1.70000000e+01, 8.00000000e+06, 6.00000000e-01]
                )



def get_stall_interpolants(cl_model, cd_model, airfoil_name, num_pts_alfa=50, num_pts_Re=40, num_pts_M=20):
    aoa_range = np.linspace(0, 17, num_pts_alfa)
    Re_range = np.linspace(1e5, 8e6, num_pts_Re)
    M_range = np.linspace(0, 0.6, num_pts_M)

    var1, var2, var3 = np.meshgrid(aoa_range, Re_range, M_range)#, indexing='ij')
    combinations = np.stack((var1, var2, var3), axis=-1).reshape(-1, 3)

    control_points = np.load(CONTROL_POINTS_FOLDER / f'{airfoil_name}.npy')

    control_points_exp = np.tile(control_points.reshape(32, 1), combinations.shape[0])
    inputs = np.zeros((combinations.shape[0], 35))
    inputs[:, 32:35] = combinations
    inputs[:, 0:32] = control_points_exp.T

    scaled_inputs = (inputs - X_min_numpy) / (X_max_numpy - X_min_numpy)

    cl_outputs = cl_model(torch.Tensor(scaled_inputs)).detach().numpy().reshape(num_pts_alfa, num_pts_Re, num_pts_M)
    max_cl = np.max(cl_outputs, axis=0)
    max_indices = np.where(cl_outputs==max_cl)
    
    aoa_stall = aoa_range[max_indices[0]].reshape(num_pts_Re, num_pts_M)
    interp_Cl_stall = RegularGridInterpolator((Re_range, M_range), max_cl, bounds_error=False)
    interp_alpha_stall = RegularGridInterpolator((Re_range, M_range), aoa_stall, bounds_error=False)

    
    cd_outputs = cd_model(torch.Tensor(scaled_inputs)).detach().numpy().reshape(num_pts_alfa, num_pts_Re, num_pts_M)
    cd_stall = cd_outputs[max_indices].reshape(num_pts_Re, num_pts_M)
    interp_Cd_stall = RegularGridInterpolator((Re_range, M_range), cd_stall, bounds_error=False)


    return interp_Cl_stall, interp_Cd_stall, interp_alpha_stall, control_points


if __name__ == '__main__':
    cl_neural_net_model = copy.deepcopy(scaler_valued_nn_model)
    cl_neural_net_model.load_state_dict(torch.load(f'/home/marius/Desktop/packages/lsdo_lab/lsdo_airfoil/lsdo_airfoil/models/scalar_valued_regressions/Cl_model'))
    cl_neural_net_model.eval()
    cl_neural_net_model.requires_grad_(False)


    cd_neural_net_model = copy.deepcopy(scaler_valued_nn_model)
    cd_neural_net_model.load_state_dict(torch.load(f'/home/marius/Desktop/packages/lsdo_lab/lsdo_airfoil/lsdo_airfoil/models/scalar_valued_regressions/Cd_model'))
    cd_neural_net_model.eval()
    cd_neural_net_model.requires_grad_(False)

    interp_Cl_stall, interp_Cd_stall, interp_alpha_stall = get_stall_interpolants(cl_model=cl_neural_net_model, cd_model=cd_neural_net_model, airfoil_name='boeing_vertol_vr_12')