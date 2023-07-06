import numpy as np
from lsdo_utils.comps.bspline_comp import  get_bspline_mtx

def compute_b_spline_mat(num_pts):
    camber_B_spline_mat = get_bspline_mtx(18, num_pts, order = 4)
    B = camber_B_spline_mat.toarray()
    B_star = np.matmul(np.linalg.inv(np.matmul(B.T,B)), B.T)
    return B, B_star
