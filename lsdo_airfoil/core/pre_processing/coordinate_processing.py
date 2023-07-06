import numpy as np
import csdl
from scipy.interpolate import UnivariateSpline
from lsdo_airfoil.utils.compute_b_spline_mat import compute_b_spline_mat


class CoordinateProcessing(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('airfoil_raw_shape', types=tuple)

    def define(self):
        shape = self.parameters['airfoil_raw_shape']

        self.add_input('airfoil_camber', shape=shape)
        self.add_input('airfoil_thickness', shape=shape)

        self.add_output('control_points_camber', shape=(18, 1))
        self.add_output('control_points_thickness_raw', shape=(18, 1))

        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        camber = inputs['airfoil_camber']
        thickness = inputs['airfoil_thickness']
        shape = self.parameters['airfoil_raw_shape']
        B, B_star = compute_b_spline_mat(shape[0])

        camber_control_points = np.transpose(np.dot(B_star, camber))
        thickness_control_points = np.transpose(np.dot(B_star, thickness))

        # print(camber_control_points[1:17])
        # print(type(camber_control_points[1:17]))
        outputs['control_points_camber'] = camber_control_points
        outputs['control_points_thickness_raw'] = thickness_control_points
        
        # np.vstack((camber_control_points, thickness_control_points))

    def compute_derivatives(self, inputs, derivates):
        camber = inputs['airfoil_camber']
        thickness = inputs['airfoil_thickness']
        shape = self.parameters['airfoil_raw_shape']
        B, B_star = compute_b_spline_mat(shape[0])
        # print('B_star', B_star)
        derivates['control_points_thickness_raw', 'airfoil_thickness'] = B_star
        derivates['control_points_thickness_raw', 'airfoil_camber'] = np.zeros(B_star.shape)
        derivates['control_points_camber', 'airfoil_thickness'] = np.zeros(B_star.shape)
        derivates['control_points_camber', 'airfoil_camber'] = B_star





