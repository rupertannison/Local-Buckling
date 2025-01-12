import FEM_solver

import numpy as np

def get_imperfection_ratio(domain, corners):
    """Return array of relative imperfection sizes, all <=1"""

    intervals = FEM_solver.open_intervals_list(domain, corners)

    intervals[0] *= np.pi
    intervals[-1] *= np.pi
    imperfections_scaled = intervals/max(intervals)

    for i in range(len(intervals)):
        if i%2 == 1:
            imperfections_scaled[i] *= -1

    return imperfections_scaled

def get_imperfection_array(domain, corners, imp_factor):
    """Returns array of imperfection sizes - uses internal b/imp_factor"""

    max_internal = np.max(np.abs(np.diff(corners)))
    max_external = max(corners[0]-domain[0],domain[-1]-corners[-1]) # May choose to include outstand imperfection as limiting factor
    return get_imperfection_ratio(domain,corners)*max_internal/imp_factor #max(max_internal/imp_factor,max_external/75) - edit to include b/75 if needed


""" Example sections """

def Example_C_Section(imp_factor, nom_num_el):
    """Returns L, t, num_el, domain, corners, imperfections, a0, node_pos, stress_LB, shape"""
    L = 88.2/1000 # Initial L guess can be used, then repace with the true L for the section after the calculation has been run once
    t = 1/1000

    domain = np.array([0, 200])/1000 # The bounds of the y domain - usually start from y=0. Measured along centreline of section thickness.
    corners = np.array([35, 165])/1000 # The y-coordinates of the corners. Measured along centerline of section thickness

    imperfections = get_imperfection_array(domain, corners, imp_factor) # Amplitudes of piecewise linear and half sinusoidal imperfection shape
    a0, node_pos, num_el = FEM_solver.convert_to_FEM(domain, corners, imperfections, nom_num_el)

    stress_LB = 48.61e6
    shape = "C"

    return L, t, num_el, domain, corners, imperfections, a0, node_pos, stress_LB, shape


def Example_Lipped_C_Section(imp_factor, nom_num_el):
    """Returns L, t, num_el, domain, corners, imperfections, a0, node_pos, stress_LB, shape"""
    L = 87.3/1000 # Initial L guess can be used, then repace with the true L for the section after the calculation has been run once
    t = 1/1000

    domain = np.array([0, 500])/1000 # The bounds of the y domain - usually start from y=0. Measured along centreline of section thickness.
    corners = np.array([40, 160, 340, 460])/1000 # The y-coordinates of the corners. Measured along centerline of section thickness

    imperfections = get_imperfection_array(domain, corners, imp_factor) # Amplitudes of piecewise linear and half sinusoidal imperfection shape
    a0, node_pos, num_el = FEM_solver.convert_to_FEM(domain, corners, imperfections, nom_num_el)

    stress_LB = 29.56e6
    shape = "LC"

    return L, t, num_el, domain, corners, imperfections, a0, node_pos, stress_LB, shape



# Define your own sections by copying an example and replacing the variables (L, t, domain, corners, stress_LB, shape)
# Currently programmed shape codes for plots are: C, LC, Z, LZ, O, LO, A, LA
# If using section of other shape, don't call plot_deflected_shape from main.py
