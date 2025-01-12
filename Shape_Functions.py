import numpy as np

def shape_fn_N(start, end, y):
    """Returns vector of shape function values at given y, order [N1 M1 N2 M2]"""
    N1 = -1*(y-end)**2*(start-end+2*(start-y))/(end-start)**3
    M1 = (y-start)*(y-end)**2/(end-start)**2
    N2 = (y-start)**2*(end-start+2*(end-y))/(end-start)**3
    M2 = (y-start)**2*(y-end)/(end-start)**2
    return np.array([[N1, M1, N2, M2]])

def shape_fn_B(start, end, y):
    """Returns vector of second derivatives of shape function values at given y, order [N1 M1 N2 M2]"""
    N1 = (12*y+2*(end-start)-8*end-4*start)/(end-start)**3
    M1 = (6*y-2*start-4*end)/(end-start)**2
    N2 = (-12*y+2*(end-start)+8*start+4*end)/(end-start)**3
    M2 = (6*y-4*start-2*end)/(end-start)**2
    return np.array([[N1, M1, N2, M2]])

def shape_fn_N_derivative(start, end, y):
    """Returns vector of first derivative of shape function values at given y, order [N1 M1 N2 M2]"""
    N1 = (6*y**2-8*end*y+2*(end-start)*y-4*start*y+2*end**2-2*(end-start)*end+4*end*start)/(end-start)**3
    M1 = (3*y**2-2*start*y-4*end*y+2*start*end+end**2)/(end-start)**2
    N2 = (-6*y**2+8*start*y+2*(end-start)*y+4*end*y-2*start**2-2*(end-start)*start-4*end*start)/(end-start)**3
    M2 = (3*y**2-4*start*y-2*end*y+start**2+2*start*end)/(end-start)**2
    return np.array([[N1, M1, N2, M2]])

def shape_fn_B_derivative(start, end, y):
    """Returns vector of third derivatives of shape function values at given y, order [N1 M1 N2 M2]"""
    N1 = 12/(end-start)**3
    M1 = 6/(end-start)**2
    N2 = -12/(end-start)**3
    M2 = 6/(end-start)**2
    return np.array([[N1, M1, N2, M2]])