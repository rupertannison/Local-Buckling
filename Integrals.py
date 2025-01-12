import Shape_Functions

import numpy as np

def integral_1(start, end):
    """Returns 4x4 matrix result of integral 1 (BB), using 2 guass points"""
    result = np.zeros((4,4))
    gauss_pnts = [-0.5773502692, 0.5773502692]
    gauss_wghts = [1, 1]

    for i in range(2):
        B_i = Shape_Functions.shape_fn_B(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        result += np.transpose(B_i)*B_i*(end-start)/2*gauss_wghts[i]
    result_2 = result
    return result_2

def integral_2(start, end, A):
    """Returns 4x4 matrix result of integral 2 (NB), using 3 guass points"""
    result = np.zeros((4,4))
    gauss_pnts = [-0.7745966692, 0, 0.7745966692]
    gauss_wghts = [0.5555555555556, 0.888888888889, 0.5555555555556]

    for i in range(3):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        B_i = Shape_Functions.shape_fn_B(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        result += np.transpose(N_i)*B_i*(end-start)/2*gauss_wghts[i]
    result_2 = -1*A*result
    return result_2

def integral_3(start, end, B):
    """Returns 4x4 matrix result of integral 3 (NN), using 4 guass points"""
    result = np.zeros((4,4))
    gauss_pnts = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
    gauss_wghts = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]

    for i in range(4):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        result += np.transpose(N_i)*N_i*(end-start)/2*gauss_wghts[i]
    result_2 = B*result
    return result_2

def integral_4(start, end, a0_elem, C):
    """Returns 4x4 matrix result of integral 4 (N(Na0)^2N), using 7 guass points"""
    result = np.zeros((4,4))
    gauss_pnts = [-0.9491079123, -0.7415311856, -0.4058451514, 0, 0.4058451514, 0.7415311856, 0.9491079123]
    gauss_wghts = [0.1294849662, 0.2797053915, 0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

    # Verify input a0_elem is the correct shape
    try:
        assert a0_elem.shape == (4, 1)
    except:
        raise NameError('input a0 is wrong shape, should be: (4,1')

    for i in range(7):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        N_a0_i = np.dot(N_i,a0_elem)[0][0]
        result += np.transpose(N_i)*N_i*(N_a0_i)**2*(end-start)/2*gauss_wghts[i]
    result_2 = 2*C*result
    return result_2

def integral_5(start, end, b_elem, C):
    """Returns 4x4 matrix result of integral 5 (N(Nb)^2N), using 7 guass points"""
    result = np.zeros((4,4))
    gauss_pnts = [-0.9491079123, -0.7415311856, -0.4058451514, 0, 0.4058451514, 0.7415311856, 0.9491079123]
    gauss_wghts = [0.1294849662, 0.2797053915, 0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

    # Verify input b_elem is the correct shape
    try:
        assert b_elem.shape == (4, 1)
    except:
        raise NameError('input b is wrong shape, should be: (4,1')

    for i in range(7):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        N_b_i = np.dot(N_i,b_elem)[0][0]
        result += np.transpose(N_i)*N_i*(N_b_i)**2*(end-start)/2*gauss_wghts[i]
    result_2 = 3*C*result
    return result_2

def integral_6(start, end, a0_elem, b_elem, C):
    """Returns 4x4 matrix result of integral 6 (N(Nb)(Na0)N), using 7 guass points"""
    result = np.zeros((4,4))
    gauss_pnts = [-0.9491079123, -0.7415311856, -0.4058451514, 0, 0.4058451514, 0.7415311856, 0.9491079123]
    gauss_wghts = [0.1294849662, 0.2797053915, 0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

    # Verify inputs are the correct shape
    try:
        assert a0_elem.shape == (4, 1)
    except:
        raise NameError('input a0 is wrong shape, should be: (4,1')
    try:
        assert b_elem.shape == (4, 1)
    except:
        raise NameError('input b is wrong shape, should be: (4,1')

    for i in range(7):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        N_a0_i = np.dot(N_i,a0_elem)[0][0]
        N_b_i = np.dot(N_i,b_elem)[0][0]
        result += np.transpose(N_i)*N_i*(N_a0_i)*(N_b_i)*(end-start)/2*gauss_wghts[i]
    result_2 = 6*C*result
    return result_2

def integral_7(start, end, b_elem, C):
    """Returns 4x1 vector result of integral 7 (N(Nb)^3), using 7 guass points"""
    result = np.zeros((4,1))
    gauss_pnts = [-0.9491079123, -0.7415311856, -0.4058451514, 0, 0.4058451514, 0.7415311856, 0.9491079123]
    gauss_wghts = [0.1294849662, 0.2797053915, 0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

    # Verify input b_elem is the correct shape
    try:
        assert b_elem.shape == (4, 1)
    except:
        raise NameError('input b is wrong shape, should be: (4,1')

    for i in range(7):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        N_b_i = np.dot(N_i,b_elem)[0][0]
        result += np.transpose(N_i)*(N_b_i)**3*(end-start)/2*gauss_wghts[i]
    result_2 = 2*C*result
    return result_2

def integral_8(start, end, a0_elem, b_elem, C):
    """Returns 4x1 vector result of integral 8 (N(Nb)^2(Na0)), using 7 guass points"""
    result = np.zeros((4,1))
    gauss_pnts = [-0.9491079123, -0.7415311856, -0.4058451514, 0, 0.4058451514, 0.7415311856, 0.9491079123]
    gauss_wghts = [0.1294849662, 0.2797053915, 0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

    # Verify inputs are the correct shape
    try:
        assert a0_elem.shape == (4, 1)
    except:
        raise NameError('input a0 is wrong shape, should be: (4,1')
    try:
        assert b_elem.shape == (4, 1)
    except:
        raise NameError('input b is wrong shape, should be: (4,1')

    for i in range(7):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        N_a0_i = np.dot(N_i,a0_elem)[0][0]
        N_b_i = np.dot(N_i,b_elem)[0][0]
        result += np.transpose(N_i)*(N_b_i)**2*(N_a0_i)*(end-start)/2*gauss_wghts[i]
    result_2 = 3*C*result
    return result_2

def integral_9(start, end, a0_elem, F):
    """Returns 4x1 vector result of integral 9 (N(Na0)), using 7 guass points"""
    result = np.zeros((4,1))
    gauss_pnts = [-0.9491079123, -0.7415311856, -0.4058451514, 0, 0.4058451514, 0.7415311856, 0.9491079123]
    gauss_wghts = [0.1294849662, 0.2797053915, 0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

    # Verify input a0_elem is the correct shape
    try:
        assert a0_elem.shape == (4, 1)
    except:
        raise NameError('input b is wrong shape, should be: (4,1')

    for i in range(7):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        N_a0_i = np.dot(N_i,a0_elem)[0][0]
        result += np.transpose(N_i)*(N_a0_i)*(end-start)/2*gauss_wghts[i]
    result_2 = F*result
    return result_2

def integral_10_elem(start, end, elem_sol, G):
    """Returns result of integral 10 (YY), using 4 guass points, elem_sol is 2 node solution vector 4x1 for the element"""
    result = 0
    gauss_pnts = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
    gauss_wghts = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]

    for i in range(4):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        result += np.dot(N_i,elem_sol)[0][0]**2*(end-start)/2*gauss_wghts[i]
    result_2 = G*result
    return result_2

def integral_11_elem(start, end, elem_sol, elem_initial, G):
    """Returns result of integral 11 (YY0), using 4 guass points, elem_sol is 2 node solution vector 4x1 for the element
    elem_initial is 2 node initial vector 4x1 for the element"""
    result = 0
    gauss_pnts = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
    gauss_wghts = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]

    for i in range(4):
        N_i = Shape_Functions.shape_fn_N(start, end, gauss_pnts[i]*(end-start)/2+(end+start)/2)
        result += np.dot(N_i,elem_sol)[0][0]*np.dot(N_i,elem_initial)[0][0]*(end-start)/2*gauss_wghts[i]
    result_2 = 2*G*result
    return result_2