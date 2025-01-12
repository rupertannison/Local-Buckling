import Integrals

import numpy as np

def convert_to_FEM(domain, corners, imperfections, num_el):
    """Find a vector to describe a given initial imperfection - linear for free edges and sines for internal elements, with amplitudes given"""
    
    # Check inputs are consistent lengths
    try:
        assert len(imperfections) == len(corners)+1
    except:
        raise NameError('Imperfection vector should contain one more entry than the corners list')

    # Create list of interval lengths between corners
    intervals = open_intervals_list(domain, corners)
    
    # Create list of the number of elements between corners
    stable_num_el = False
    while not(stable_num_el):
        num_el_list = np.zeros(len(corners)+1)
        for i in range(len(num_el_list)):
            num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
            if num_el_list[i] == 0:
                raise TypeError("Part of element modelled by 0 elements, increase number of elements")
        if num_el == int(np.sum(num_el_list)):
            stable_num_el = True
        num_el = int(np.sum(num_el_list))
    
    # Create node_pos with correctly spaced elements 
    node_pos = np.array([])
    node_pos = np.append(node_pos, np.linspace(domain[0], corners[0], num=int(num_el_list[0]), endpoint=False))
    for i in range(len(corners)-1):
        node_pos = np.append(node_pos, np.linspace(corners[i], corners[i+1], num=int(num_el_list[i+1]), endpoint=False))
    node_pos = np.append(node_pos, np.linspace(corners[-1], domain[-1], num=int(num_el_list[-1]), endpoint=False))
    node_pos = np.append(node_pos, np.array([domain[-1]]))

    # Create list of values and grads
    a_array = np.zeros((2*(num_el+1), 1))
    for i in range(num_el+1):
        if node_pos[i]<=corners[0]:
            a_array[2*i][0] = imperfections[0]*(corners[0]-domain[0]-node_pos[i])/(corners[0]-domain[0])
            a_array[2*i+1][0] = -1*imperfections[0]/(corners[0]-domain[0])
        for j in range(len(corners)-1):
            if (node_pos[i]>=corners[j]) & (node_pos[i]<corners[j+1]):
                a_array[2*i][0] = imperfections[j+1]*np.sin(np.pi*(node_pos[i]-corners[j])/(corners[j+1]-corners[j]))
                a_array[2*i+1][0] = imperfections[j+1]*np.cos(np.pi*(node_pos[i]-corners[j])/(corners[j+1]-corners[j]))*np.pi/(corners[j+1]-corners[j])
        if node_pos[i]>=corners[-1]:
            a_array[2*i][0] = imperfections[-1]*(node_pos[i]-corners[-1])/(domain[-1]-corners[-1])
            a_array[2*i+1][0] = imperfections[-1]/(domain[-1]-corners[-1])

    # Return vector solution a
    return(a_array, node_pos, num_el)


def open_intervals_list(domain, corners):
    """Return list of intervals between corners"""
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]
    return intervals


def FEM1D(domain, corners, a0_gl, b_gl, ABCF, L, v):
    """Find solution to displacement field, a, given initial guess of solution, b"""
    
    num_el = int(len(a0_gl)/2-1)
    corner_pos = np.zeros(len(corners))
    
    # Verify inputs are the correct shape
    try:
        assert a0_gl.shape == b_gl.shape
    except:
        raise NameError('input a0 and b are different shapes')
    try:
        assert domain.shape == (2,)
    except:
        raise NameError('input domain is wrong shape, should be: (2,)')

    # Create list of interval lengths between corners
    intervals = open_intervals_list(domain, corners)
    
    # Create list of the number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
        if num_el_list[i] == 0:
            raise TypeError("Part of element modelled by 0 elements, increase number of elements")
    num_el_check = np.sum(num_el_list)
    try:
        assert num_el == num_el_check
    except:
        raise NameError('Number of elements after dividing section between corners different to initial num_el')
    
    # Create node_pos with correctly spaced elements 
    node_pos = np.array([])
    node_pos = np.append(node_pos, np.linspace(domain[0], corners[0], num=int(num_el_list[0]), endpoint=False))
    corner_pos[0] = int(len(node_pos))
    for i in range(len(corners)-1):
        node_pos = np.append(node_pos, np.linspace(corners[i], corners[i+1], num=int(num_el_list[i+1]), endpoint=False))
        corner_pos[i+1] = int(len(node_pos))
    node_pos = np.append(node_pos, np.linspace(corners[-1], domain[-1], num=int(num_el_list[-1]), endpoint=False))
    node_pos = np.append(node_pos, np.array([domain[-1]]))
    
    # Extract list of element a0's
    a0_el = np.zeros((num_el, 4, 1))
    for i in range(num_el):
        a0_el[i] = np.array([[a0_gl[2*i][0]], [a0_gl[2*i+1][0]], [a0_gl[2*i+2][0]], [a0_gl[2*i+3][0]]])

    # Extract list of element b's
    b_el = np.zeros((num_el, 4, 1))
    for i in range(num_el):
        b_el[i] = np.array([[b_gl[2*i][0]], [b_gl[2*i+1][0]], [b_gl[2*i+2][0]], [b_gl[2*i+3][0]]])

    # Make a list of element matricies
    K0_el = np.zeros((num_el, 4, 4))
    for i in range(num_el):
        start = node_pos[i]
        end = node_pos[i+1] #
        K0_el[i] += Integrals.integral_1(start, end) + Integrals.integral_2(start, end, ABCF[0]) + Integrals.integral_3(start, end, ABCF[1])
        K0_el[i] += Integrals.integral_4(start, end, a0_el[i], ABCF[2]) + Integrals.integral_5(start, end, b_el[i], ABCF[2]) + Integrals.integral_6(start, end, a0_el[i], b_el[i], ABCF[2])
    
    # Make a list of RHS vectors
    RHS_vec_el = np.zeros((num_el, 4, 1))
    for i in range(num_el):
        start = node_pos[i]
        end = node_pos[i+1]
        RHS_vec_el[i] += Integrals.integral_7(start, end, b_el[i], ABCF[2]) + Integrals.integral_8(start, end, a0_el[i], b_el[i], ABCF[2]) + Integrals.integral_9(start, end, a0_el[i], ABCF[3])

    # Assemble to global matrix
    K_gl = np.zeros((2*(num_el+1), 2*(num_el+1)))
    for i in range(num_el):
        for j in range(4):
            for k in range(4):
                K_gl[2*i+j][2*i+k] += K0_el[i][j][k]
    
    # Assemble to global RHS vector
    RHS_vec_gl = np.zeros((2*(num_el+1), 1))
    for i in range(num_el):
        for j in range(4):
            RHS_vec_gl[2*i+j][0] += RHS_vec_el[i][j][0]

    # Modify K & RHS to include BC of y=0 points
    # BC currently are nodes pinned
    for i in range(len(corners)):
        RHS_vec_gl[2*int(corner_pos[i])][0] = 0
        K_gl[2*int(corner_pos[i])] = np.zeros(2*(num_el+1))
        for j in range(2*(num_el+1)):
            K_gl[j][2*int(corner_pos[i])] = 0
        K_gl[2*int(corner_pos[i])][2*int(corner_pos[i])] = 1

    # Modify K to include free edge BC
    K_gl[-1][-2] -= (np.pi/L)**2*v
    K_gl[1][0] += (np.pi/L)**2*v
    K_gl[-2][-1] += (np.pi/L)**2*(2-v)
    K_gl[0][1] -= (np.pi/L)**2*(2-v)

    # Find Solution a_result
    a_result = np.zeros((2*(num_el+1), 1))
    a_result += np.linalg.solve(K_gl, RHS_vec_gl)

    # Return Solution
    return (a_result, node_pos)


def iterate_for_Y(domain, corners, a0, b0, U0, L, t, v, tol=1e-10):
    """Runs the FEM iteration to find failure load given imperfection and initial guess"""

    # Calc constants
    gamma = L*t**2/(6*(1-v**2))
    A = 2*(np.pi/L)**2
    B = (np.pi/L)**2*((np.pi/L)**2-2*U0/gamma)
    C = L/(2*gamma)*(np.pi/L)**4
    F = 2*U0/gamma*(np.pi/L)**2
    ABCF = np.array([A, B, C, F])

    # Iterate for solution
    iterations = 0
    a_old = b0
    soltuion_found = False
    while not(soltuion_found):
        a_new, node_pos = FEM1D(domain, corners, a0, a_old, ABCF, L, v)
        iterations += 1
        change = a_new - a_old
        #print(max(change))
        less_than_tol = np.zeros(len(change))
        for i in range(len(change)):
            if abs(change[i]) < tol:
                less_than_tol[i] = True
            else:
                less_than_tol[i] = False
        if np.all(less_than_tol):
            soltuion_found = True
        else:
            a_old = a_new

    return (a_new, node_pos)
