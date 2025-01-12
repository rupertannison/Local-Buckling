import Integrals
import FEM_solver
import Plotters

import numpy as np
import matplotlib.pyplot as plt

def get_force(a_sol, node_pos, a0, U0, L, t, E):
    """Find the force associated with a given displacement profile"""

    num_elem = node_pos.size - 1

    # Get constants
    G = E*t*(np.pi/L)**2/4
    H = E*U0*t/L

    # Form list of element solutions and list of element boundaries
    a_sol_elem = np.zeros((num_elem, 4, 1))
    a0_elem = np.zeros((num_elem, 4, 1))
    elem_boundaries = np.zeros((num_elem, 2))
    for i in range(num_elem):
        a_sol_elem[i] = np.array([[a_sol[2*i][0]], [a_sol[2*i+1][0]], [a_sol[2*i+2][0]], [a_sol[2*i+3][0]]])
        a0_elem[i] = np.array([[a0[2*i][0]], [a0[2*i+1][0]], [a0[2*i+2][0]], [a0[2*i+3][0]]])
        elem_boundaries[i] = np.array([node_pos[i], node_pos[i+1]])
    
    # Calc P
    P = 0
    for i in range(num_elem):
        P -= Integrals.integral_10_elem(elem_boundaries[i][0], elem_boundaries[i][1], a_sol_elem[i], G)
        P -= Integrals.integral_11_elem(elem_boundaries[i][0], elem_boundaries[i][1], a_sol_elem[i], a0_elem[i], G)
    P += H*(node_pos[-1]-node_pos[0])

    return P

def load_vs_U0(a_list, node_pos, a0, U0_list, L, t, E, fy, sigma_LB, normalise_PLB = False):
    """Produce plot of load (normalised by P_LB) as U0 varies (normalised as % of L)"""

    if normalise_PLB:
        P_LB = local_buckle_force(np.array([node_pos[0], node_pos[-1]]), t, sigma_LB)
        y_title = "Force (normalsied by Sigma LB)"
    else:
        P_LB = 1000
        y_title = "Force (kN)"

    P_list = np.zeros(U0_list.size)
    #failed_list = np.zeros(U0_list.size)
    #stress_list = np.zeros(U0_list.size)

    for i in range(len(U0_list)):
        P_list[i] = get_force(a_list[i], node_pos, a0, U0_list[i], L, t, E) / P_LB
        #failed_list[i] = test_if_failed(a_list[i], node_pos, a0, U0_list[i], L, E, fy)
        #stress_list[i] = get_max_stress(a_list[i], node_pos, a0, U0_list[i], L, E) / fy
        print("U0 = {}%' has a force of: {} kN".format(np.round(U0_list[i]/L*100, 3), round(P_list[i]*P_LB/1000, 3)))
    
    Plotters.simple_plot(U0_list/L*100, P_list, x_label="U0 ('%' of L)", y_label=y_title, title="Evolution of Force with U0")

    # save the data to a text file
    #np.savetxt('load_U0_data_C2_perfectish.txt', np.column_stack((U0_list*1000, P_list/1000)), header='U0 (mm),Load (kN)', delimiter=',')

def should_fail_U0(E, fy, L):
    #print("Should Fail when U0/L =", fy/E)
    #print("Which is when U0 is", str(fy/E*100) + "'%' of L")
    return(fy*L/E)

def load_vs_imp(a_list, node_pos, a0_unfactor, imp_factor_list, U0, L, t, E, fy, b_max, normalise_py = True):
    """Produce plot of failure load as imperfection varies"""

    if normalise_py:
        P_y = yeild_force(np.array([node_pos[0], node_pos[-1]]), t, fy)
        y_label = "Force (normalised by fy)"
    else:
        P_y = 1000
        y_label = "Force (kN)"

    P_list = np.zeros(imp_factor_list.size)
    #failed_list = np.zeros(imp_factor_list.size)

    for i in range(len(imp_factor_list)):
        P_list[i] = get_force(a_list[i], node_pos, a0_unfactor*b_max/imp_factor_list[i], U0, L, t, E) / P_y
        #failed_list[i] = test_if_failed(a_list[i], node_pos, a0_unfactor*b_max/imp_factor_list[i], U0, L, E, fy)
        print("Imperfection of b/{} fails at: {}".format(imp_factor_list[i], round(P_list[i]*P_y/1000,3)))
    
    plt.plot(1/imp_factor_list, P_list, marker="o")
    plt.ylim(bottom=0)
    plt.ylim(top=max(P_list)*1.2)
    plt.grid()
    plt.xlabel("Imperfection size (fraction of b)")
    plt.ylabel(y_label)
    plt.title("Variation of failure load with initial imperfetion amplitude - Fixed L")
    plt.show()
    #np.savetxt('S12_LC2_load_imp_fixed_L.txt', np.column_stack((1/imp_factor_list*100, P_list)), header='Imperfection size ( of b),Ultimate Failure Load (normalised by fy)', delimiter=',')
    

def yeild_force(domain, t, fy):
    """Return Py = force plate yield if not locally buckle"""
    return (domain[-1]-domain[0])*t*fy

def local_buckle_force(domain, t, sigma_LB):
    """Return P_cr = force plate start to locally buckle"""
    return (domain[-1]-domain[0])*t*sigma_LB

def load_vs_max_y(a_list, node_pos, a0, U0_list, L, t, E, fy, sigma_LB, normalise_PLB = True):
    """Produce plot of load (normalised by P_LB) as U0 varies - plot against max y"""

    if normalise_PLB:
        P_LB = local_buckle_force(np.array([node_pos[0], node_pos[-1]]), t, sigma_LB)
        y_title = "Force (normalsied by Sigma LB)"
    else:
        P_LB = 1000
        y_title = "Force (kN)"

    P_list = np.zeros(U0_list.size)
    y_list = np.zeros(U0_list.size)

    for i in range(len(U0_list)):
        P_list[i] = get_force(a_list[i], node_pos, a0, U0_list[i], L, t, E) / P_LB
        x, y = Plotters.extract_xy_data(a_list[i]+a0, node_pos)
        y_list[i] = abs(y[0])

    Plotters.simple_plot(y_list*1000, P_list, x_label="Free edge displacement (mm)", y_label=y_title, title="Evolution of Force with Lateral Displacement of Free Edge")
    
    # save the data to a text file
    #np.savetxt('S12_LC2_load_w_imp_200.txt', np.column_stack((y_list*1000, P_list/1000)), header='w (mm),Load (kN)', delimiter=',')


def elastic_local_buckle_stress(K,v,E,t,b):
    return K*np.pi**2*E/(12*(1-v**2))*(t/b)**2

def outstand_rho(lam_bar):
    if lam_bar<=0.748:
        return 1
    else:
        rho = (lam_bar-0.188)/lam_bar**2
        if rho>1:
            return 1
        else:
            return rho

def internal_rho(lam_bar):
    if lam_bar<=0.5+(0.085-0.055*1)**0.5:
        return 1
    else:
        rho = (lam_bar-0.055*(3+1))/lam_bar**2
        if rho>1:
            return 1
        else:
            return rho

def EC3_force(domain, corners, t, fy, E, v):
    """Function to find the local buckling load using the effective width concept"""

    effective_area = 0

    # Create list of interval lengths between corners
    intervals = FEM_solver.open_intervals_list(domain, corners)

    # Outstand length first entry
    K = 0.43
    stress_cr = elastic_local_buckle_stress(K,v,E,t,intervals[0])
    lambda_bar = (fy/stress_cr)**0.5
    #print(outstand_rho(lambda_bar))
    effective_area += outstand_rho(lambda_bar)*intervals[0]*t

    # Internal lengths
    if len(corners) > 1:
        K = 4
        for i in range(len(intervals)-2):
            stress_cr = elastic_local_buckle_stress(K,v,E,t,intervals[i+1])
            lambda_bar = (fy/stress_cr)**0.5
            #print(internal_rho(lambda_bar))
            effective_area += internal_rho(lambda_bar)*intervals[i+1]*t

    # Outstand length last entry
    K = 0.43
    stress_cr = elastic_local_buckle_stress(K,v,E,t,intervals[-1])
    lambda_bar = (fy/stress_cr)**0.5
    #print(outstand_rho(lambda_bar))
    effective_area += outstand_rho(lambda_bar)*intervals[-1]*t

    return fy*effective_area

def cross_section_slenderness(fy,sigma_LB):
    return (fy/sigma_LB)**0.5


def load_vs_desired_y(a_list, node_pos, a0, U0_list, L, t, E, sigma_LB, target_y, normalise_PLB = False):
    """Produce plot of load (normalised by P_LB) as U0 varies - plot against max y"""

    if normalise_PLB:
        P_LB = local_buckle_force(np.array([node_pos[0], node_pos[-1]]), t, sigma_LB)
        y_title = "Force (normalsied by Sigma LB)"
    else:
        P_LB = 1000
        y_title = "Force (kN)"

    P_list = np.zeros(U0_list.size)
    y_list = np.zeros(U0_list.size)

    for i in range(len(U0_list)):
        P_list[i] = get_force(a_list[i], node_pos, a0, U0_list[i], L, t, E) / P_LB
        x, y = Plotters.extract_xy_data(a_list[i]+a0, node_pos)
        closest_index = np.argmin(np.abs(x - target_y))
        y_list[i] = abs(y[closest_index])

    Plotters.simple_plot(y_list*1000, P_list, x_label="y={}mm lateral displacement (mm)".format(target_y*1000), y_label=y_title, title="Evolution of Force with Lateral Displacement at y={}mm".format(target_y*1000))


def plot_stess_x(a_sol, node_pos, a0, U0, L, E, fy, domain, corners, v, t):
    """plot variation of stress x along the cross section"""

    # Get xy results
    x_data, y_data = Plotters.extract_xy_data(a_sol, node_pos)
    x_data, y0_data = Plotters.extract_xy_data(a0, node_pos)

    # Solve Stress
    sigma_list = np.zeros(x_data.size)
    plus_fy = np.zeros(x_data.size)
    minus_fy = np.zeros(x_data.size)
    for i in range(len(x_data)):
        sigma_list[i] = E/(2*L)*(y_data[i]**2*np.pi**2/(2*L) + np.pi**2/L*y_data[i]*y0_data[i] - 2*U0)
        plus_fy[i] = fy
        minus_fy[i] = -1*fy

    # Create list of interval lengths between corners
    intervals = FEM_solver.open_intervals_list(domain, corners)
    switch_points = np.zeros(2*len(corners))

    area = 0

    # Outstand length first entry
    K = 0.43
    stress_cr = elastic_local_buckle_stress(K,v,E,t,intervals[0])
    lambda_bar = (fy/stress_cr)**0.5
    switch_points[0] = corners[0] - outstand_rho(lambda_bar)*intervals[0]

    # Internal lengths
    if len(corners) > 1:
        K = 4
        for i in range(1, len(intervals)-1):
            stress_cr = elastic_local_buckle_stress(K,v,E,t,intervals[i])
            lambda_bar = (fy/stress_cr)**0.5
            switch_points[2*i-1] = corners[i-1] + internal_rho(lambda_bar)*intervals[i]/2
            switch_points[2*i] = corners[i] - internal_rho(lambda_bar)*intervals[i]/2

    # Outstand length last entry
    K = 0.43
    stress_cr = elastic_local_buckle_stress(K,v,E,t,intervals[-1])
    lambda_bar = (fy/stress_cr)**0.5
    switch_points[-1] = corners[-1] + outstand_rho(lambda_bar)*intervals[-1]

    EC3_x_coord = np.concatenate(([domain[0]], np.repeat(switch_points, 2), [domain[-1]]))
    EC3_y_coord = np.zeros(len(EC3_x_coord))  
    EC3_y_coord[2::4] = -1
    EC3_y_coord[3::4] = -1  

    #plt.plot(x_data*1000, plus_fy/fy*-1, color='#ff7f0e')
    #plt.plot(x_data*1000, minus_fy/fy*-1, color='#ff7f0e')
    plt.plot(x_data*1000, sigma_list/fy*-1, color='#1f77b4', label='Predicted Stress')
    plt.plot(EC3_x_coord*1000, EC3_y_coord*-1, color='#ff7f0e', label='EWM Idealised Stress', zorder=-1)
    
    plt.grid()
    plt.xlabel("y (mm)")
    plt.ylabel("Compressive longitudinal stress (normalised by fy)")
    plt.title("Compressive longitudinal membrane stress across section at failure")
    plt.legend()
    plt.ylim(-0.4, 1.2)
    plt.show()


def find_dsm_load(domain, t, fy, stress_LB):

    yield_f = yeild_force(domain, t, fy)

    lam = (fy/stress_LB)**0.5

    if lam <= 0.776:
        print("Not Locally buckle")
        return(yield_f)
    else:
        dsm_load = (1-0.15*(stress_LB/fy)**(0.4))*(stress_LB/fy)**(0.4)*yield_f
        return dsm_load
