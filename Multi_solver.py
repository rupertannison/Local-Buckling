import FEM_solver
import Plotters
import Verify_solution
import Force_finder

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def find_min_L(domain, corners, a0, b, U0, L_0, t, v, E, fy, tol=1e-8):

    def solve_and_get_force(L_guess):
        U0_i = Force_finder.should_fail_U0(E, fy, L_guess)
        a_sol, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, b, U0_i, L_guess, t, v, tol)
        P_i = Force_finder.get_force(a_sol, node_pos, a0, U0_i, L_guess, t, E)
        print("L = {} mm gives P = {} kN".format(L_guess*1000, P_i/1000))
        return P_i
    
    res = minimize(solve_and_get_force, L_0, bounds=[(L_0*0.1, L_0*3)], tol=1e-4)

    min_L = res.x
    if min_L in (L_0*0.4, L_0*1.6):
        print("Solution for Min L on boundary of L allowed")

    return min_L, res.fun


def solve_for_range_U0_sequential(domain, corners, a0, b0, U0_list, L, t, v, tolerence=1e-10, plot=False, factor=1):
    """Solve and plot for range of U0 values, using previous solution as guess for next"""

    a_list = np.zeros((int(len(U0_list)), a0.size, 1))

    #Run first iteration
    if U0_list[0] == 0:
        a_0, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, 0*b0, U0_list[0], L, t, v, tol=tolerence)
    else:
        a_0, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, b0, U0_list[0], L, t, v, tol=tolerence)
    a_list[0] = a_0

    #Run second iteration
    a_1, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, b0, U0_list[1], L, t, v, tol=tolerence)
    a_list[1] = a_1

    for i in range(2, len(U0_list)):
        a_i, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, factor*a_list[i-1], U0_list[i], L, t, v, tol=tolerence)
        a_list[i] = a_i

    if plot:
        Plotters.plot_multiple_solution(a_list, node_pos, a0, U0_list/L*100, title="Deflected shape as U0 gradually increased (U0 given as '%' of L)")

    # Display validity of solutions
    """for i in range(len(U0_list)):
        Verify_solution.verification_RHS_LHS_plot(a_list[i], node_pos, a0, U0_list[i], L, t, v)"""
    
    return (a_list, node_pos)


def plot_for_range_num_el(domain, corners, imperfections, num_el_list, U0, L, t, v, E, fy, validity=False, force=False, normalise_py=False):
    """Solve and plot for range of number of elements"""

    if normalise_py:
        P_y = Force_finder.yeild_force(np.array([domain[0], domain[-1]]), t, fy)
        y_label = "Predicted Failure Load (normalised by fy)"
    else:
        P_y = 1000
        y_label = "Force (kN)"

    x_data = []
    LHS_data = []
    RHS_data = []
    P_list = np.zeros(num_el_list.size) 

    for i in range(len(num_el_list)):
        a0_i, node_pos_i, num_el_list[i] = FEM_solver.convert_to_FEM(domain, corners, imperfections, num_el_list[i])
        b0_i = 100*a0_i
        a_sol_i, node_pos_i = FEM_solver.iterate_for_Y(domain, corners, a0_i, b0_i, U0, L, t, v, tol=1e-7)
        
        x_veri_i, LHS_i, RHS_i = Verify_solution.verification_RHS_LHS_plot(a_sol_i, node_pos_i, a0_i, U0, L, t, v, plot=False)
        x_data.append(x_veri_i)
        LHS_data.append(LHS_i)
        RHS_data.append(RHS_i)
        
        x_i, y_i = Plotters.extract_xy_data(a_sol_i+a0_i, node_pos_i)
        plt.plot(x_i*1000, y_i*1000, label = str(num_el_list[i]))

        P_list[i] = Force_finder.get_force(a_sol_i, node_pos_i, a0_i, U0, L, t, E)/P_y
    
    if force:
        for i in range(len(num_el_list)):
            print("{} elements gives a failure force of: {}kN".format(num_el_list[i], round(P_list[i]*P_y/1000,3)))
        
    plt.grid()
    plt.legend()
    plt.xlabel("y (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Deflected shape at failure as number of elements varies")
    plt.show()

    if validity:
        for i in range(len(num_el_list)):
            plt.plot(x_data[i], LHS_data[i], label = str(num_el_list[i])+' LHS')
            plt.plot(x_data[i], RHS_data[i], label = str(num_el_list[i])+' RHS')
            
        plt.grid()
        plt.legend()
        plt.show()
    
    if force:
        # create logarithmic x-axis
        fig, ax = plt.subplots()
        ax.set_xscale('log')

        # plot x,y data
        ax.plot(num_el_list, P_list, marker="o")

        # set labels
        ax.set_xlabel("Number of Elements")
        ax.set_ylabel(y_label)
        ax.set_title("Variation in predicted failure with number of elements")
        ax.grid()

        # show plot
        plt.show()
     

def solve_for_range_imp_factor_FixedL(domain, corners, a0_unfactor, imp_factor_list, b, U0, L, t , v, b_max, plot_disp=True):
    """Solve and plot for range of a0 imperfections, a0_unfactored is should be less than 1, and imp_factor is b/? factor needed"""

    a_list = np.zeros((int(len(imp_factor_list)), a0_unfactor.size, 1))
    
    for i in range(len(imp_factor_list)):
        a_i, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0_unfactor*b_max/imp_factor_list[i], b, U0, L, t, v)
        a_list[i] = a_i
        
        if plot_disp:
            a_i_2 = a_i+a0_unfactor*b_max/imp_factor_list[i]
            x_i, y_i = Plotters.extract_xy_data(a_i_2, node_pos)
            plt.plot(x_i, y_i, label = 'b/'+str(np.round(imp_factor_list[i], 0)))

    if plot_disp:    
        plt.grid()
        plt.legend()
        plt.xlabel("y")
        plt.ylabel("Y")
        plt.title("Deflected shape as imperfetion varies - Fixed L")
        plt.show()

    return a_list, node_pos


def solve_for_range_imp_factor_VariableL(domain, corners, a0_unfactor, imp_factor_list, b, U0, L_0, t , v, b_max, E, fy, normalise_py=True, plot_disp=False, plot_force=True, plot_L=True):
    """Solve and plot for range of a0 imperfections, a0_unfactored is should be less than 1, and imp_factor is b/? factor needed"""

    a_list = np.zeros((int(len(imp_factor_list)), a0_unfactor.size, 1))
    
    if normalise_py:
        P_y = Force_finder.yeild_force(np.array([domain[0], domain[-1]]), t, fy)
        y_label = "Ultimate Failure Load (normalised by fy)"
    else:
        P_y = 1000
        y_label = "Force (kN)"

    P_list = np.zeros(imp_factor_list.size)
    L_list = np.zeros(imp_factor_list.size)

    for i in range(len(imp_factor_list)):
        L_i, P_i = find_min_L(domain, corners, a0_unfactor*b_max/imp_factor_list[i], b, U0, L_0, t, v, E, fy)
        U0_i = Force_finder.should_fail_U0(E, fy, L_i)
        a_i, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0_unfactor*b_max/imp_factor_list[i], b, U0_i, L_i, t, v)
        a_list[i] = a_i
        P_list[i] = P_i/P_y
        L_list[i] = L_i
 
        if plot_disp:
            a_i_2 = a_i+a0_unfactor*b_max/imp_factor_list[i]
            x_i, y_i = Plotters.extract_xy_data(a_i_2, node_pos)
            plt.plot(x_i*1000, y_i*1000, label = 'b/'+str(int(imp_factor_list[i])))
    
    print(imp_factor_list)
    print(L_list)

    if plot_disp:    
        plt.grid()
        plt.legend()
        plt.xlabel("y (mm)")
        plt.ylabel("Y (mm)")
        plt.title("Deflected shape at failure as initial imperfetion amplitude varies - Variable L")
        plt.show()

    if plot_force:
        plt.plot(1/imp_factor_list*100, P_list, marker="o")
        plt.ylim(bottom=0)
        plt.ylim(top=max(P_list)*1.2)
        plt.grid()
        plt.xlabel("Imperfection size (%' of b)")
        plt.ylabel(y_label)
        plt.title("Variation of failure load with initial imperfetion amplitude - Variable L")
        plt.show()
        #np.savetxt('S12_LC2_load_imp_vary_L.txt', np.column_stack((1/imp_factor_list*100, P_list)), header='Imperfection size ( of b),Ultimate Failure Load (normalised by fy)', delimiter=',')
    
    if plot_L:
        plt.plot(1/imp_factor_list*100, L_list*1000, marker="o")
        plt.ylim(bottom=0)
        plt.ylim(top=max(L_list)*1.2*1000)
        plt.grid()
        plt.xlabel("Imperfection size (%' of b)")
        plt.ylabel("Longitudinal half wavelength, L (mm)")
        plt.title("Variation of failure longitudinal half wavelength with initial imperfetion amplitude")
        plt.show()
        #np.savetxt('S12_LC2_L_imp_vary_L.txt', np.column_stack((1/imp_factor_list*100, L_list*1000)), header='Imperfection size ( of b),Longitudinal half wavelength, L (mm)', delimiter=',')
    