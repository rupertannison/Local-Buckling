import Plotters

import numpy as np
import matplotlib.pyplot as plt

def differentiate(x_list, y_list):
    """Compute first differential of y_list with respect to x_list
    List is one value shorter!"""

    #Gradients between values in y_list
    y_prime_interval_list = np.diff(y_list)/np.diff(x_list)

    #Form x_position list
    x_prime_list = np.array([])
    for i in range(len(y_prime_interval_list)):
        average_x = (x_list[i]+x_list[i+1])/2
        x_prime_list = np.append(x_prime_list, average_x)

    return x_prime_list, y_prime_interval_list

def second_derivative(x_list, y_list):
    """Returns list of second derivate values - Note the list is 2 values shorter (lose first and last position)"""
    x_prime, y_prime = differentiate(x_list, y_list)
    x_double_p, y_double_p = differentiate(x_prime, y_prime)
    return (x_double_p, y_double_p)


def verification_RHS_LHS_plot(a_solution, node_pos, a0, U0, L, t, v, plot=True):
    """Produce plots of the LHS and RHS of the equation for verification
    Returns x positions and LHS and RHS data"""
    
    # Extract x and y data for solution and y0
    x_data, y_data = Plotters.extract_node_xy_data(a_solution, node_pos)
    x_data, y0_data = Plotters.extract_node_xy_data(a0, node_pos)

    # Compute derivatives of y_data
    x_second_d_list, y_second_d_list = second_derivative(x_data, y_data)
    x_fourth_d_list, y_fourth_d_list = second_derivative(x_second_d_list, y_second_d_list)

    # Cut off end values so all lists same length
    y_data_2 = np.delete(y_data, [0, 1, -2, -1])
    y0_data_2 = np.delete(y0_data, [0, 1, -2, -1])
    y_second_d_list_2 = np.delete(y_second_d_list, [0, -1])

    # Calc constants
    gamma = L*t**2/(6*(1-v**2))
    A = 2*(np.pi/L)**2
    B = (np.pi/L)**2*((np.pi/L)**2-2*U0/gamma)
    C = L/(2*gamma)*(np.pi/L)**4
    F = 2*U0/gamma*(np.pi/L)**2

    # Compute LHS 
    LHS_data = np.zeros(len(y_fourth_d_list))
    for i in range(len(LHS_data)):
        LHS_data[i] = y_fourth_d_list[i] - A*y_second_d_list_2[i] + B*y_data_2[i] + 2*C*y_data_2[i]*y0_data_2[i]**2
    
    # Compute RHS
    RHS_data = np.zeros(len(y_fourth_d_list))
    for i in range(len(RHS_data)):
        RHS_data[i] = -1*C*y_data_2[i]**3 - 3*C*y_data_2[i]**2*y0_data_2[i] + F*y0_data_2[i]

    # Plot results
    if plot:
        plt.plot(x_fourth_d_list*1000, LHS_data, label='LHS')
        plt.plot(x_fourth_d_list*1000, RHS_data, label='RHS')
        plt.title("LHS and RHS of governing differential equation")
        plt.xlabel("y (mm)")
        plt.ylabel("Governing Differential Equation Value")
        plt.legend(loc="upper center")
        plt.grid()
        plt.legend()
        plt.show()
    else:
        return(x_fourth_d_list, LHS_data, RHS_data)


