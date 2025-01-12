import Shape_Functions

import numpy as np
import matplotlib.pyplot as plt

def plot_shape_fns():
    """Plots shape functions"""
    x = np.linspace(0, 1, num=50)
    N1 = np.zeros(len(x))
    M1 = np.zeros(len(x))
    N2 = np.zeros(len(x))
    M2 = np.zeros(len(x))
    for i in range(len(x)):
        result = Shape_Functions.shape_fn_N(0, 1, x[i])
        N1[i] = result[0][0]
        M1[i] = result[0][1]
        N2[i] = result[0][2]
        M2[i] = result[0][3]
    plt.plot(x, N1, label='N1')
    plt.plot(x, M1, label='M1')
    plt.plot(x, N2, label='N2')
    plt.plot(x, M2, label='M2')
    plt.legend()
    plt.grid()
    plt.xlabel("Fraction across element")
    #plt.xlabel("Y(y) (mm)         Y0(y)+Y(Y) (mm)")
    plt.ylabel("Shape Function Value")
    plt.title("Elemental Shape Functions")
    plt.show()

def plot_global_shape_fns():
    """Plots shape functions"""
    x = np.linspace(0, 1, num=50)
    N1 = np.zeros(len(x))
    M1 = np.zeros(len(x))
    N2 = np.zeros(len(x))
    M2 = np.zeros(len(x))
    for i in range(len(x)):
        result = Shape_Functions.shape_fn_N(0, 1, x[i])
        N1[i] = result[0][0]
        M1[i] = result[0][1]
        N2[i] = result[0][2]
        M2[i] = result[0][3]
    fig = plt.figure()
    plt.plot(x, N1, label='Ni', color='#1f77b4')
    plt.plot(x, M1, label='Mi', color='#ff7f0e', zorder=-1)
    plt.plot(x-1, N2, color='#1f77b4')
    plt.plot(x-1, M2, color='#ff7f0e', zorder=-1)
    plt.plot([-1.5,-1], [0,0], color='#1f77b4')
    plt.plot([1,1.5], [0,0], color='#1f77b4')
    plt.legend()
    plt.grid()
    plt.xlabel("Fraction across element")
    plt.ylabel("Shape Function Value")
    plt.title("Global Shape Functions")
    plt.show()

def plot_shape_fns_B():
    """Plots shape functions"""
    x = np.linspace(0, 1, num=20)
    N1 = np.zeros(20)
    M1 = np.zeros(20)
    N2 = np.zeros(20)
    M2 = np.zeros(20)
    for i in range(20):
        result = Shape_Functions.shape_fn_B(0, 1, x[i])
        N1[i] = result[0][0]
        M1[i] = result[0][1]
        N2[i] = result[0][2]
        M2[i] = result[0][3]
    fig = plt.figure()
    plt.plot(x, N1, label='N1\'\'')
    plt.plot(x, M1, label='M1\'\'')
    plt.plot(x, N2, label='N2\'\'')
    plt.plot(x, M2, label='M2\'\'')
    plt.legend()
    plt.show()


def extract_xy_data(a_gl, node_pos, points_per_elem=20):
    """Returns the x and y data from the node positions and solution vector"""
    num_elem = node_pos.size - 1

    # Form list of element matricies
    a_elem = np.zeros((num_elem, 4, 1))
    for i in range(num_elem):
        a_elem[i] = np.array([[a_gl[2*i][0]], [a_gl[2*i+1][0]], [a_gl[2*i+2][0]], [a_gl[2*i+3][0]]])
    
    # Form the x and y lists, calculating each element in turn
    x = np.zeros((num_elem, points_per_elem))
    y = np.zeros((num_elem, points_per_elem))
    
    for i in range(num_elem):
        x[i] = np.linspace(node_pos[i], node_pos[i+1], num=points_per_elem, endpoint=False)
        for j in range(points_per_elem):
            y[i][j] = np.dot(Shape_Functions.shape_fn_N(node_pos[i], node_pos[i+1], x[i][j]), a_elem[i])[0][0]

    x1 = np.reshape(x, num_elem*points_per_elem)
    y1 = np.reshape(y, num_elem*points_per_elem)
    x2 = np.append(x1, np.array([node_pos[-1]]))
    final_y = np.dot(Shape_Functions.shape_fn_N(node_pos[-2], node_pos[-1], x2[-1]), a_elem[-1])[0][0]
    y2 = np.append(y1, final_y)

    return (x2, y2)

def extract_node_xy_data(a_gl, node_pos):
    """Returns the x and y data of node positions"""

    #y_data is every other node position
    y_data = np.zeros(len(node_pos))
    for i in range(len(y_data)):
        y_data[i] = a_gl[2*i][0]

    return (node_pos, y_data)

def plot_rel_solution(a_gl, node_pos, num_elem_points=10, imperfection=False):
    """Plot the solution found from FEM given the vector a"""

    x, y = extract_xy_data(a_gl, node_pos, points_per_elem=num_elem_points)

    # Plot x and y
    plt.plot(x*1000, y*1000)
    plt.grid()
    plt.xlabel("y (mm)")
    plt.ylabel("Y0 (mm)")
    if imperfection:
        plt.title("Imperfection profile across section for LC2 S12")
    else:
        plt.title("Deflected Shape")
    plt.show()

def plot_abs_solution(a_gl, node_pos, a0, num_elem_points=10):
    """Plot the solution found from FEM given the vector a"""
    
    # Extract absolute displacement from relative displacement
    a_gl_2 = a_gl+a0

    x, y = extract_xy_data(a_gl_2, node_pos, points_per_elem=num_elem_points)

    # Plot x and y
    plt.plot(x, y)
    plt.grid()
    plt.xlabel("y")
    plt.ylabel("Y")
    plt.title("Deflected Shape")
    plt.show()

def plot_BC1(a_gl, node_pos, v, L, D):
    """Plot the BC that should be zero at free ends, found from FEM given the vector a"""
    num_elem = node_pos.size - 1
    points_per_elem = 10
    
    # Form list of element matricies
    a_elem = np.zeros((num_elem, 4, 1))
    for i in range(num_elem):
        a_elem[i] = np.array([[a_gl[2*i][0]], [a_gl[2*i+1][0]], [a_gl[2*i+2][0]], [a_gl[2*i+3][0]]])
    
    # Form the x and y lists, calculating each element in turn
    x = np.zeros((num_elem, points_per_elem))
    y = np.zeros((num_elem, points_per_elem))
    
    for i in range(num_elem):
        x[i] = np.linspace(node_pos[i], node_pos[i+1], num=points_per_elem, endpoint=False)
        for j in range(points_per_elem):
            y[i][j] -= v*(np.pi/L)**2*np.dot(Shape_Functions.shape_fn_N(node_pos[i], node_pos[i+1], x[i][j]), a_elem[i])[0][0]
            y[i][j] += np.dot(Shape_Functions.shape_fn_B(node_pos[i], node_pos[i+1], x[i][j]), a_elem[i])[0][0]

    x1 = np.reshape(x, num_elem*points_per_elem)
    y1 = np.reshape(y, num_elem*points_per_elem)
    x2 = np.append(x1, np.array([node_pos[-1]]))
    final_y = -1*v*(np.pi/L)**2*np.dot(Shape_Functions.shape_fn_N(node_pos[-2], node_pos[-1], x2[-1]), a_elem[-1])[0][0]
    final_y += np.dot(Shape_Functions.shape_fn_B(node_pos[-2], node_pos[-1], x2[-1]), a_elem[-1])[0][0]
    y2 = np.append(y1, final_y)

    # Plot x and y
    plt.plot(x2*1000, y2*-1*D)
    plt.grid()
    plt.xlabel("y (mm)")
    plt.ylabel("Moment, My (Nm)")
    plt.title("Moment, My, across the section")
    plt.show()

def plot_BC2(a_gl, node_pos, v, L, D):
    """Plot the BC that should be zero at free ends, found from FEM given the vector a"""
    num_elem = node_pos.size - 1
    points_per_elem = 10
    
    # Form list of element matricies
    a_elem = np.zeros((num_elem, 4, 1))
    for i in range(num_elem):
        a_elem[i] = np.array([[a_gl[2*i][0]], [a_gl[2*i+1][0]], [a_gl[2*i+2][0]], [a_gl[2*i+3][0]]])
    
    # Form the x and y lists, calculating each element in turn
    x = np.zeros((num_elem, points_per_elem))
    y = np.zeros((num_elem, points_per_elem))
    
    for i in range(num_elem):
        x[i] = np.linspace(node_pos[i], node_pos[i+1], num=points_per_elem, endpoint=False)
        for j in range(points_per_elem):
            y[i][j] -= (2-v)*(np.pi/L)**2*np.dot(Shape_Functions.shape_fn_N_derivative(node_pos[i], node_pos[i+1], x[i][j]), a_elem[i])[0][0]
            y[i][j] += np.dot(Shape_Functions.shape_fn_B_derivative(node_pos[i], node_pos[i+1], x[i][j]), a_elem[i])[0][0]

    x1 = np.reshape(x, num_elem*points_per_elem)
    y1 = np.reshape(y, num_elem*points_per_elem)
    x2 = np.append(x1, np.array([node_pos[-1]]))
    final_y = -1*(2-v)*(np.pi/L)**2*np.dot(Shape_Functions.shape_fn_N_derivative(node_pos[-2], node_pos[-1], x2[-1]), a_elem[-1])[0][0] 
    final_y += np.dot(Shape_Functions.shape_fn_B_derivative(node_pos[-2], node_pos[-1], x2[-1]), a_elem[-1])[0][0]
    y2 = np.append(y1, final_y)

    # Plot x and y
    plt.plot(x2*1000, y2*-1*D/1000)
    plt.grid()
    plt.ylabel("Shear force, Vy (kN)")
    plt.xlabel("y (mm)")
    plt.title("Shear force, Vy, across the section")
    plt.show()


def plot_multiple_solution(a_list, node_pos, a0, legend_list, points_per_elem = 10, title="", absolute=True):
    """Plot the solution found from FEM given a list of a vectors"""
    num_elem = node_pos.size - 1
    
    # Get absolute displacement from relative displacement
    a_abs_list = np.zeros(a_list.shape)
    for i in range(len(a_list)):
        if absolute:
            a_abs_list[i] = a_list[i]+a0
        else:
            a_abs_list[i] = a_list[i]

    # Form the x list, calculating each element in turn
    x = np.zeros((num_elem, points_per_elem))
    for i in range(num_elem):
        x[i] = np.linspace(node_pos[i], node_pos[i+1], num=points_per_elem, endpoint=False)
    x1 = np.reshape(x, num_elem*points_per_elem)
    x2 = np.append(x1, np.array([node_pos[-1]]))

    # Form list of y lists and plot them
    y_list = np.zeros((len(a_list), len(x2)))
    for k in range(len(y_list)):

        # Form list of element vectors
        a_elem = np.zeros((num_elem, 4, 1))
        for i in range(num_elem):
            a_elem[i] = np.array([[a_abs_list[k][2*i][0]], [a_abs_list[k][2*i+1][0]], [a_abs_list[k][2*i+2][0]], [a_abs_list[k][2*i+3][0]]])

        y_k = np.zeros((num_elem, points_per_elem))
        for i in range(num_elem):
            for j in range(points_per_elem):
                y_k[i][j] = np.dot(Shape_Functions.shape_fn_N(node_pos[i], node_pos[i+1], x[i][j]), a_elem[i])[0][0]

        y1_k = np.reshape(y_k, num_elem*points_per_elem)
        
        final_y_k = np.dot(Shape_Functions.shape_fn_N(node_pos[-2], node_pos[-1], x2[-1]), a_elem[-1])[0][0]
        y2_k = np.append(y1_k, final_y_k)

        y_list[k] = y2_k
        plt.plot(x2*1000, y_list[k]*1000, label=legend_list[k])

    # Plot x and y
    plt.legend(loc='lower center')
    plt.grid()
    plt.xlabel("y (mm)")
    plt.ylabel("Y (mm)")
    plt.title(title)
    plt.show()

def simple_plot(xlist, ylist, x_label="", y_label="", title=""):
    plt.plot(xlist, ylist)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
