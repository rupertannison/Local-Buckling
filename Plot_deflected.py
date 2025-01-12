import Plotters

import numpy as np
import matplotlib.pyplot as plt

def plot_deflected_shape(shape, a_sol, node_pos, a0, domain, corners, amplification=1):
    """Plot deflected shape of solution, depend on shape of section"""
    if shape == "C":
        plot_deflected_shape_C(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "LC":
        plot_deflected_shape_lipped_C(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "Z":
        plot_deflected_shape_Z(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "LZ":
        plot_deflected_shape_lipped_Z(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "LA":
        plot_deflected_shape_lipped_Right_Angle(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "O":
        plot_deflected_shape_Omega(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "LO":
        plot_deflected_shape_lipped_Omega(a_sol, node_pos, a0, domain, corners, amplification)
    elif shape == "A":
        plot_deflected_shape_Angle(a_sol, node_pos, a0, domain, corners, amplification)
    else:
        print("Shape not known")


def plot_deflected_shape_C(a_sol, node_pos, a0, domain, corners, amplification=1, plot=True, points_per_el=20):
    """Plot the deflected shape of a C section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 2
    except:
        raise NameError('corner list for C section must have 2 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, -1*intervals[0], -1*intervals[0], -1*intervals[0]+intervals[2]])
    original_y = np.array([0, 0, -1*intervals[1], -1*intervals[1]])

    def calc_xy_locations(a0_true):

        # Create list of interval lengths between corners
        intervals = np.zeros(len(corners)+1)
        intervals[0] = corners[0]-domain[0]
        for i in range(len(corners)-1):
            intervals[i+1] = corners[i+1]-corners[i]
        intervals[-1] = domain[-1]-corners[-1]

        # Original undeformed shape
        original_x = np.array([0, -1*intervals[0], -1*intervals[0], -1*intervals[0]+intervals[2]])
        original_y = np.array([0, 0, -1*intervals[1], -1*intervals[1]])

        # List of number of elements between corners
        num_el_list = np.zeros(len(corners)+1)
        for i in range(len(num_el_list)):
            num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)

        # Extract xy data
        if a0_true:
            distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
        else:
            distances, deflections = Plotters.extract_xy_data(a0*2, node_pos, points_per_elem=points_per_el)
        deflections *= amplification

        x_plot = np.zeros(distances.size)
        y_plot = np.zeros(distances.size)

        # First section - top flange
        for i in range(0, int(num_el_list[0])*points_per_el+1):
            x_plot[i] = -1*distances[i]
            y_plot[i] = deflections[i]
            
        # Second section - web
        corner_i = int(num_el_list[0])*points_per_el+1
        corner_x = -1*intervals[0]
        corner_y = 0
        for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
            x_plot[i] = corner_x - deflections[i]
            y_plot[i] = corner_y - (distances[i]-corners[0])

        # Third section - bottom flange
        corner_i = corner_i + int(num_el_list[1])*points_per_el
        corner_x = -1*intervals[0]
        corner_y = -1*intervals[1]
        for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
            x_plot[i] = corner_x + (distances[i]-corners[1])
            y_plot[i] = corner_y - deflections[i]

        return x_plot, y_plot
    
    x_imp, y_imp = calc_xy_locations(False)
    x_disp, y_disp = calc_xy_locations(True)

    if plot:
        plt.plot(original_x, original_y, label='Nominal Location')
        plt.plot(x_imp, y_imp, label='Imperfection Shape')
        plt.plot(x_disp, y_disp, label='Deflected Shape')
        plt.axis('equal')
        plt.legend()
        plt.show()

    return(original_x, original_y, x_disp, y_disp)

def plot_deflected_shape_lipped_C(a_sol, node_pos, a0, domain, corners, amplification=1, plot=True, points_per_el=20):
    """Plot the deflected shape of a lipped C section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 4
    except:
        raise NameError('corner list for lipped C section must have 4 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, 0, -1*intervals[1], -1*intervals[1], -1*intervals[1]+intervals[3], -1*intervals[1]+intervals[3]])
    original_y = np.array([0, intervals[0], intervals[0], intervals[0]-intervals[2], intervals[0]-intervals[2], intervals[0]-intervals[2]+intervals[4]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - top lip
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = -1*deflections[i]
        y_plot[i] = distances[i]
        
    # Second section - top flange
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = 0
    corner_y = intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x - (distances[i]-corners[0])
        y_plot[i] = corner_y - deflections[i]

    # Third section - web
    corner_i = corner_i + int(num_el_list[1])*points_per_el
    corner_x = -1*intervals[1]
    corner_y = intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
        x_plot[i] = corner_x + deflections[i]
        y_plot[i] = corner_y - (distances[i]-corners[1])
    
    # Fourth section - bottom flange
    corner_i = corner_i + int(num_el_list[2])*points_per_el
    corner_x = -1*intervals[1]
    corner_y = intervals[0]-intervals[2]
    for i in range(corner_i, corner_i + int(num_el_list[3])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[2])
        y_plot[i] = corner_y + deflections[i]
    
    # Fith section - bottom lip
    corner_i = corner_i + int(num_el_list[3])*points_per_el
    corner_x = -1*intervals[1]+intervals[3]
    corner_y = intervals[0]-intervals[2]
    for i in range(corner_i, corner_i + int(num_el_list[4])*points_per_el):
        x_plot[i] = corner_x - deflections[i]
        y_plot[i] = corner_y + (distances[i]-corners[3])

    if plot:
        plt.plot(original_x, original_y)
        plt.plot(x_plot, y_plot)
        plt.axis('equal')
        plt.show()

    return(original_x, original_y, x_plot, y_plot)

def plot_deflected_shape_Z(a_sol, node_pos, a0, domain, corners, amplification=1, points_per_el=20):
    """Plot the deflected shape of a Z section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 2
    except:
        raise NameError('corner list for z section must have 2 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, intervals[0], intervals[0], intervals[0]+intervals[2]])
    original_y = np.array([0, 0, -1*intervals[1], -1*intervals[1]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - top flange
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = distances[i]
        y_plot[i] = -1*deflections[i]
        
    # Second section - web
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = intervals[0]
    corner_y = 0
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x - deflections[i]
        y_plot[i] = corner_y - (distances[i]-corners[0])

    # Third section - bottom flange
    corner_i = corner_i + int(num_el_list[1])*points_per_el
    corner_x = intervals[0]
    corner_y = -1*intervals[1]
    for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[1])
        y_plot[i] = corner_y - deflections[i]

    plt.plot(original_x, original_y)
    plt.plot(x_plot, y_plot)
    plt.axis('equal')
    plt.show()

def plot_deflected_shape_lipped_Z(a_sol, node_pos, a0, domain, corners, amplification=1, plot=True, points_per_el=20):
    """Plot the deflected shape of a lipped Z section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 4
    except:
        raise NameError('corner list for lipped Z section must have 4 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, 0, intervals[1], intervals[1], intervals[1]+intervals[3], intervals[1]+intervals[3]])
    original_y = np.array([0, intervals[0], intervals[0], intervals[0]-intervals[2], intervals[0]-intervals[2], intervals[0]-intervals[2]+intervals[4]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - top lip
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = -1*deflections[i]
        y_plot[i] = distances[i]
        
    # Second section - top flange
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = 0
    corner_y = intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[0])
        y_plot[i] = corner_y + deflections[i]

    # Third section - web
    corner_i = corner_i + int(num_el_list[1])*points_per_el
    corner_x = intervals[1]
    corner_y = intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
        x_plot[i] = corner_x + deflections[i]
        y_plot[i] = corner_y - (distances[i]-corners[1])
    
    # Fourth section - bottom flange
    corner_i = corner_i + int(num_el_list[2])*points_per_el
    corner_x = intervals[1]
    corner_y = intervals[0]-intervals[2]
    for i in range(corner_i, corner_i + int(num_el_list[3])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[2])
        y_plot[i] = corner_y + deflections[i]
    
    # Fith section - bottom lip
    corner_i = corner_i + int(num_el_list[3])*points_per_el
    corner_x = intervals[1]+intervals[3]
    corner_y = intervals[0]-intervals[2]
    for i in range(corner_i, corner_i + int(num_el_list[4])*points_per_el):
        x_plot[i] = corner_x + deflections[i]
        y_plot[i] = corner_y + (distances[i]-corners[3])

    if plot:
        plt.plot(original_x, original_y)
        plt.plot(x_plot, y_plot)
        plt.axis('equal')
        plt.show()

def plot_deflected_shape_lipped_Right_Angle(a_sol, node_pos, a0, domain, corners, amplification=1, points_per_el=20):
    """Plot the deflected shape of a lipped right angle section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 3
    except:
        raise NameError('corner list for lipped right angled section must have 3 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, -1*intervals[0], -1*intervals[0], -1*intervals[0]+intervals[2], -1*intervals[0]+intervals[2]])
    original_y = np.array([0, 0, -1*intervals[1], -1*intervals[1], -1*intervals[1]+intervals[3]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - top flange
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = -1*distances[i]
        y_plot[i] = deflections[i]
        
    # Second section - web
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = -1*intervals[0]
    corner_y = 0
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x - deflections[i]
        y_plot[i] = corner_y - (distances[i]-corners[0])

    # Third section - bottom flange
    corner_i = corner_i + int(num_el_list[1])*points_per_el
    corner_x = -1*intervals[0]
    corner_y = -1*intervals[1]
    for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[1])
        y_plot[i] = corner_y - deflections[i]
    
    # Fourth section - bottom lip
    corner_i = corner_i + int(num_el_list[2])*points_per_el
    corner_x = -1*intervals[0]+intervals[2]
    corner_y = -1*intervals[1]
    for i in range(corner_i, corner_i + int(num_el_list[3])*points_per_el):
        x_plot[i] = corner_x + deflections[i]
        y_plot[i] = corner_y + (distances[i]-corners[2])

    plt.plot(original_x, original_y)
    plt.plot(x_plot, y_plot)
    plt.axis('equal')
    plt.show()

def plot_deflected_shape_Omega(a_sol, node_pos, a0, domain, corners, amplification=1, plot=True, points_per_el=20):
    """Plot the deflected shape of a Omega section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 4
    except:
        raise NameError('corner list for Omega section must have 4 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, intervals[0], intervals[0], intervals[0]+intervals[2], intervals[0]+intervals[2], intervals[0]+intervals[2]+intervals[4]])
    original_y = np.array([0, 0, intervals[1], intervals[1], intervals[1]-intervals[3], intervals[1]-intervals[3]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - bottom left flange
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = distances[i]
        y_plot[i] = deflections[i]
        
    # Second section - left web
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = intervals[0]
    corner_y = 0
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x - deflections[i] 
        y_plot[i] = corner_y + (distances[i]-corners[0])

    # Third section - top flange
    corner_i = corner_i + int(num_el_list[1])*points_per_el
    corner_x = intervals[0]
    corner_y = intervals[1]
    for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[1])
        y_plot[i] = corner_y + deflections[i]
    
    # Fourth section - right web
    corner_i = corner_i + int(num_el_list[2])*points_per_el
    corner_x = intervals[0]+intervals[2]
    corner_y = intervals[1]
    for i in range(corner_i, corner_i + int(num_el_list[3])*points_per_el):
        x_plot[i] = corner_x + deflections[i]
        y_plot[i] = corner_y - (distances[i]-corners[2])
    
    # Fith section - bottom right flange
    corner_i = corner_i + int(num_el_list[3])*points_per_el
    corner_x = intervals[0]+intervals[2]
    corner_y = intervals[1]-intervals[3]
    for i in range(corner_i, corner_i + int(num_el_list[4])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[3])
        y_plot[i] = corner_y + deflections[i]

    if plot:
        plt.plot(original_x, original_y)
        plt.plot(x_plot, y_plot)
        plt.axis('equal')
        plt.show()


def plot_deflected_shape_lipped_Omega(a_sol, node_pos, a0, domain, corners, amplification=1, plot=True, points_per_el=20):
    """Plot the deflected shape of a lipped Omega section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 6
    except:
        raise NameError('corner list for lipped Omega section must have 6 entries')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, 0, intervals[1], intervals[1], intervals[1]+intervals[3], intervals[1]+intervals[3], intervals[1]+intervals[3]+intervals[5], intervals[1]+intervals[3]+intervals[5]])
    original_y = np.array([0, -1*intervals[0], -1*intervals[0], -1*intervals[0]+intervals[2], -1*intervals[0]+intervals[2], -1*intervals[0]+intervals[2]-intervals[4], -1*intervals[0]+intervals[2]-intervals[4], -1*intervals[0]+intervals[2]-intervals[4]+intervals[6]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - bottom left lip
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = deflections[i]
        y_plot[i] = -1*distances[i]
    
    # Second section - bottom left flange
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = 0
    corner_y = -1*intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[0])
        y_plot[i] = corner_y + deflections[i] 

    # Third section - left web
    corner_i = corner_i + int(num_el_list[1])*points_per_el
    corner_x = intervals[1]
    corner_y = -1*intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[2])*points_per_el):
        x_plot[i] = corner_x - deflections[i] 
        y_plot[i] = corner_y + (distances[i]-corners[1])

    # Fourth section - top flange
    corner_i = corner_i + int(num_el_list[2])*points_per_el
    corner_x = intervals[1]
    corner_y = -1*intervals[0]+intervals[2]
    for i in range(corner_i, corner_i + int(num_el_list[3])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[2])
        y_plot[i] = corner_y + deflections[i]
    
    # Fith section - right web
    corner_i = corner_i + int(num_el_list[3])*points_per_el
    corner_x = intervals[1]+intervals[3]
    corner_y = -1*intervals[0]+intervals[2]
    for i in range(corner_i, corner_i + int(num_el_list[4])*points_per_el):
        x_plot[i] = corner_x + deflections[i]
        y_plot[i] = corner_y - (distances[i]-corners[3])
    
    # Sixth section - bottom right flange
    corner_i = corner_i + int(num_el_list[4])*points_per_el
    corner_x = intervals[1]+intervals[3]
    corner_y = -1*intervals[0]+intervals[2]-intervals[4]
    for i in range(corner_i, corner_i + int(num_el_list[5])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[4])
        y_plot[i] = corner_y + deflections[i]
    
    # Seventh section - bottom right lip
    corner_i = corner_i + int(num_el_list[5])*points_per_el
    corner_x = intervals[1]+intervals[3]+intervals[5]
    corner_y = -1*intervals[0]+intervals[2]-intervals[4]
    for i in range(corner_i, corner_i + int(num_el_list[6])*points_per_el):
        x_plot[i] = corner_x - deflections[i]
        y_plot[i] = corner_y + (distances[i]-corners[5])

    if plot:
        plt.plot(original_x, original_y)
        plt.plot(x_plot, y_plot)
        plt.axis('equal')
        plt.show()


def plot_deflected_shape_Angle(a_sol, node_pos, a0, domain, corners, amplification=1, plot=True, points_per_el=20):
    """Plot the deflected shape of a angle section"""
    num_el = node_pos.size - 1

    try:
        assert len(corners) == 1
    except:
        raise NameError('corner list for lipped Angle section must have 1 entry')

    # Create list of interval lengths between corners
    intervals = np.zeros(len(corners)+1)
    intervals[0] = corners[0]-domain[0]
    for i in range(len(corners)-1):
        intervals[i+1] = corners[i+1]-corners[i]
    intervals[-1] = domain[-1]-corners[-1]

    # Original undeformed shape
    original_x = np.array([0, 0, intervals[1]])
    original_y = np.array([0, -1*intervals[0], -1*intervals[0]])

    # List of number of elements between corners
    num_el_list = np.zeros(len(corners)+1)
    for i in range(len(num_el_list)):
        num_el_list[i] = round(intervals[i]/(domain[1]-domain[0])*num_el)
    
    # Extract xy data
    distances, deflections = Plotters.extract_xy_data(a_sol+a0, node_pos, points_per_elem=points_per_el)
    deflections *= amplification

    x_plot = np.zeros(distances.size)
    y_plot = np.zeros(distances.size)

    # First section - web
    for i in range(0, int(num_el_list[0])*points_per_el+1):
        x_plot[i] = deflections[i]
        y_plot[i] = -1*distances[i]
    
    # Second section - flange
    corner_i = int(num_el_list[0])*points_per_el+1
    corner_x = 0
    corner_y = -1*intervals[0]
    for i in range(corner_i, corner_i + int(num_el_list[1])*points_per_el):
        x_plot[i] = corner_x + (distances[i]-corners[0])
        y_plot[i] = corner_y + deflections[i] 

    if plot:
        plt.plot(original_x, original_y)
        plt.plot(x_plot, y_plot)
        plt.axis('equal')
        plt.show()
