import Section_data
import Multi_solver
import FEM_solver
import Plotters
import Verify_solution
import Force_finder
import Plot_deflected

import numpy as np

""" Change the variables below to match your section properties and choose what functions to run by toggling the True/False if statements. 
    Use Section_data.py to add details of the section shape. """

# Material properties
v = 0.3
E = 210e9 # Input value
fy = 240e6 # Input value

# Imperfection factor - max imperfection amplitude is b/imp_factor
imp_factor = 200

# Nominal number of elements to divide the secion into
nominal_num_el = 50

# Import cross section from Section_data.py
L, t, num_el, domain, corners, imperfections, a0, node_pos, stress_LB, shape = Section_data.Example_Lipped_C_Section(imp_factor, nominal_num_el) # Set up section shape in Section_data.py

# Critical U0 for failure
U0 = Force_finder.should_fail_U0(E, fy, L)

# Initial guess solution
b_factor = 1000
b = b_factor*a0


"""Single Solver"""
if True:

    # If L is unknown then iterate for it, when re-running calcs on same section update L in Section_data.py so not need to solve again
    if True:
        # Iterate for L that gives minimum failure force for section
        L_min, P_min = Multi_solver.find_min_L(domain, corners, a0, b, U0, L, t, v, E, fy, tol=1e-8)
        print("Minimum L for seciton is {} mm, giving a force of {} kN".format(np.round(L_min*1000, 1), np.round(P_min/1000, 2)))

        # Update U0 once correct L has been found
        U0_min = Force_finder.should_fail_U0(E, fy, L_min)
        a_sol, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, b, U0_min, L_min, t, v, tol=1e-8)
        L, U0 = L_min, U0_min

    else:
        # If L for min force has already been calcuated, use the value from Section_data.py
        a_sol, node_pos = FEM_solver.iterate_for_Y(domain, corners, a0, b, U0, L, t, v, tol=1e-8)
        print("Failure force of: {} kN".format(str(round(Force_finder.get_force(a_sol, node_pos, a0, U0, L, t, E)/1000,2))))

    print("Cross sectional slenderness: {}".format(str(round(Force_finder.cross_section_slenderness(fy,stress_LB), 2))))
    print("Yield force of: {} kN".format(str(round(Force_finder.yeild_force(domain, t, fy)/1000,2))))
    print("EC3 effective width force of: {} kN".format(str(round(Force_finder.EC3_force(domain, corners, t, fy, E, v)/1000,2))))
    print("DSM force of: {} kN".format(str(round(Force_finder.find_dsm_load(domain, t, fy, stress_LB)/1000,2))))

    # Plot the imperfection shape and deflection shape against the y-axis
    Plotters.plot_multiple_solution(np.array([a0, a_sol+a0]), node_pos, a0, ["Imperfection","Failure Deflection"], absolute=False, title="Plot of failure displacement and imperfection profiles")
    
    # Draw the nominal section shape and the deflected shape - with deflection amplificaiton possible
    Plot_deflected.plot_deflected_shape(shape, a_sol, node_pos, a0, domain, corners, amplification=1)

    # Plot stress across the section
    Force_finder.plot_stess_x(a_sol, node_pos, a0, U0, L, E, fy, domain, corners, v, t)

     # Verify the boundary conditions are met at the free edges and the solution satifies the original equation
    if False:
        D = E*t**3/(12*(1-v**2))
        Plotters.plot_BC1(a_sol, node_pos, v, L, D)
        Plotters.plot_BC2(a_sol, node_pos, v, L, D)
        Verify_solution.verification_RHS_LHS_plot(a_sol, node_pos, a0, U0, L, t, v)


"""Multi Solve - run Single Solver to solve for correct L (and update Section_data.py) before running mult-solve functions"""

# Compare failure forece with number of modelling elements
if False:
    num_el_list = np.array([1, 1.5, 2, 4, 7.5, 10, 25, 50, 100])*10
    Multi_solver.plot_for_range_num_el(domain, corners, imperfections, num_el_list, U0, L, t, v, E, fy, validity=False, force=True, normalise_py=False)

# Find load and deflection evolution throughout shortening
if False:
    U0_list = np.linspace(0, U0, 10, True)
    factor_on_ai = 1 #Factor applied to previous solution when using as the 'guess' to find the next solution
    a_sol_list, node_pos = Multi_solver.solve_for_range_U0_sequential(domain, corners, a0, b, U0_list, L, t, v, plot=True, factor=factor_on_ai)
    Force_finder.load_vs_U0(a_sol_list, node_pos, a0, U0_list, L, t, E, fy, stress_LB, normalise_PLB=False)
    Force_finder.load_vs_max_y(a_sol_list, node_pos, a0, U0_list, L, t, E, fy, stress_LB, normalise_PLB=False)

    # Plot evolution of defleciton at given y position
    Force_finder.load_vs_desired_y(a_sol_list, node_pos, a0, U0_list, L, t, E, stress_LB, target_y=250/1000, normalise_PLB=False)

# Plot variation of failure force vs initial imperfection amplitude, for both for variable L and fixed L assumptions
if False:
    section_max_b = np.max(np.abs(np.diff(corners)))
    imp_factor_list = np.array([1e8, 500, 250, 125])

    # Fixed L imperfection sensitivity - use the L in Section_data.py for all imperfection amplitudes
    a_sol_list, node_pos = Multi_solver.solve_for_range_imp_factor_FixedL(domain, corners, a0/(section_max_b/imp_factor), imp_factor_list, b, U0, L, t, v, section_max_b, plot_disp=False)
    Force_finder.load_vs_imp(a_sol_list, node_pos, a0/(section_max_b/imp_factor), imp_factor_list, U0, L, t, E, fy, section_max_b)
    
    # Variable L imperfection sensitivity - re-solve L for each imperfection amplitude
    Multi_solver.solve_for_range_imp_factor_VariableL(domain, corners, a0/(section_max_b/imp_factor), imp_factor_list, b, U0, L, t, v, section_max_b, E, fy, normalise_py=True, plot_disp=False, plot_force=True, plot_L=True)
    