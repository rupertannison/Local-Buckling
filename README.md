# Local-Buckling
Python code to accompany the analytical design approach to local buckling of thin-walled cross-sections by Rupert Annison and Jurgen Becque

To use this code to find the local buckling failure load of a section:
1. Define section geometry in Section_data.py by copying an example section definition function
2. Enter the material properties and input variables in the code in Main.py
3. Run Main.py and the failure force will be printed to the terminal. Pyplotters will also produce graphs of the failure deflection profile

There are many functions that can be called from Main.py, change the functions being called by toggling the True/False in the if statments
Once the L corresponding to the minimum local buckling failure load has been found, update the section definition in Section_data.py and toggle the True/False if statment so that the L is not solved each time you re-run the code

You will need to install the numpy, pyplot, and scipy packages to run the code

This code only solves failure force in the pure local buckling mode, it does not account for any interactions with distortional or global buckling

Any questions email rupert.annison@gmail.com
