# Operating System/File Utilities
import os
import glob
import re
import shutil

# Space Group Lib
import spglib as spg

# ASE
from ase.io import read, write
from ase.io.vasp import read_vasp_xml, read_vasp_out
from ase.calculators.vasp import Vasp
from ase import Atoms
from ase.formula import Formula

# Pymatgen
from pymatgen.io.vasp import Vasprun, Chgcar, Wavecar, Poscar, Kpoints

# Python utils
import numpy as np
from scipy.integrate import cumulative_simpson, simpson
from scipy.sparse import csc_matrix
import pandas as pd

# -------------------------------------------------------------------- #
# Goal: process high resolution DOS and calculate the moments of their projections.
# Jupyter kernels are too wimpy to do this for some reason.

# ------------------------------------------------------------------------------------------- #
# Function Definitions
def calculate_moment(x, y, n):
    # x = x data
    # y = y data
    # n = order of moment

    # returns: moment, centroid

    powers = np.full((x.shape), n)

    num = simpson(y = y * np.power(x,powers), x = x)
    denom = simpson(y = y, x = x)

    return num, num / denom

# ------------------------------------------------------------------------------ #
# Define Default Structure:
struct = read('POSCAR_Initial')

# -------------------------------------------------------------------------------------------------- #
# Electronic Structure
dirlist = glob.glob('PRESS_*/')
print(dirlist)
# _ = dirlist.pop(0)

outputs=['Pressures',
        'Actual Pressures',
        'Fermi Level',
        'Energies', # for DOS
        'Total Densities Spin Up', # normalized
        'Total Densities Spin Down', # normalized
        'Integrated Total Densities Spin Up', # normalized
        'Integrated Total Densities Spin Down', # normalized 
        'Volumes']

# ------------------------------------------ #
    # Element Projected Density of States
pdos = []
for species in np.unique(struct.get_chemical_symbols()):
    pdos.append(species + ' Partial Energy')
    for spin in ['Up', 'Down']:
        pdos.append(species + ' Partial Density' + ' Spin ' + spin)
        pdos.append(species + ' Integrated Partial Density' + ' Spin ' + spin)

    # Orbital Projected Density of States
odos = []
for orbital in ['s', 'p', 'd', 'f']:
    odos.append(orbital + ' Partial Energy')
    for spin in ['Up', 'Down']:
        odos.append(orbital + ' Partial Density' + ' Spin ' + spin)
        odos.append(orbital + ' Integrated Partial Density' + ' Spin ' + spin)

# ------------------------------------------ #
# Energy Window Properties: Zeroth, First, Second Moments
    # Define the set of Orbitals/Projections you're looking at:
projections_and_windows = {
                        'f': [(-20,0)],
                        'd': [(-20,0)]
                        } #'I': [(-10,0), (-20,0)],
                        #'Nd': [(-10,0),(-20,0)],

moments = []
for key in projections_and_windows.keys():
    for w_ind, window in enumerate(projections_and_windows[key]):
        moments.append(key + ' Window ' + str(w_ind) + ' Zeroth Moment')
        moments.append(key + ' Window ' + str(w_ind) + ' First Moment')
        moments.append(key + ' Window ' + str(w_ind) + ' Centroid')

# -------------------------------------------------------------------------------------------- #
# Start Data Collection Loop:

columns = np.concatenate((outputs, pdos, odos, moments))

data = pd.DataFrame(np.empty((len(dirlist),len(columns))), columns=columns, dtype='object')

for ind, dir in enumerate(dirlist):
    os.chdir(dir.strip('\''))
    print(os.getcwd())
    #------------------------------------------------------------------------------------------------
    # APPLIED CONDITIONS
    point = re.findall(r"[-+]?(?:\d*\.*\d+)", dir.strip('\''))
    data.at[ind,'Pressure'] = int(point[0])
    print('Pressure: ', int(point[0]))

    # -----------------------------------------------------------------------------------------------
    # Structure
    struct = read('POSCAR_Final', format = 'vasp')

        # Number of formula units with which to normalize properties
    form = Formula(struct.get_chemical_formula())
    n_formunits = form.reduce()[1]
        
        # Volume
    data.at[ind, 'Volumes'] = struct.get_volume() / n_formunits
    calc_ase = read_vasp_out('Dynamic_Relax/OUTCAR')
        
        # Stress
    stresses = calc_ase.get_stress(voigt=False) * 1602.17
    pressure = -1 * np.sum(np.trace(stresses)) / 3
    data.at[ind, 'Actual Pressures'] = pressure

# -----------------------------------------------------------------------------------------------
# ELECTRONIC PROPERTIES
    calc = Vasprun('Static_Relax/vasprun.xml', parse_dos=True, parse_projected_eigen=False, parse_eigen=False, parse_potcar_file=False, occu_tol = 1e-8)
    efermi = calc.efermi
    data.at[ind, 'Fermi Level'] = efermi

        # ------------------------------------------ #
        # TOTAL DOS
    total = calc.tdos
        
    energies = total.energies - efermi # shifted so ef is at 0
    data.at[ind, 'Energies'] = energies

    data.at[ind, 'Total Densities Spin Up'] = csc_matrix(list(total.densities.values())[0] / n_formunits)
    data.at[ind, 'Total Densities Spin Down'] = csc_matrix(list(total.densities.values())[1] / n_formunits)

    data.at[ind, 'Integrated Total Densities Spin Up'] = csc_matrix(cumulative_simpson(y = list(calc.tdos.densities.values())[0], x = energies, initial = 0) / n_formunits)
    data.at[ind, 'Integrated Total Densities Spin Down'] = csc_matrix(cumulative_simpson(y = list(calc.tdos.densities.values())[1], x = energies, initial = 0) / n_formunits)

        # ------------------------------------------ #
        # ELEMENT PROJECTED DOS
    partial = calc.complete_dos.get_element_dos()
    p_keys = list(partial.keys())

    for key in p_keys:
        energies = partial[key].energies - efermi # shifted so ef is at 0
        data.at[ind, str(key) + ' Partial Energy'] = energies

        data.at[ind, str(key) + ' Partial Density Spin Up'] = csc_matrix(list(partial[key].densities.values())[0] / n_formunits)
        data.at[ind, str(key) + ' Integrated Partial Density Spin Up'] = csc_matrix(cumulative_simpson(y = list(partial[key].densities.values())[0], x = energies, initial = 0) / n_formunits)

        data.at[ind, str(key) + ' Partial Density Spin Down'] = csc_matrix(list(partial[key].densities.values())[1] / n_formunits)
        data.at[ind, str(key) + ' Integrated Partial Density Spin Down'] = csc_matrix(cumulative_simpson(y = list(partial[key].densities.values())[1], x = energies, initial = 0) / n_formunits)
            
        # ------------------------------------------ #
        # ORBITAL PROJECTED DOS
    orbital = calc.complete_dos.get_spd_dos()
    o_keys = list(orbital.keys())

    for key in o_keys:
        energies = orbital[key].energies - efermi # shifted so ef is at 0
        data.at[ind, str(key) + ' Partial Energy'] = energies
            
        data.at[ind, str(key) + ' Partial Density Spin Up'] = csc_matrix(list(orbital[key].densities.values())[0] / n_formunits)
        data.at[ind, str(key) + ' Integrated Partial Density Spin Up'] = csc_matrix(cumulative_simpson(y = list(orbital[key].densities.values())[0], x = energies, initial = 0) / n_formunits)

        data.at[ind, str(key) + ' Partial Density Spin Down'] = csc_matrix(list(orbital[key].densities.values())[1] / n_formunits)
        data.at[ind, str(key) + ' Integrated Partial Density Spin Down'] = csc_matrix(cumulative_simpson(y = list(orbital[key].densities.values())[1], x = energies, initial = 0) / n_formunits)

    # --------------------------------------- #
    # # Charge Density
    # chg = Chgcar.from_file('CHGCAR')
    # data.at[ind, 'Charge Density'] = chg

    # --------------------------------------- #
    # Moments:
    for key in projections_and_windows.keys():
        for w_ind, window in enumerate(projections_and_windows[key]):
            # get indices matching energies in window from DOS
            ind_low = np.abs(data[key + ' Partial Energy'][ind] - window[0]).argmin()
            ind_high = np.abs(data[key + ' Partial Energy'][ind] - window[1]).argmin()

            # Calculate Moments                
            energy = data[key + ' Partial Energy'][ind][ind_low:ind_high]
            sum_partial_density = np.abs(data[key + ' Partial Density Spin Up'][ind].toarray()) + np.abs(data[key + ' Partial Density Spin Down'][ind].toarray())
            sliced_partial_density = sum_partial_density.flatten()[ind_low:ind_high]

            data.at[ind, key + ' Window ' + str(w_ind) + ' Zeroth Moment'], _ = calculate_moment(energy, sliced_partial_density, 0) # sum total of absolute values; spin up and spin down electrons.
            _, data.at[ind, key + ' Window ' + str(w_ind) + ' Centroid'] = calculate_moment(energy, sliced_partial_density, 1) # no sum total, track if spin up is notably separated from spin down...?
            data.at[ind, key + ' Window ' + str(w_ind) + ' First Moment'], _ = calculate_moment(energy, sliced_partial_density, 1)

    os.chdir('../')

data.sort_values(by = 'Pressure', ignore_index=True, inplace=True)
print(data['Pressure'])

data.to_pickle('processed_data.pkl')

cols_to_del = []
for col in data:
    if ('Moment' not in col) and ('Centroid' not in col):
        cols_to_del.append(col)

data.drop(cols_to_del, axis=1, inplace=True)
data.to_csv('moments.csv', data)



