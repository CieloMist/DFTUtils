# A collection of things that I find useful
import numpy as np


# ------------------------------------------------ #
# ASE Relaxation Functions
def write_vasp_settings(updated_vasp_settings):
    """Pull default vasp settings from DFTUtils, then update them with updated_vasp_settings

    Args:
    updated_vasp_settings (dict): dictionary of ASE keywords, user-specified vasp settings
    """
    import json

    # Collect default vasp settings from DFTUtils json
    with open('/projects/p32212/DefaultScripts/DFT_Utilities/vasp_settings.json') as json_file:
        vasp_settings = json.load(json_file)
        json_file.close()

    # Update vasp settings
    for key in updated_vasp_settings.keys():
        vasp_settings[key] = updated_vasp_settings[key]

    # Write out vasp settings json
    with open('vasp_settings.json', 'w') as f:
        json.dump(vasp_settings, f)
        f.close()

    return

# Write strain relaxation -----------------
def write_settings_json(generic_settings, destination):
    """Write filter type and mask OR mace calculation settings to a generic json file

    Args:
    generic_settings dict (filter exmaple): Mask (6 item list in voigt notation), filter type ('strain' or 'Frechet')
    generic_settings dict (mace exmaple): {device:'cpu', 'weights_only': True, 'default_dtype': 'float64'}...

    """
    import json
    with open(destination, 'w') as f:
        json.dump(generic_settings, f)
        f.close()

    return

##########################################################################
# Data Analysis
##########################################################################
def get_dos(orbitals = False, elements = False):
    """Use pymatgen Vasprun w vasprun.xml to get the electronic DOS.
    """
    from pymatgen.io.vasp import Vasprun
    from pymatgen.electronic_structure.core import Spin, OrbitalType

    def produce_spin_separated_dos(pmg_dos, energies):
        # create numpy array which holds spin up, spin down dos
        spin_separated_dos = np.zeros((2, energies.size))
        
        # iterate through density object keys, return dos
        keys = list(pmg_dos.keys())
        for key in keys:
            if key == Spin.up:
                spin_separated_dos[0,:] = pmg_dos[key]
            else:
                spin_separated_dos[1,:] = -1 * pmg_dos[key]

        return spin_separated_dos


    # ------------------------------- #
    # Import calculation results
    calc = Vasprun('vasprun.xml', parse_projected_eigen = True)

    # Get spins and efermi
    efermi = calc.efermi
    energies = calc.tdos.energies - efermi

    energies_dict = {'Energies': energies}

    # ------------------------------- #
    # Get energies, total densities
    total = produce_spin_separated_dos(calc.tdos.densities, energies)
    total_dos = {'Total': total}

    # ------------------------------- #
    # get partial densities - elementwise
    element_dos = {}
    if elements == True:
        element_partial = calc.complete_dos.get_element_dos()
        element_list = list(element_partial.keys())
        
        for element in element_list:
            e_dos = produce_spin_separated_dos(element_partial[element].densities, energies)
            element_dos[str(element).strip('Element ')] = e_dos

    # ------------------------------- #
    # get partial densities - orbitalwise
    orbital_dos = {}
    if orbitals == True:
        orbital_partial = calc.complete_dos.get_spd_dos()
        orbital_list = list(orbital_partial.keys())

        for orbital in orbital_list:
            o_dos = produce_spin_separated_dos(orbital_partial[orbital].densities, energies)
            orbital_dos[str(orbital)] = o_dos
        

    all_results = energies_dict | total_dos | {'Element DOS': element_dos, 'Orbital DOS': orbital_dos}
    
    return all_results


def calculate_moment(x, y, n):
    """Calculate the nth moment of a given dataset.

    Parameters:
    -----------

    x (list, `np.array): x values
    y (list, `np.array): y values
    n (int): nth moment order

    Returns:
    -----------
    Moment, Moment / Area
    """
    # x = x data
    # y = y data
    # n = order of moment

    # returns: moment, centroid
    from scipy.integrate import simpson

    powers = np.full((x.shape), n)

    num = simpson(y = y * np.power(x,powers), x = x)
    denom = simpson(y = y, x = x)

def get_bader_charges(clustering_tol=1e-4):
    """Get the bader charges from a finished DFT calculation. 
    Returns a dictionary of elements and charges. Elements with multiple charge states (within tolerance) are indexed starting from 0.
    **round_kwargs: keywords passed to numpy's rounding function; how many decimals do you want charge rounded to?
    """
    import os

    from pymatgen.command_line.bader_caller import bader_analysis_from_path
    from pymatgen.core.structure import Structure
    from pymatgen.core.composition import Species
    from pymatgen.io.vasp import Outcar
    import numpy as np
    import scipy.cluster.hierarchy as hcluster

    # ---------------------------------------------------------- #
    # EXTRACT BADER RESULTS
    bader = bader_analysis_from_path(os.getcwd())

    # read structure
    struct = Structure.from_file(filename='POSCAR')

    # read outcar, create dictionary of pseudopotential electron counts
    pmg_out = Outcar(filename='OUTCAR')
    pmg_out.read_pseudo_zval()
    # ---------------------------------------------------------- #
    # PROCESS BADER RESULTS INTO WHATEVER FORM YOU LIKE
    # Get bader charges and element for each atomic index
    partial_charges = np.zeros(len(struct))
    element_symbols = np.zeros(len(struct), dtype = 'object')
    for atom_ind, atom in enumerate(struct):
        partial_charges[atom_ind] = -1 * bader['charge_transfer'][atom_ind] # -1 converts charge transfer into partial charge
        element_symbols[atom_ind] = atom.species_string

    # Aggregate unique charges, place results in dictionary for return
    formated_charges = np.vstack((partial_charges, np.zeros(len(partial_charges)))).T
    clusters = hcluster.fclusterdata(formated_charges, clustering_tol, criterion='distance')
    
    clustered_charges, clustered_indices = np.unique(clusters, return_index=True)
    unique_charges = partial_charges[clustered_indices]
    corresponding_elements = element_symbols[clustered_indices].astype('str')

    # ---------------------------------------------------------- #
    # Convert to dictionary - handle case where you have two of the same element with different charge
    for element in corresponding_elements:
        match_indices = np.where(corresponding_elements == element)[0]
        if match_indices.size > 1:
            corresponding_elements[match_indices] = np.char.add(corresponding_elements[match_indices],(match_indices - match_indices[0] + 1).astype('str'))

    return dict(zip(corresponding_elements, unique_charges))


def get_strain(final, initial):
    """Calculate cell strain between two ASE atoms objects. immediately kills user if they input two cells with different numbers of atoms.

    Args:
    final (ASE Atoms):
    initial (ASE Atoms):
    """

    # Put each cell in standard form
    rcell_f, q_f = final.cell.standard_form()
    rcell_i, q_i = initial.cell.standard_form()

    # Get cell parameters from each std form cell
    f_cell = rcell_f.cellpar()
    i_cell = rcell_i.cellpar()

    # calculate strains
    normal_strains = (f_cell[:3] - i_cell[:3]) / i_cell[:3]
    shear_strains = np.tan(np.radians(f_cell[3:]) - np.radians(i_cell[3:]))
    strains = np.concatenate((normal_strains, shear_strains))

    return strains


def copy_files_from_DFTUtilities(filelist):
    """
    Args: filelist (list of strings): names of files to copy from DFTUtilities folder to current working directory.
    """
    import os
    import shutil as sh

    for file in filelist:
        sh.copy('/projects/p32212/DefaultScripts/DFT_Utilities/' + file, os.getcwd())

def get_interstitials(struct, species):
    """
    Args:
    struct (ASE Atoms): structure from which to find interstitial sites
    species (str): interstitial species to insert
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
    from ase import Atoms

    pmg_struct = Structure.from_ase_atoms(struct)

    interstitial = VoronoiInterstitialGenerator()
    interstitial.get_defects(pmg_struct, insert_species=[species])
    defects = interstitial.generate(pmg_struct, insert_species=[species])

    defect_structures = []
    for defect in defects:
        # Create Defect
        insert = Atoms('H', positions=[defect.site.coords], cell = struct.get_cell())
        defect_structure = insert + struct

        # Adjust settings for defect structure
        defect_structure.set_pbc(1)

        # write defect structure to list
        defect_structures.append(defect_structure)

    return defect_structures

def remove_files(files):
    """
    Args: 
    files (list of str): file names to attempt to remove. 
    """
    import os
    import glob
    import numpy as np

    # remove filenames containing stars for globbing 
    to_glob = []
    for ind, file in enumerate(files):
        if '*' in file:
            contains_star = files.pop(ind)
            to_glob.append(contains_star)

    # concatenate glob lists
    for multifile in to_glob:
        to_remove = glob.glob(multifile)
        files = np.concatenate((files, to_remove))
    
    print(f'Removing files: {files}')

    for file in files:
        try:
            os.remove(file)
        except:
            print('Could not remove ' + file)

def make_directories_from_list(directories, delete):
    """
    Args:
    directories [list(str)] - names of directories to create
    delete [bool] - whether to delete directories if they are found
    """
    import os
    import shutil as sh

    for directory in directories:
        directory = str(directory) # handles case where key is an int
        if delete == True and os.path.isdir(directory):
            sh.rmtree(directory)

            os.mkdir(directory)
        elif delete == False and os.path.isdir(directory):
            print('Directory ' + directory + ' already exists.')
        else:
            os.mkdir(directory)


def aggregate_unique_structures(structures, **symmetry_kwargs):
    """
    Aggregates unique structures from a list of structures using XTALComp's symmetry equivalence checker implemented in ASE. 

    Args: 
    structures (list of ASE atoms): input structures to compare
    """
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck
    
    structures_np = np.zeros(len(structures), dtype = 'object')
    for ind, struct in enumerate(structures):
        structures_np[ind] = struct

    unique_structures = []
    comp = SymmetryEquivalenceCheck(**symmetry_kwargs)

    while structures_np.size > 0:
        equivalent_structures = np.zeros(len(structures_np))
        for ind, struct in enumerate(structures_np):
            result = comp.compare(structures_np[0], struct)
            equivalent_structures[ind] = result
        unique_structures.append(structures_np[np.where(equivalent_structures == True)[0][0]])
        structures_np = np.delete(structures_np, [equivalent_structures == True][0])

    return unique_structures
