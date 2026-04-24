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
def get_dos(orbitals = False, elements = False, orbitals_and_elements = False, energy_window = (-10,1)):
    """Use pymatgen Vasprun w vasprun.xml to get the electronic DOS.
    """
    from pymatgen.io.vasp import Vasprun
    from pymatgen.electronic_structure.core import Spin
    import glob

    def produce_spin_separated_dos(pmg_dos, energies, energy_window):
        # create numpy array which holds spin up, spin down dos
        spin_separated_dos = np.zeros((2, energies.size))
        energy_match = (energies > energy_window[0]) & (energies < energy_window[1])

        # iterate through density object keys, return dos
        keys = list(pmg_dos.keys())
        for key in keys:
            if key == Spin.up:
                spin_separated_dos[0,:] = pmg_dos[key]
            else:
                spin_separated_dos[1,:] = -1 * pmg_dos[key]

        return spin_separated_dos[:,energy_match]


    # ------------------------------- #
    # Import calculation results
    vasprun_file = glob.glob('vasprun.xml*')[0] # handles zipped vasprun.xml
    calc = Vasprun(vasprun_file, parse_projected_eigen = True)

    # Get spins and efermi
    efermi = calc.efermi
    energies = calc.tdos.energies - efermi

    energy_match = (energies > energy_window[0]) & (energies < energy_window[1]) # technically duplicated code; pass energy_match instead of energy_window to produce_spin_separated_dos
    energies_dict = {'Energies': energies[energy_match]}

    # ------------------------------- #
    # Get energies, total densities
    total = produce_spin_separated_dos(calc.tdos.densities, energies, energy_window)
    total = total
    total_dos = {'Total': total}

    # ------------------------------- #
    # get partial densities - elementwise
    element_dos = {}
    if elements == True:
        element_partial = calc.complete_dos.get_element_dos()
        element_list = list(element_partial.keys())
        
        for element in element_list:
            e_dos = produce_spin_separated_dos(element_partial[element].densities, energies, energy_window)
            element_dos[str(element).strip('Element ')] = e_dos

    # ------------------------------- #
    # get partial densities - orbitalwise
    orbital_dos = {}
    if orbitals == True:
        orbital_partial = calc.complete_dos.get_spd_dos()
        orbital_list = list(orbital_partial.keys())

        for orbital in orbital_list:
            o_dos = produce_spin_separated_dos(orbital_partial[orbital].densities, energies, energy_window)
            orbital_dos[str(orbital)] = o_dos

    # ------------------------------- #
    # get partial densities - orbitalwise and elementwise
    orbital_and_element_dos = {}
    if orbitals_and_elements == True:
        element_list = calc.complete_dos.structure.elements

        for element in element_list:
            element_str = str(element).strip('Element ')
            orbital_and_element_partial = calc.complete_dos.get_element_spd_dos(element_str)
            orbital_list = list(orbital_and_element_partial.keys())

            for orbital in orbital_list:
                o_and_e_dos = produce_spin_separated_dos(orbital_and_element_partial[orbital].densities, energies, energy_window)
                orbital_and_element_dos[element_str + ' ' + str(orbital)] = o_and_e_dos
        

    all_results = energies_dict | total_dos | {'Element DOS': element_dos, 'Orbital DOS': orbital_dos, 'Element and Orbital DOS': orbital_and_element_dos}
    
    return all_results

def get_suborbital_dos(complete_dos, site, shell):
    """
    Gets sub-orbital density of states from pymatgen complete_dos.

    Args:
    complete_dos: pymatgen completedos object
    site: pymatgen PeriodicSite (accessed through calc.complete_dos.structure.sites[ind])
    shell (str): orbital shell, eg. d, f, p, s
    """
    from pymatgen.electronic_structure.core import Orbital, Spin

    orbital_mapping = {'s': [Orbital.s],
                    'p': [Orbital.px, Orbital.py, Orbital.pz],
                    'd': [Orbital.dx2, Orbital.dxy, Orbital.dyz, Orbital.dxz, Orbital.dz2],
                    'f': [Orbital.f_3, Orbital.f_2, Orbital.f_1, Orbital.f0, Orbital.f1, Orbital.f2, Orbital.f3]}

    str_orbital_mapping = {'s': ['s'],
                           'p': ['px', 'py', 'pz'],
                           'd': ['dx2', 'dxy', 'dyz', 'dxz', 'dz2'],
                           'f': ['f-3', 'f-2', 'f-1', 'f0', 'f1', 'f2', 'f3']}

    energies = complete_dos.energies - complete_dos.efermi
    dos = np.zeros((len(orbital_mapping[shell]), energies.size))
    for ind, sub_orbital in enumerate(orbital_mapping[shell]):
        pmg_dos = complete_dos.get_site_orbital_dos(site, sub_orbital)
        dos[ind, :] = pmg_dos.densities[Spin.up]
    return energies, dos, str_orbital_mapping[shell]

def plot_filled_dos_segment(energies, dos, ax, filled = True, energies_on_x = True, kwargs = {}):
    """plots a DOS segment and fills beneath it.
    
    Args:
    -------
    energies (array): energies
    DOS (array): DOS
    ax (plt Axes): axis on which to plot
    filled (bool): whether or not to fill beneath the DOS with a lighter version of the color.
    energies_on_x (bool): horizontal or vertical DOS plot, horizontal default.
    kwargs (dict): kwargs to pass to ax.plot. Combine common settings with specialized ones with common_settings | {'color': 'blue}
    """

    label = kwargs.pop('label', None)

    x_plot, y_plot = energies, dos
    fill = ax.fill_between
    
    if energies_on_x == False:
        x_plot, y_plot = dos, energies
        fill = ax.fill_betweenx
    
    if filled == True:
        fill(x_plot, y_plot, alpha = 0.25, label = label, **kwargs)
    ax.plot(x_plot, y_plot, label = [label if filled == False else None], **kwargs)

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

    return num, num/denom

def get_bader_charges(clustering_tol=1e-4, return_volumes = True):
    """Get the bader charges from a finished DFT calculation. 
    Returns a dictionary of elements and charges. Elements with multiple charge states (within tolerance) are indexed starting from 1.
    """

    import os
    import numpy as np
    import glob

    from pymatgen.command_line.bader_caller import bader_analysis_from_path
    from pymatgen.core.structure import Structure
    from pymatgen.core.composition import Species
    from pymatgen.io.vasp import Outcar
    
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
    atomic_volumes = np.zeros(len(struct))
    element_symbols = np.zeros(len(struct), dtype = 'object')
    for atom_ind, atom in enumerate(struct):
        partial_charges[atom_ind] = -1 * bader['charge_transfer'][atom_ind] # -1 converts charge transfer into partial charge
        atomic_volumes[atom_ind] = bader['atomic_volume'][atom_ind]
        element_symbols[atom_ind] = atom.species_string

    # Aggregate unique charges, place results in dictionary for return
    formated_charges = np.vstack((partial_charges, np.zeros(len(partial_charges)))).T
    clusters = hcluster.fclusterdata(formated_charges, clustering_tol, criterion='distance')

    clustered_charges, clustered_indices = np.unique(clusters, return_index=True)
    unique_charges = partial_charges[clustered_indices]
    corresponding_elements = element_symbols[clustered_indices].astype('str')
    corresponding_volumes = atomic_volumes[clustered_indices]

    # ---------------------------------------------------------- #
    # Convert to dictionary - handle case where you have two of the same element with different charge
    for element in corresponding_elements:
        match_indices = np.where(corresponding_elements == element)[0]
        if match_indices.size > 1:
            corresponding_elements[match_indices] = np.char.add(corresponding_elements[match_indices],(match_indices - match_indices[0] + 1).astype('str'))

    to_return = {'charges': dict(zip(corresponding_elements, unique_charges))}
    if return_volumes == True:
        to_return = to_return | {'volumes': dict(zip(corresponding_elements, corresponding_volumes))}

    return to_return


def get_strain(final, initial, standard_form=True):
    """Calculate cell strain between two ASE atoms objects. immediately kills user if they input two cells with different numbers of atoms.

    Args:
    final (ASE Atoms):
    initial (ASE Atoms):
    """

    # Put each cell in standard form
    if standard_form == True:
        rcell_f, q_f = final.cell.standard_form()
        rcell_i, q_i = initial.cell.standard_form()
    else:
        rcell_f = final.cell
        rcell_i = initial.cell

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

def render_povray(filename, struct, atom_render_settings, povray_settings, isosurfaces = None, bonds = True):
    """
    Renders an ASE atoms object using POVRAY. Basically just so I don't have to repeat all this code every time.
    Args:
    filename (str): Path (relative to current working directory) to place the rendered image (WITHOUT .PNG)
    struct (ASE atoms): structure to render
    atom_render_settings (dict): keyword args for atom render, e.g. 'radii', 'colors', 'show_unit_cell', etc.
    povray_settings (dict): render settings for povray, e.g. 'transparent', 'textures', etc.
    isosurfaces (list of POVRAYIsosurfaces): isosurface data
    bonds (bool): whether to render bonds or not

    """

    import os
    from ase.io.pov import POVRAY, get_bondpairs, set_high_bondorder_pairs
    from ase.io import write

    if bonds == True:
        bondpairs = get_bondpairs(struct, radius=1.0)
        high_bondorder_pairs = {}
        def setbond(target, order):
            high_bondorder_pairs[(0, target)] = ((0, 0, 0), order, (0.1, -0.2, 0))
        bondpairs = set_high_bondorder_pairs(bondpairs, high_bondorder_pairs)

        povray_settings['bondatoms'] = bondpairs

    pov_object = write(filename + '.pov', struct, isosurface_data = isosurfaces, **atom_render_settings, povray_settings = povray_settings)
    print(pov_object.path)
    pov_object.render(povray_executable='/gpfs/projects/p32212/Software_LifeEasy/povray/povray/unix/povray')


def render_chg_slice(struct, slice_axis, slice_depth, ax = None, chg_file = 'CHGCAR', darken_sliced_atoms = True, contourf = True, contour_settings = {}, contourf_settings = {}, plot_atoms_settings = {}):
    """
    Renders the charge density of a particular slice of a structure.

    Args
    ----------------------
    struct (ASE Atoms): Structure to render
    slice_axis (str: 'x, y or z'): Plane to look at
    slice_depth (float): Scaled position of the slice to look at
    contour_settings (dict): passed directly to plt.contour
    plot_atoms_settings (dict): passed directly to ase.visualize.plot_atoms
    """
    from ase.calculators.vasp import VaspChargeDensity
    from ase.io.utils import PlottingVariables
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors
    # ----------------------------------------- #
    def get_offset(struct, rotation):
        pvars = PlottingVariables(struct, rotation = rotation, scale=1.0)
        offset = pvars.to_atom_positions(pvars.offset) / 2
        
        bbox = pvars.get_bbox()
        return offset, bbox

    def get_plane(point, normal):
        A, B, C = normal
        D = np.dot(point, normal)
        return [A, B, C, D]
        
    # ------------------------------------------ #
    # set struct to standard form:
    struct.set_cell(struct.cell.standard_form()[0], scale_atoms = True)
    cell_lengths = struct.cell.lengths()

    # ------------------------------------------ #
    # Read in and normalize charge
    vchg = VaspChargeDensity(chg_file)
    chg = vchg.chg[0] / np.max(vchg.chg[0])

    # ------------------------------------------ #
    # Define coordinate mesh for charge
    x_coords = np.linspace(0, 1, len(chg[:,0,0]))
    y_coords = np.linspace(0, 1, len(chg[0,:,0]))
    z_coords = np.linspace(0, 1, len(chg[0,0,:]))

    full_data = np.stack(np.meshgrid(x_coords,y_coords,z_coords, indexing = 'ij'), axis = 3)

    # ------------------------------------------ #
    # Choose what slice you're looking at:
    if slice_axis == 'yz':
        rotation = '-90x,0y,0z'

        offset, bbox = get_offset(struct, rotation)
        scaled_data = np.abs((full_data[:,:,:,:] @ struct.cell[:]) - offset)
        
        mesh = full_data[:,:,0,:2]
        X = scaled_data[:,0,:,0]
        Y = scaled_data[:,0,:,2]
        
        slice_index = int(slice_depth * chg.shape[1])
        Z = chg[:,slice_index,:]

        # -------------- #
        # Get plane equation
        A,B,C,D = get_plane([0,slice_depth * cell_lengths[1],0], [0,1,0])

    if slice_axis == 'xz':
        rotation = '0x,-90y,0z'

        offset, bbox = get_offset(struct, rotation)
        scaled_data = np.abs((full_data[:,:,:,:] @ struct.cell[:]) - offset)

        mesh = full_data[:,:,0,:2]
        X = scaled_data[:,:,0,0]
        Y = scaled_data[:,:,0,1]
        
        slice_index = int(slice_depth * chg.shape[2])
        Z = chg[:,:,slice_index]

        # -------------- #
        # Get plane equation
        A,B,C,D = get_plane([0,0,slice_depth * cell_lengths[2]], [0,0,1])
    
    if slice_axis == 'xy':
        rotation = '0x,0y,-90z'

        offset, bbox = get_offset(struct, rotation)
        scaled_data = np.abs((full_data[:,:,:,:] @ struct.cell[:]) - offset)

        mesh = full_data[:,:,0,:2]
        X = scaled_data[:,:,0,0]
        Y = scaled_data[:,:,0,1]

        slice_index = int(slice_depth * chg.shape[2])
        Z = chg[:,:,slice_index]

        # -------------- #
        # Get plane equation
        A,B,C,D = get_plane([0,0,slice_depth * cell_lengths[2]], [0,0,1])

    # ------------------------------------------ #
    # Get intersection of atoms with plane:
    sphere_centers = struct.get_positions()
    ones = -1 * np.ones((1,sphere_centers.shape[0]))

    sphere_centers_concat = np.concatenate((sphere_centers, ones.T), axis = 1)
    radii = np.array([covalent_radii[atom.number] for atom in struct])

    plane_distances = np.sum([A,B,C,D] * sphere_centers_concat, axis = 1) / np.linalg.norm([A,B,C])

    back_intersections = np.logical_and(plane_distances < radii, plane_distances <= 0)
    front_intersections = np.logical_and(plane_distances > -1 * radii, plane_distances >= 0)

    back_non_intersections = np.logical_and(plane_distances < -1 * radii, plane_distances <= 0)
    front_non_intersections = np.logical_and(plane_distances > radii, plane_distances >= 0)

    # --------------------------------------------------------- #
    # Tag and color atoms based on intersection type
    # Turn atoms entirely in front of plane clear 
    front_non_intersection_colors = np.full(len(radii), 1)
    back_non_intersection_colors = np.full(len(radii), 2)

    colors = np.array([jmol_colors[atom.number] for atom in struct])
    colors_w_alpha = np.concatenate((colors, np.ones((1,colors.shape[0])).T), axis = 1)
    
    colors_w_alpha[:,3][front_non_intersections] = 0

    # --------------------------------------------------------- #
    # For atoms whose radius intersects the plane, establish the radius of the cut circle
    radii_cut = np.sqrt(radii ** 2 - plane_distances ** 2)
    to_combine = np.logical_and(front_intersections == False, back_intersections == False)
    radii_cut[to_combine] = radii[to_combine]

    colors_cut_atoms = colors_w_alpha.copy()
    if darken_sliced_atoms == True:
        colors_cut_atoms[to_combine == False, :3] = colors_cut_atoms[to_combine == False, :3] * 0.7

    # --------------------------------------------------------- #
    # Modify copied (second plotted structure) to render atom intersections
    
    # delete atoms in front of image plane that don't intersect it
    # then delete atoms behind image plane with no intersections
    struct_copy = struct.copy()

    to_delete_copy = np.logical_or(front_non_intersections, back_non_intersections)

    del struct_copy[to_delete_copy]
    colors_cut_atoms = np.delete(colors_cut_atoms, to_delete_copy, axis=0)
    radii_cut = np.delete(radii_cut, to_delete_copy, axis=0)

    # --------------------------------------------------------- #
    # Modify atoms (original structure) to render those without intersections

    # delete atoms in front of image plane that don't intersect it
    # then delete atoms with back intersections
    struct_original = struct.copy()
    
    to_delete_original = np.logical_or(front_non_intersections, back_intersections)

    del struct_original[to_delete_original]
    colors_w_alpha = np.delete(colors_w_alpha, to_delete_original, axis=0)

    # ------------------------------------------ #
    # Plot
    if ax == None:
        fig, ax = plt.subplots()

    plotting_variables = {'bbox': bbox}

    if np.any(to_delete_original == False):
        plot_atoms(struct_original, ax, rotation = rotation, **plot_atoms_settings, **plotting_variables, colors = colors_w_alpha)
    plot_atoms(struct_copy, ax, rotation = rotation, **plot_atoms_settings, **plotting_variables, colors = colors_cut_atoms, radii = radii_cut)
    
    if contourf == True:
        ax.contourf(X, Y, Z, **contourf_settings)

    ax.contour(X, Y, Z, **contour_settings)

    return ax

def interchange_atoms_ase_phonopy(struct):
    """
    Converts ASE atoms to phonopy atoms and vice versa. Made for easier readability/interoperability between the packages.

    Args:
    ase_atoms (Atoms): ASE atoms object
    phonopy_atoms (PhonopyAtoms): Phonopy atoms object
    """

    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms

    if isinstance(struct, Atoms):
        atoms = PhonopyAtoms(symbols=struct.get_chemical_symbols(),
                    numbers=struct.get_atomic_numbers(),
                    masses=struct.get_masses(),
                    scaled_positions=struct.get_scaled_positions(),
                    positions=struct.get_positions(),
                    cell=struct.get_cell())

    elif isinstance(struct, PhonopyAtoms):
        atoms = Atoms(symbols=struct.symbols,
                      positions=struct.positions,
                      masses=struct.masses,
                      cell=struct.cell)

    return atoms

def interchange_atoms_ase_spglib(struct):
    """
    Converts between ase atoms and spglib cell.

    Args
    --------
    struct (Ase Atoms or spglib Cell): object to convert
    """
    from ase import Atoms

    if isinstance(struct, Atoms):
        to_return = (struct.cell[:], struct.get_scaled_positions(), struct.get_atomic_numbers())
    else:
        to_return = Atoms(numbers = struct[2], scaled_positions=struct[1], cell=struct[0])

    return to_return

def symmetrize_cell(struct, primitive = True, symprec = 1e-4):
    """
    Symmetrizes a unit cell into primitive or conventional using spglib.
    
    Args:
    struct (ASE Atoms): structure to symmetrize
    primitive (bool): whether or not to return primitive (True) or conventional (False)
    """
    from DFTUtils import interchange_atoms_ase_spglib
    import spglib

    # convert ASE structure to spglib cell:
    cell = interchange_atoms_ase_spglib(struct)
    
    # make spglib cell
    conv_cell = spglib.standardize_cell(cell, 
                                        to_primitive=primitive, 
                                        no_idealize=False, 
                                        symprec=symprec)

    # reconvert to ASE
    struct = interchange_atoms_ase_spglib(conv_cell)
    struct.set_pbc([1,1,1])

    return struct

def run_phonons(initial_struct, phonopy_settings = {}, updated_vasp_settings = None, directory_suffix = None):
    """
    Setup and submit phonons as a set of batch calculations via Phonopy.

    --Args--
    atoms (ASE Atoms):
    """


    from DFTUtils import interchange_atoms_ase_phonopy, copy_files_from_DFTUtilities, write_pickle
    from ase.io import read, write
    from phonopy import Phonopy
    import subprocess
    import shutil as sh
    import os

    # ----------------------------------------------- #
    # Create phonons directory
    dirlist = ['Phonons' if directory_suffix == None else 'Phonons_' + directory_suffix]
    make_directories_from_list(dirlist, delete = True)
    os.chdir(dirlist[0])

    # ------------------------------------------------ #
    # Setup folders, run calculation:
    atoms = interchange_atoms_ase_phonopy(initial_struct)
    phonons = Phonopy(atoms, **phonopy_settings)

    directory_suffix = phonopy_settings.pop('directory_suffix', None)
    displacement_distance = phonopy_settings.pop('displacement_distance', 0.01)

    phonons.generate_displacements(distance = displacement_distance, is_plusminus = True) # can move is_plusminus into settings later
    supercells = phonons.supercells_with_displacements
    # phonons.save('phonopy_disp.yaml')
    write_pickle('phonons.pickle', phonons)

    # ------------------------------------------------ #
    # Setup folders, run calculation:
    displacement_directories = [f'disp-{ind:03}' for ind in range(len(supercells))]

    make_directories_from_list(displacement_directories, delete = False)
    for ind, phonopy_atoms in enumerate(supercells):
        os.chdir(displacement_directories[ind])
        # ------------------------------------ #
        # Get Potential Energies/Run Calculation:
        # Convert to ASE, set calc parameters
        struct = interchange_atoms_ase_phonopy(phonopy_atoms)
        struct.set_pbc([1,1,1])
        write('Initial.traj', struct, format = 'traj')

        # get files from dftutils:
        copy_files_from_DFTUtilities(['ASE_SinglePoint.py', 'HPC_Submission_Scripts/JobSubmission_SinglePoint.q'])

        if updated_vasp_settings == None:
            sh.copy('../../vasp_settings.json', 'vasp_settings.json')
        else:
            write_vasp_settings(updated_vasp_settings)

        # run calculation
        subprocess.run(['sbatch','JobSubmission_SinglePoint.q'])

        os.chdir('../')

    return

def process_phonons():
    """
    Post process the results of batch-submitted phonon calculations with ASE and Phonopy.
    """

    import glob
    import os

    from ase.io import read
    import phonopy
    from DFTUtils import read_pickle

    # ----------------------------------- #
    # Read displacement dataset into a Phonopy() object
    #phonons = phonopy.load('phonopy_disp.yaml')
    phonons = read_pickle('phonons.pickle')
    force_sets = np.zeros((len(phonons.displacements), len(phonons.supercell), 3)) # m displacements, n atoms in structures, 3 DOF
    
    supercells = phonons.supercells_with_displacements
    # ----------------------------------- #
    # read force sets:
    displacement_directories = [f'disp-{ind:03}' for ind in range(len(supercells))]

    for ind, disp_dir in enumerate(displacement_directories):
        os.chdir(disp_dir)

        struct = read('Final.traj', format = 'traj')
        force_sets[ind][:,:] = struct.get_forces()

        os.chdir('../')

    # ------------------------------------ #
    # Tell phonopy object what the forces were and make force constants
    phonons.forces = force_sets
    phonons.produce_force_constants()

    phonons.save() # write results to file

    return phonons

def modulate_phonons(phonons, q_point, band_index, amplitude = 0.01, phase_factor = 0, return_pmg = False):
    """
    Runs phonopy modulation given a phonopy object, qpoint, band, and amplitude.

    Args:
    phonons (phonopy object): phonopy object from process_phonons
    q_point (NDArray): qpoint at which to modulate
    band_index (int): band, indexes starting at 0
    amplitude (float): amplitude of modulation in Angstroms
    phase_factor (float): phase factor, eg. timing difference for the wave
    """
    from phonopy.phonon.modulation import Modulation
    from DFTUtils import interchange_atoms_ase_phonopy

    # ---------------------------------------------- #
    # compile inputs into list for Modulation object
    phonon_modes = [q_point, band_index, amplitude, phase_factor] # [q_point, band_index, amplitude, phase_factor] -> defaults: [[0,0,0], 0, 0, 0]
    
    # ---------------------------------------------- #
    # Run modulation
    mod = Modulation(phonons.dynamical_matrix, np.diag(phonons.supercell_matrix), [phonon_modes])
    mod.run()

    # ---------------------------------------------- #
    # return modulated supercell
    supercell = interchange_atoms_ase_phonopy(mod.get_modulated_supercells()[0])
    supercell.set_pbc([1,1,1])

    if return_pmg:
        from pymatgen.io.ase import AseAtomsAdaptor
        supercell = AseAtomsAdaptor.get_structure(supercell)

    return supercell


def repair_phonons(updated_vasp_settings = None):
    """
    Find which phonon calculations didn't finish and restart them.
    """

    from pymatgen.io.vasp.outputs import Vasprun
    import glob
    from DFTUtils import write_vasp_settings
    import os
    import subprocess
    from ase.io import read

    # ----------------------------------- #
    # find which phonon calculations didn't work:
    displacement_directories = glob.glob('disp*/')

    gone_wrong = []
    for ind, disp_dir in enumerate(displacement_directories):
        os.chdir(disp_dir)

        try:
            struct = read('Final.traj')
            struct.get_potential_energy()
            # vasprun = Vasprun('vasprun.xml')
            # if not vasprun.converged_electronic:
            #     gone_wrong.append(disp_dir)
        except:
            gone_wrong.append(disp_dir)

        os.chdir('../')

    # ----------------------------------- #
    # Go through wrong directories and restart calculations:
    for directory in gone_wrong:
        os.chdir(directory)

        if updated_vasp_settings != None:
            write_vasp_settings(updated_vasp_settings)

        print(f'Restarting {directory} calculation...')
        subprocess.run(['sbatch', 'JobSubmission_SinglePoint.q'])

        os.chdir('../')
    
    return

def repair_calcs(directory_root, script_name = 'SinglePoint', copy_script = True, updated_vasp_settings = None):
    """
    Find which calculations didn't finish and restart them.

    Params
    ----------
    directory_root (str): glob expression for the directories to search.
    """

    from pymatgen.io.vasp.outputs import Vasprun
    import glob
    from DFTUtils import write_vasp_settings, copy_files_from_DFTUtilities
    import os
    import subprocess

    origin = os.getcwd()
    # ----------------------------------- #
    # find which calculations didn't work:
    directories = glob.glob(directory_root)
    print(directories)

    gone_wrong = []
    for ind, disp_dir in enumerate(directories):
        try:
            vr = glob.glob(disp_dir + '/vasprun.xml*')[0] # handle zipped vasprun
            vasprun = Vasprun(vr)
            if not vasprun.converged_electronic:
                gone_wrong.append(disp_dir)
        except:
            gone_wrong.append(disp_dir)

    # ----------------------------------- #
    # Go through wrong directories and restart calculations:
    for directory in gone_wrong:
        print(f'Restarting {directory} calculation...')
        
        # ---------------------- #
        # find top level dir
        normpath = os.path.normpath(directory)
        splitpath = normpath.split(os.sep)

        # ----------------------- #
        # change dirs and start calc
        os.chdir(splitpath[0])

        script_to_run = glob.glob(os.getcwd() + '/*' + script_name + '*.q')[0]
        if updated_vasp_settings != None:
            write_vasp_settings(updated_vasp_settings)

        if copy_script == True:
            script_to_copy = os.path.basename(script_to_run)
            copy_files_from_DFTUtilities(['HPC_Submission_Scripts/' + script_to_copy])

        # subprocess.run(['sbatch', script_to_run])
        os.chdir(origin)

    return


def write_pickle(filename, to_pickle):
    """
    Pickles whatever you pass.

    Args
    -------
    filename (str): pickle file name.
    to_pickle (Any): object, dict, etc. to pickle.
    """
    import numpy as np
    import pickle
    # ------------------------------- #
    # open pickle file
    with open(filename, 'wb') as f:
        # Use the highest protocol available for best performance and compatibility
        pickle.dump(to_pickle, f, pickle.HIGHEST_PROTOCOL)

    return

def read_pickle(filename):
    """
    Unpickles whatever you pass.

    Args
    -------
    filename (str): pickle file name.
    to_pickle (Any): object, dict, etc. to pickle.
    """
    import numpy as np
    import pickle
    # ------------------------------- #
    # open pickle file
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def process_cohp_from_lobsterpy_analysis(ana, site_bond, orbital_bond, energy_window = None, length_window = [0,10], cohp_or_icohp = 'COHP'):
    """
    kinda a stupid function to process data from lobsterpy. In the future, rework so that the orbital resolved cohps are stored in a 2d numpy array and carry out the sum after all the if statements.
    """
    from pymatgen.electronic_structure.core import Spin

    # pymatgen - handles actual data
    comp = ana.completecoxx
    energies = comp.energies - comp.efermi
    orb_res_cohps = comp.orb_res_cohp

    # lobsterpy - handles accessing data
    orb_bonds = ana.get_site_orbital_resolved_labels()
    relevant_sub_orbitals = orb_bonds[site_bond][orbital_bond]['relevant_sub_orbitals']
    bond_labels = orb_bonds[site_bond][orbital_bond]['bond_labels']
    
    # Add each orbital bond e.g. [5d-2p]
    cohp_to_collapse = np.zeros(len(energies))
    for bond in bond_labels:
        for ind, sub_orbital in enumerate(relevant_sub_orbitals):

            length = orb_res_cohps[bond][sub_orbital]['length']
            
            if (length > length_window[0]) & (length < length_window[1]):
                cohp_to_collapse += orb_res_cohps[bond][sub_orbital][cohp_or_icohp][Spin.up]
                
    # do processing to return correct energy window:
    if energy_window != None:
        match = (energies > energy_window[0]) & (energies < energy_window[1])
        energies = energies[match]
        cohp_to_collapse = cohp_to_collapse[match]

    return cohp_to_collapse, energies
