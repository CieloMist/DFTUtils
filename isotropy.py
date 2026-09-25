# Python interface to a local install of the ISOTROPY suite (https://stokes.byu.edu/iso/isotropy.php)
import os
import shutil
import subprocess
import tempfile

# Directory holding the ISOTROPY executables AND the data_*.txt files. Set independently on each hpc.
ISOTROPY_PATH = '/gpfs/projects/p32212/DefaultScripts/Softwares/Isotropy'


def findsym(fd, struct, l_tol = 1e-5, ap_tol = 1e-3, apm_tol = .33, o_tol = 1e-3):
    """Use FINDSYM from the ISOTROPY suite to find the symmetry of struct and write a symmetrized cif.

    Args:
    fd (str): path the symmetrized cif is named after. A trailing .cif is dropped, so
        'dir/PRESS_0_kbar.cif' (or 'dir/PRESS_0_kbar') -> 'dir/PRESS_0_kbar_sym.cif'
    struct (ase Atoms): structure to symmetrize
    l_tol (float): FINDSYM latticeTolerance
    ap_tol (float): FINDSYM atomicPositionTolerance
    apm_tol (float): FINDSYM atomicPositionMaxTolerance
    o_tol (float): FINDSYM occupationTolerance

    Returns:
    str: path to the symmetrized cif
    """
    from ase.io import write

    root, ext = os.path.splitext(os.path.abspath(fd))
    if ext.lower() != '.cif':
        root += ext
    sym_cif = f'{root}_sym.cif'

    # FINDSYM locates its data files (data_space.txt, data_wyckoff.txt, ...) through $ISODATA.
    # Without it FINDSYM prints an error but still exits 0 and writes no cif.
    env = dict(os.environ, ISODATA = ISOTROPY_PATH + '/')

    # FINDSYM keyword -> value, the value goes on the line after the keyword
    tolerances = {'!latticeTolerance': l_tol,
                  '!atomicPositionTolerance': ap_tol,
                  '!atomicPositionMaxTolerance': apm_tol,
                  '!occupationTolerance': o_tol}

    # FINDSYM always writes findsym.cif / findsym.log to its working directory, so run each call in its own
    # scratch directory to avoid clobbering between calls and littering the notebook directory
    with tempfile.TemporaryDirectory() as tmp:
        write(os.path.join(tmp, 'temp_findsym.cif'), struct)

        # Generate a FINDSYM input from the non-symmetrized cif
        result = subprocess.run([f'{ISOTROPY_PATH}/findsym_cifinput', 'temp_findsym.cif'],
                                cwd = tmp, env = env, capture_output = True, text = True)
        lines = result.stdout.splitlines()
        if result.returncode != 0 or '!atomPosition' not in lines:
            raise RuntimeError(f'findsym_cifinput failed for {fd}:\n{result.stdout}{result.stderr}')

        # Change the tolerances in the FINDSYM input to what was specified above
        for i, line in enumerate(lines[:-1]):
            if line.strip() in tolerances:
                lines[i + 1] = str(tolerances[line.strip()])
        with open(os.path.join(tmp, 'findsym.in'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

        # Run findsym to generate the symmetrized cif (findsym.cif)
        result = subprocess.run([f'{ISOTROPY_PATH}/findsym', 'findsym.in'],
                                cwd = tmp, env = env, capture_output = True, text = True)
        if result.returncode != 0 or not os.path.isfile(os.path.join(tmp, 'findsym.cif')):
            raise RuntimeError(f'findsym failed for {fd}:\n{result.stdout}{result.stderr}')

        # Change the default cif name to <fd>_sym.cif
        shutil.move(os.path.join(tmp, 'findsym.cif'), sym_cif)

    space_group = next((line.split(':', 1)[1].strip() for line in result.stdout.splitlines()
                        if line.startswith('Space Group:')), 'unknown')
    print(f'{os.path.relpath(sym_cif)}: Space Group {space_group}')

    return sym_cif
