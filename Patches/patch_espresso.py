#!/usr/bin/env python3
"""Patch ASE's espresso.py so it can read spin-polarized DFT+U (Hubbard)
pw.x output.

Problem
-------
In a collinear spin-polarized DFT+U run, pw.x prints the Hubbard occupation
matrices inside ``enter write_ns`` ... ``exit write_ns`` blocks. Those blocks
contain ``SPIN`` headers and float values. ASE's eigenvalue parser only tries
to skip a write_ns block if it sits on the line *immediately* after
"End of self-consistent calculation"; in QE 7.x there is usually a blank line
in between, so the skip is missed, the occupation-matrix floats are parsed as
eigenvalues, and this assertion trips:

    assert len(eigenvalues[0]) == len(ibzkpts)

which makes ase.io.read() fail for the whole file -- including energies and
forces.

Fix
---
Replace the "# Bands" parsing block in read_espresso_out with a version that:
  * skips ``enter write_ns`` ... ``exit write_ns`` blocks wherever they occur,
  * bounds the loop and stores a trailing eigenvalue group safely,
  * degrades gracefully (warns + skips only the k-point/eigenvalue data)
    instead of asserting, so energies and forces are always returned.

Usage
-----
    python patch_espresso.py            # patch the ase install on this PATH
    python patch_espresso.py --dry-run  # show what would change, write nothing

A timestamped backup (espresso.py.bak-YYYYMMDD-HHMMSS) is written next to the
original before any change. Re-running detects an already-patched file and
does nothing.
"""

import argparse
import datetime
import py_compile
import re
import shutil
import sys


NEW_BLOCK = '''        # Bands
        kpts = None
        kpoints_warning = "Number of k-points >= 100: " + \\
                          "set verbosity='high' to print the bands."

        for bands_index in indexes[_PW_BANDS] + indexes[_PW_BANDSTRUCTURE]:
            if image_index < bands_index < next_index:
                bands_index += 1

                # Without parsed k-point coordinates we cannot build the
                # eigenvalue/k-point structure. Energies, forces, stress and
                # magmoms are read independently and are unaffected.
                if ibzkpts is None:
                    continue

                spin, bands, eigenvalues = 0, [], [[], []]
                kpoints_warning_found = False

                while bands_index < next_index:
                    line = pwo_lines[bands_index]

                    if line.strip() == kpoints_warning:
                        kpoints_warning_found = True
                        break

                    L = line.replace('-', ' -').split()
                    if len(L) == 0:
                        if len(bands) > 0:
                            eigenvalues[spin].append(bands)
                            bands = []
                    elif L[0] == 'enter' and 'write_ns' in L:
                        # Spin-polarised DFT+U prints Hubbard occupation
                        # matrices inside 'enter write_ns' ... 'exit write_ns'
                        # blocks. These contain 'SPIN' headers and float
                        # values that would otherwise be parsed as
                        # eigenvalues, so skip them wherever they appear.
                        while ('exit write_ns' not in pwo_lines[bands_index]
                               and bands_index + 1 < next_index):
                            bands_index += 1
                    elif L == ['occupation', 'numbers']:
                        # Skip the lines with the occupation numbers
                        bands_index += len(eigenvalues[spin][0]) // 8 + 1
                    elif L[0] == 'k' and L[1].startswith('='):
                        pass
                    elif 'SPIN' in L:
                        if 'DOWN' in L:
                            spin += 1
                    else:
                        try:
                            bands.extend(map(float, L))
                        except ValueError:
                            break
                    bands_index += 1

                if kpoints_warning_found:
                    continue

                # Store the final group of eigenvalues if it was not
                # terminated by a trailing blank line.
                if len(bands) > 0:
                    eigenvalues[spin].append(bands)

                # Be forgiving rather than crashing: if the parsed
                # eigenvalues cannot be matched to the k-points (which can
                # happen with some DFT+U output) skip only the
                # eigenvalue/k-point data so that energies and forces are
                # still returned.
                if spin == 1 and len(eigenvalues[0]) != len(eigenvalues[1]):
                    warnings.warn(
                        'Inconsistent number of spin-up and spin-down '
                        'k-point eigenvalue blocks; skipping eigenvalue '
                        'parsing. Energies and forces are unaffected.')
                    continue

                if len(eigenvalues[0]) != len(ibzkpts):
                    warnings.warn(
                        'Number of eigenvalue blocks ({}) does not match '
                        'the number of k-points ({}); skipping eigenvalue '
                        'parsing. Energies and forces are unaffected.'
                        .format(len(eigenvalues[0]), len(ibzkpts)))
                    continue

                kpts = []
                for s in range(spin + 1):
                    for w, k, e in zip(weights, ibzkpts, eigenvalues[s]):
                        kpt = SinglePointKPoint(w, s, k, eps_n=e)
                        kpts.append(kpt)
'''

# Match the existing "# Bands" block: from the "# Bands" comment up to and
# including the final "kpts.append(kpt)" line. DOTALL lets .*? span the
# (whitespace-messy) middle so we don't depend on exact inner formatting.
BLOCK_RE = re.compile(
    r"        # Bands\n.*?                        kpts\.append\(kpt\)\n",
    re.DOTALL,
)


def find_target():
    try:
        import ase.io.espresso as mod
    except Exception as exc:  # pragma: no cover
        sys.exit("ERROR: could not import ase.io.espresso: %r\n"
                 "Run this with the same Python/environment you use for ASE."
                 % exc)
    import ase
    print("ASE version : %s" % getattr(ase, "__version__", "unknown"))
    print("Target file : %s" % mod.__file__)
    return mod.__file__


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change without writing")
    args = ap.parse_args()

    path = find_target()
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    if "elif L[0] == 'enter' and 'write_ns' in L:" in content:
        print("Already patched -- nothing to do.")
        return

    matches = BLOCK_RE.findall(content)
    if len(matches) != 1:
        sys.exit(
            "ERROR: expected exactly 1 '# Bands' block, found %d.\n"
            "Your espresso.py differs from the expected layout; patch "
            "aborted so nothing is damaged. Apply the change by hand "
            "(see the block printed below).\n\n%s"
            % (len(matches), NEW_BLOCK))

    new_content = BLOCK_RE.sub(lambda m: NEW_BLOCK, content, count=1)

    if args.dry_run:
        print("\n--- dry run: the '# Bands' block would be replaced with: ---\n")
        print(NEW_BLOCK)
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = "%s.bak-%s" % (path, stamp)
    shutil.copy2(path, backup)
    print("Backup      : %s" % backup)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_content)

    # Sanity check: the patched file must still compile.
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, path)
        sys.exit("ERROR: patched file failed to compile, reverted from "
                 "backup.\n%s" % exc)

    print("Patched OK. Re-run your ase.io.read(..., format='espresso-out').")


if __name__ == "__main__":
    main()
