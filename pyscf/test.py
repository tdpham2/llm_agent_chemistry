import pyscf
from pyscf import gto, scf, grad

# Define the water molecule
# Geometry: O-H bond length = 0.96 Å, H-O-H angle = 104.5 degrees (in radians)
# The coordinates are given in Angstroms
mol = gto.Mole()
mol.atom = """
    O   0.000000   0.000000   0.000000
    H   0.000000   -0.757    0.586
    H   0.000000    0.757    0.586
"""
mol.basis = 'sto-3g'  # Define the basis set
mol.unit = 'Angstrom'  # Specify the unit of the coordinates
mol.build()

# Set up the Hartree-Fock calculation
mf = scf.RHF(mol)

# Perform the geometry optimization
# The optimization is performed using the PySCF Gradients module
from pyscf.geomopt.berny_solver import optimize

optimized_mol = optimize(mf)

# Print the optimized geometry
print("\nOptimized Geometry (Angstrom):")
for atom in optimized_mol.atom:
    print(atom)

# Print the total energy of the optimized geometry
print("\nTotal Energy (Hartree):", mf.e_tot)

