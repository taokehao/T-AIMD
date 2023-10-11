# from pymatgen.core import Structure
# from matminer.featurizers.structure import GlobalSymmetryFeatures
# structure = Structure.from_file("./LiCaBO3.vasp")
# print(structure)
# sites = [site for site in structure.sites if site.specie.symbol == "Li"]
# print(sites)
# neighbor_num = [len(structure.get_neighbors(site, r=5.0)) for site in sites]
# print(neighbor_num)
# structure = Structure.from_file("./LiCaBO3.cif")
# print(structure)
# GSF = GlobalSymmetryFeatures()
# a = GSF.featurize(structure)
# space_group, crystal_system_int, is_centrosymmetric, n_symmetry_ops = GSF.featurize(structure)
# print(a)
from pymatgen.core.periodic_table import Element
print(Element('Li').average_ionic_radius)
