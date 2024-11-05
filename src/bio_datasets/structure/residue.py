from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_ccd
from biotite.structure.io.pdbx import get_component
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices


def get_atom_elements():
    ccd_data = get_ccd()
    return np.unique(ccd_data["chem_comp_atom"]["type_symbol"].as_array())


ALL_ELEMENT_TYPES = get_atom_elements()


PROTEIN_TYPES = [
    "D-PEPTIDE LINKING",
    "D-PEPTIDE NH3 AMINO TERMINUS",
    "D-beta-peptide, C-gamma linking",
    "D-gamma-peptide, C-delta linking",
    "D-peptide NH3 amino terminus",
    "D-peptide linking",
    "D-saccharide, beta linking",
    "L-PEPTIDE COOH CARBOXY TERMINUS",
    "L-PEPTIDE LINKING",
    "L-beta-peptide, C-gamma linking",
    "L-gamma-peptide, C-delta linking",
    "L-peptide COOH carboxy terminus",
    "L-peptide NH3 amino terminus",
    "L-peptide linking",
    "PEPTIDE LINKING",
    "PEPTIDE-LIKE",
    "Peptide-like",
    "peptide linking",
    "peptide-like",
]


DNA_TYPES = [
    "DNA LINKING",
    "DNA OH 3 PRIME TERMINUS",
    "DNA OH 3 prime terminus",
    "DNA OH 5 prime terminus",
    "DNA linking",
    "L-DNA LINKING",
    "L-DNA linking",
]


RNA_TYPES = [
    "L-RNA LINKING",
    "L-RNA linking",
    "RNA LINKING",
    "RNA OH 3 prime terminus",
    "RNA OH 5 prime terminus",
    "RNA linking",
]


SACCHARIDE_TYPES = [
    "D-SACCHARIDE",
    "D-saccharide",
    "D-saccharide, alpha linking",
    "D-saccharide, beta linking",
    "L-SACCHARIDE",
    "L-saccharide",
    "L-saccharide, alpha linking",
    "L-saccharide, beta linking",
    "saccharide",
    "SACCHARIDE",
]


CHEMICAL_TYPES = ["NON-POLYMER", "non-polymer", "other"]


def get_component_types():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["type"].as_array()
    return {name: type for name, type in zip(res_names, res_types)}


CHEM_COMPONENT_TYPES = get_component_types()


# TODO: speed these up
def get_component_categories():
    categories = {}
    for name, chem_type in CHEM_COMPONENT_TYPES.items():
        if chem_type in PROTEIN_TYPES:
            categories[name] = "protein"
        elif chem_type in DNA_TYPES:
            categories[name] = "dna"
        elif chem_type in RNA_TYPES:
            categories[name] = "rna"
        elif chem_type in SACCHARIDE_TYPES:
            categories[name] = "saccharide"
        elif chem_type in CHEMICAL_TYPES:
            categories[name] = "chemical"
        else:
            raise ValueError(f"Unknown chemical component type: {chem_type}")
    return categories


def get_component_3to1():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["one_letter_code"].as_array()
    return {name: code for name, code in zip(res_names, res_types) if code}


CHEM_COMPONENT_CATEGORIES = get_component_categories()
CHEM_COMPONENT_3TO1 = get_component_3to1()


@dataclass
class ResidueDictionary:
    residue_names: List[str]
    residue_atoms: Dict[str, List]  # defines composition and atom order
    residue_elements: Dict[str, List[str]]
    unknown_residue_name: str
    # types define one-hot representations
    residue_types: Optional[List[str]] = None  # one letter codes
    element_types: Optional[List[str]] = None
    atom_types: Optional[List[str]] = None
    backbone_atoms: Optional[List[str]] = None
    conversions: Optional[List[Dict]] = None

    def __post_init__(self):
        assert len(self.residue_names) == len(self.residue_types)
        if self.conversions is not None:
            for conversion in self.conversions:
                # tuples get converted to lists during serialization so we need to convert them back for eq checks
                conversion["atom_swaps"] = [
                    tuple(swaps) for swaps in conversion["atom_swaps"]
                ]

    def __str__(self):
        return (
            f"ResidueDictionary ({len(self.residue_names)} residue types, "
            f"{len(self.atom_types)} atom types)"
        )

    @property
    def residue_sizes(self):
        return np.array(
            [len(self.residue_atoms[resname]) for resname in self.residue_names]
        )

    @property
    def relative_atom_indices_mapping(self) -> np.ndarray:
        """
        Get a mapping from atom type index to expected index relative to the start of a given residue.
        """
        all_atom_indices_mapping = []
        for resname in self.residue_names:
            if resname == self.unknown_residue_name:
                # n.b. in some structures, UNK also contains CB, CG, ...
                residue_atom_list = self.backbone_atoms
            else:
                residue_atom_list = self.residue_atoms[resname]
            atom_indices_mapping = []
            for atom in self.atom_types:
                if atom in residue_atom_list:
                    relative_index = residue_atom_list.index(atom)
                    atom_indices_mapping.append(relative_index)
                else:
                    atom_indices_mapping.append(-100)
            all_atom_indices_mapping.append(np.array(atom_indices_mapping))
        return np.stack(all_atom_indices_mapping, axis=0)

    @property
    def max_residue_size(self):
        return self.residue_sizes.max()

    @property
    def unique_atom_types(self):
        """How many atom names across all residues (37 for proteins).

        Typically should be equal to len(self.atom_types)
        """
        raise NotImplementedError()

    @property
    def total_element_types(self):
        """How many element types across all proteins"""
        raise NotImplementedError()

    @property
    def standard_atoms_by_residue(self):
        """Return a fixed size array of atom names for each residue type.

        Shape (num_residue_types x max_atoms_per_residue)
        e.g. for proteins we use atom14 (21 x 14)
        """
        arr = np.full((len(self.residue_names), self.max_residue_size), "", dtype="U6")
        for ix, residue_name in enumerate(self.residue_names):
            residue_atoms = self.residue_atoms[residue_name]
            arr[ix, : len(residue_atoms)] = residue_atoms
        return arr

    def get_residue_sizes(
        self, restype_index: np.ndarray, chain_id: np.ndarray
    ) -> np.ndarray:
        return self.residue_sizes[restype_index]

    def get_expected_relative_atom_indices(self, restype_index, atomtype_index):
        return self.relative_atom_indices_mapping[restype_index, atomtype_index]

    def get_atom_names(
        self,
        restype_index: np.ndarray,
        relative_atom_index: np.ndarray,
        chain_id: np.ndarray,
    ):
        return self.standard_atoms_by_residue[
            restype_index,
            relative_atom_index,
        ]

    def resname_to_index(self, resname: np.ndarray) -> np.ndarray:
        # n.b. protein resnames are sorted in alphabetical order, apart from UNK
        if not np.all(np.isin(resname, np.array(self.residue_names))):
            raise ValueError(
                f"resname contains elements not in the allowed list: "
                f"{np.unique(resname[~np.isin(resname, np.array(self.residue_names))])}"
            )
        return map_categories_to_indices(resname, self.residue_names)

    def restype_to_index(self, restype: np.ndarray) -> np.ndarray:
        if not np.all(np.isin(restype, np.array(self.residue_types))):
            raise ValueError(
                f"restype contains elements not in the allowed list: "
                f"{np.unique(restype[~np.isin(restype, np.array(self.residue_names))])}"
            )
        return map_categories_to_indices(restype, self.residue_types)

    def atomtype_index_full_to_short(self):
        # return a num_residues, num_full, num_short mapping array (e.g. atom37 -> atom14 for each residue)
        # raise NotImplementedError()
        return np.stack()

    def resname_to_onehot(self, resname: np.ndarray) -> np.ndarray:
        masks = [resname == r for r in self.residue_names]
        return np.stack(masks, axis=-1)

    def restype_to_onehot(self, restype: np.ndarray) -> np.ndarray:
        masks = [restype == r for r in self.residue_types]
        return np.stack(masks, axis=-1)

    def decode_restype_index(self, restype_index: np.ndarray) -> np.ndarray:
        return "".join(np.array(self.residue_types)[restype_index])

    def atom_full_to_atom_short(self):
        # eg atom37->atom14
        raise NotImplementedError()


def tile_residue_annotation_to_atoms(
    atoms: bs.AtomArray, residue_annotation: np.ndarray, residue_starts: np.ndarray
) -> np.ndarray:
    # use residue index as cumsum of residue starts
    assert len(residue_annotation) == len(residue_starts)
    residue_index = np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1
    return residue_annotation[residue_index]


def get_residue_starts_mask(
    atoms: bs.AtomArray, residue_starts: Optional[np.ndarray] = None
) -> np.ndarray:
    if residue_starts is None:
        residue_starts = get_residue_starts(atoms)
    mask = np.zeros(len(atoms), dtype=bool)
    mask[residue_starts] = True
    return mask


class ChemicalComponentDictionary:
    """
    Uses biotite's CCD.
    TODO: decide whether to save CCD or just use user's biotite CCD.

    N.B. most components don't have a one letter code.

    What is appropriate unknown residue name?
    """

    @classmethod
    def from_biotite_ccd(cls, category: str, keep_hydrogens: bool = False):
        assert category in [
            "protein",
            "dna",
            "rna",
            "saccharide",
            "chemical",
        ], f"Unknown category: {category}"
        ccd_data = get_ccd()
        res_names = np.unique(ccd_data["chem_comp_atom"]["comp_id"].as_array())
        mask = np.array(
            [CHEM_COMPONENT_CATEGORIES[name] == category for name in res_names]
        )
        res_names = list(res_names[mask])
        res_types = [CHEM_COMPONENT_3TO1[name] for name in res_names]
        res_atom_names = {}
        res_element_types = {}
        for name in res_names:
            atom_names = []
            element_types = []
            comp = get_component(ccd_data, res_name=name)
            for at, elem in zip(comp.atom_name, comp.element_symbol):
                if keep_hydrogens or (elem != "H" and elem != "D"):
                    atom_names.append(at)
                    element_types.append(elem)

            res_atom_names[name] = atom_names
            res_element_types[name] = element_types

        return cls(
            residue_names=res_names[mask],
            residue_types=res_types[mask],
            residue_atoms=res_atom_names,
            residue_elements=res_element_types,
            backbone_atoms=None,
            unknown_residue_name="UNK",
            element_types=ALL_ELEMENT_TYPES.copy(),
            atom_types=None,
            # atom_types=ALL_ATOM_TYPES.copy(),
        )
