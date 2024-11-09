"""Macromolecular structure feature types for compatibility with HF datasets.

Features are decoded into biotite atom arrays.
"""
import functools
import gzip
import logging
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
from biotite import structure as bs
from biotite.structure.filter import filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from biotite.structure.residues import get_residue_starts
from datasets import Array1D, Array2D, config
from datasets.download import DownloadConfig
from datasets.features.features import Value, get_nested_type
from datasets.table import array_cast, cast_array_to_feature
from datasets.utils.file_utils import is_local_path, xopen, xsplitext
from datasets.utils.py_utils import no_op_if_value_is_null, string_to_dict

from bio_datasets import config as bio_config
from bio_datasets.structure import Biomolecule, BiomoleculeChain, BiomoleculeComplex
from bio_datasets.structure.biomolecule import (
    create_complete_atom_array_from_restype_index,
)
from bio_datasets.structure.parsing import load_structure
from bio_datasets.structure.protein import (
    ProteinChain,
    ProteinComplex,
    ProteinDictionary,
    ProteinMixin,
)
from bio_datasets.structure.protein import constants as protein_constants
from bio_datasets.structure.residue import ResidueDictionary, get_residue_starts_mask

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

from .features import CustomFeature, register_bio_feature

logger = logging.getLogger(__name__)

extra_annots = [
    "b_factor",
    "occupancy",
    "charge",
    "atom_id",
]


def element_from_atom_name(atom_name: np.ndarray, molecule_type: np.ndarray):
    # TODO: write a vectorised version
    # I think there is actually ambiguity here - CA can be alpha carbon or Calcium...
    raise NotImplementedError()


def protein_atom_array_from_dict(
    d: Dict, backbone_atoms: Optional[List[str]] = None
) -> bs.AtomArray:
    backbone_atoms = backbone_atoms or ["N", "CA", "C", "O"]
    sequence = d["sequence"]
    annots_keys = [k for k in d.keys() if k in extra_annots]
    swaps = {
        "U": "C",
        "O": "K",
        "B": "X",
        "J": "X",
        "Z": "X",
    }
    if "backbone_coords" in d:
        backbone_coords = d["backbone_coords"]
        assert len(sequence) == len(d["backbone_coords"]["N"])
        atoms = []
        for res_ix, aa in enumerate(sequence):
            # TODO: better support for non-standard amino acids
            aa = swaps.get(aa, aa)
            res_name = protein_constants.restype_1to3[aa]
            for atom_name in backbone_atoms:
                annots = {}
                for k in annots_keys:
                    annots[k] = d[k][res_ix]
                atom = bs.Atom(
                    coord=backbone_coords[atom_name][res_ix],
                    chain_id="A",
                    res_id=res_ix + 1,
                    res_name=res_name,
                    hetero=False,
                    atom_name=atom_name,
                    element=atom_name[0],  # for protein backbone atoms this is correct
                    **annots,
                )
                atoms.append(atom)
        arr = bs.array(atoms)
        return arr
    elif "atom37_coords" in d:
        raise NotImplementedError("Atom37 not supported yet")
    else:
        raise ValueError("No coordinates found")


def encode_biotite_atom_array(
    array: bs.AtomArray, encode_with_foldcomp: bool = False, name: Optional[str] = None
) -> str:
    """
    Encode a biotite AtomArray to pdb string bytes.

    TODO: support foldcomp encoding
    """
    pdbf = PDBFile()
    pdbf.set_structure(array)
    contents = "\n".join(pdbf.lines) + "\n"
    if encode_with_foldcomp:
        import foldcomp

        if name is None:
            name = getattr(array, "name", str(uuid.uuid4()))
        return foldcomp.compress(name, contents)
    else:
        return contents.encode()


def load_structure_from_file_dict(
    d: dict,
    token_per_repo_id: Optional[Dict[str, int]] = None,
    extra_fields: Optional[List[str]] = None,
) -> bs.AtomArray:
    token_per_repo_id = token_per_repo_id or {}

    path, bytes_ = d.get("path"), d.get("bytes")
    file_type = d["type"]
    assert file_type is not None

    if bytes_ is None:
        assert path is not None, "path is required when bytes is None"
        return _load_from_path(path, file_type, extra_fields, token_per_repo_id, d)
    else:
        return _load_from_bytes(bytes_, file_type, extra_fields)


def _load_from_path(
    path: Optional[str],
    file_type: str,
    extra_fields: Optional[List[str]],
    token_per_repo_id: Dict[str, int],
) -> bs.AtomArray:

    if is_local_path(path):
        return load_structure(
            path, file_type=file_type.replace(".gz", ""), extra_fields=extra_fields
        )

    source_url = path.split("::")[-1]
    pattern = (
        config.HUB_DATASETS_URL
        if source_url.startswith(config.HF_ENDPOINT)
        else config.HUB_DATASETS_HFFS_URL
    )
    try:
        repo_id = string_to_dict(source_url, pattern)["repo_id"]
        token = token_per_repo_id.get(repo_id)
    except ValueError:
        token = None

    download_config = DownloadConfig(token=token)
    with xopen(path, "r", download_config=download_config) as f:
        return load_structure(f, file_type=file_type, extra_fields=extra_fields)


def _load_from_bytes(
    bytes_: bytes,
    file_type: str,
    extra_fields: Optional[List[str]],
) -> bs.AtomArray:
    fhandler = _file_handler_from_bytes(bytes_, file_type)
    return load_structure(
        fhandler, file_type=file_type.replace(".gz", ""), extra_fields=extra_fields
    )


def _file_handler_from_bytes(bytes_: bytes, file_type: Optional[str]):
    if file_type in ["fcz", "bcif"]:
        return BytesIO(bytes_)
    elif file_type.endswith(".gz"):
        decompressed = gzip.decompress(bytes_)
        return _file_handler_from_bytes(decompressed, file_type[:-3])
    elif file_type in ["pdb", "cif"]:
        return StringIO(bytes_.decode())
    else:
        raise ValueError(f"Unsupported file type: {file_type} for bytes input")


@dataclass
class AtomArrayFeature(CustomFeature):
    """
    AtomArray [`Feature`] to read macromolecular atomic structure data from a PDB or CIF file.

    This feature stores the array directly as a pa struct (basically a dictionary of arrays),
    as defined in the AtomArrayExtensionType.

    Input: The AtomArrayFeature feature accepts as (encodeable) input (Q. where would 'input' typically occur):
    - A `biotite.structure.AtomArray` object.
    - TODO: a Biopython structure object
    - TODO: a file handler or file contents string?
    - A dictionary with the required keys:
        - sequence
        - backbone_coords: a dictionary with keys:
            - N
            - CA
            - C
            - O
        - atom37_coords: an array of shape (N, 37, 3)
        - atom37_mask: a boolean array of shape (N, 37)
        Only backbone_coords or atom37_coords + mask need to be provided, not both.
        All other keys are optional, but must correspond to fields in the AtomArrayExtensionType:
        - chain_id
        - res_id
        - ins_code
        - res_name
        - hetero
        - atom_name
        - box
        - bonds
        - occupancy
        - b_factor
        - atom_id
        - charge
        - element
    """

    residue_dictionary: Optional[Union[ResidueDictionary, Dict]] = None
    backbone_only: ClassVar[bool] = False
    requires_encoding: bool = True
    requires_decoding: bool = True
    all_atoms_present: bool = (
        False  # when all atoms are present, we dont need to store atom name
    )
    decode: bool = True
    load_as: str = "biotite"  # biomolecule or chain or complex or biotite; if chain must be monomer
    constructor_kwargs: Optional[Dict] = None
    coords_dtype: str = "float32"
    b_factor_is_plddt: bool = False
    b_factor_dtype: str = "float32"
    with_element: bool = True
    with_hetero: bool = True  # TODO: can be inferred from res_name I guess...
    with_box: bool = False
    with_bonds: bool = False
    with_occupancy: bool = False
    with_b_factor: bool = False
    with_res_id: bool = False  # can be inferred...
    with_atom_id: bool = False
    with_charge: bool = False
    with_ins_code: bool = False
    # Automatically constructed
    _type: str = field(
        default="AtomArrayFeature", init=False, repr=False
    )  # registered feature name

    def _make_features_dict(self):
        # TODO: maybe just don't ever store restype_index?
        if self.residue_dictionary is not None:
            residue_identifier = ("restype_index", Array1D((None,), "uint8"))
        else:
            residue_identifier = ("res_name", Array1D((None,), "string"))
        features = [
            ("coords", Array2D((None, 3), self.coords_dtype)),
            residue_identifier,
            (
                "chain_id",
                Array1D((None,), "string"),
            ),  # TODO: could make Value(string) if load_as == "chain"
        ]
        if not self.all_atoms_present:
            features.append(("atom_name", Array1D((None,), "string")))
            features.append(("residue_starts", Array1D((None,), "uint32")))
        if self.with_res_id:
            features.append(("res_id", Array1D((None,), "uint32")))
        if self.with_hetero:
            features.append(("hetero", Array1D((None,), "bool")))
        if self.with_ins_code:
            features.append(("ins_code", Array1D((None,), "string")))
        if self.with_box:
            features.append(("box", Array2D((3, 3), "float32")))
        if self.with_bonds:
            features.append(("bond_edges", Array2D((None, 2), "uint16")))
            features.append(("bond_types", Array1D((None,), "uint8")))
        if self.with_occupancy:
            features.append(("occupancy", Array1D((None,), "float16")))
        if self.with_b_factor:
            # TODO: maybe have specific storage format for plddt bfactor (fixed range)
            features.append(("b_factor", Array1D((None,), self.b_factor_dtype)))
        if self.with_charge:
            features.append(("charge", Array1D((None,), "int8")))
        if self.with_element:
            features.append(("element", Array1D((None,), "string")))
        return OrderedDict(
            features
        )  # order may not be important due to Features.recursive_reorder

    def __post_init__(self):
        # init the StructFeature - since it inherits from dict, pa type inference is automatic (via get_nested_type)
        if self.all_atoms_present:
            assert (
                self.residue_dictionary is not None
            ), "residue_dictionary is required when all_atoms_present is True"
        self.deserialize()
        self._features = self._make_features_dict()
        if not self.with_element and not self.all_atoms_present:
            # TODO: support element inference
            raise ValueError("with_element must be True if all_atoms_present is False")

    def __call__(self):
        return get_nested_type(self._features)

    def fallback_feature(self):
        return self._features

    def deserialize(self):
        if isinstance(self.residue_dictionary, dict):
            self.residue_dictionary = ResidueDictionary(**self.residue_dictionary)

    @property
    def required_keys(self):
        required_keys = ["coords", "atom_name", "res_name", "chain_id"]
        if self.with_box:
            required_keys.append("box")
        if self.with_bonds:
            required_keys.append("bonds")
        return required_keys

    @property
    def extra_fields(self):
        # values that can be passed to biotite load_structure
        extra_fields = []
        if self.with_occupancy:
            extra_fields.append("occupancy")
        if self.with_b_factor:
            extra_fields.append("b_factor")
        if self.with_atom_id:
            extra_fields.append("atom_id")
        if self.with_charge:
            extra_fields.append("charge")
        return extra_fields

    def cast_storage(self, array: pa.StructArray) -> pa.StructArray:
        null_mask = array.is_null()
        if null_mask.sum() == len(null_mask):
            null_array = pa.array([None] * len(array))
            arrays = [
                cast_array_to_feature(null_array, subfeature)
                for _, subfeature in self._features.items()
            ]
        else:
            array_fields = {field.name for field in array.type}
            # c.f. cast_array_to_feature: since we don't inherit from dict, we reproduce the logic here
            arrays = [
                cast_array_to_feature(
                    array.field(name) if name in array_fields else null_array,
                    subfeature,
                )
                for name, subfeature in self._features.items()
            ]
        return pa.StructArray.from_arrays(
            arrays, names=list(self._features), mask=null_mask
        )

    def _encode_example(
        self,
        value: Union[bs.AtomArray, Dict, Biomolecule, os.PathLike, bytes],
        is_standardised: bool = False,  # if encoding a standardised Biomolecule, avoid re-standardising
    ) -> dict:
        if isinstance(value, Biomolecule):
            return self._encode_example(
                value.atoms, is_standardised=value.is_standardised
            )
        if isinstance(value, dict):
            return self._encode_dict(value)
        elif isinstance(value, bs.AtomArray):
            return self._encode_atom_array(value, is_standardised)
        elif isinstance(value, (str, os.PathLike)):
            return self._encode_path(value)
        elif isinstance(value, bytes):
            return self._encode_bytes(value)
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def _encode_dict(self, value: Dict) -> dict:
        if "bytes" in value or "path" in value or "type" in value:
            struct = load_structure_from_file_dict(
                value, extra_fields=self.extra_fields
            )
            return self._encode_example(struct)
        if all(attr in value for attr in self.required_keys):
            return value
        else:
            raise ValueError("Expected keys bytes/path/type in dict")

    def _encode_atom_array(self, value: bs.AtomArray, is_standardised: bool) -> dict:
        if self.all_atoms_present and not is_standardised:
            assert self.residue_dictionary is not None
            value = Biomolecule.standardise_atoms(value, self.residue_dictionary)
        if self.load_as == "chain":
            chain_ids = np.unique(value.chain_id)
            assert (
                len(chain_ids) == 1
            ), "Only single chain supported when `load_as` == 'chain'"
        residue_starts = get_residue_starts(value)
        if len(residue_starts) > 65535:
            raise ValueError("AtomArray too large to fit in uint16 (residue starts)")
        return self._build_atom_array_struct(value, residue_starts)

    def _build_atom_array_struct(
        self, value: bs.AtomArray, residue_starts: np.ndarray
    ) -> dict:
        atom_array_struct = {"coords": value.coord}
        if self.residue_dictionary is not None:
            atom_array_struct[
                "restype_index"
            ] = self.residue_dictionary.res_name_to_index(
                value.res_name[residue_starts]
            )
        else:
            atom_array_struct["res_name"] = value.res_name[residue_starts]
        if not self.all_atoms_present:
            atom_array_struct["residue_starts"] = residue_starts
            atom_array_struct["atom_name"] = value.atom_name
        atom_array_struct["chain_id"] = value.chain_id[residue_starts]
        self._add_optional_attributes(atom_array_struct, value, residue_starts)
        return atom_array_struct

    def _add_optional_attributes(
        self, atom_array_struct: dict, value: bs.AtomArray, residue_starts: np.ndarray
    ):
        for attr in [
            "box",
            "occupancy",
            "b_factor",
            "atom_id",
            "charge",
            "element",
            "res_id",
            "ins_code",
            "hetero",
        ]:
            if getattr(self, f"with_{attr}"):
                if (attr == "b_factor" and self.b_factor_is_plddt) or attr == "res_id":
                    atom_array_struct[attr] = getattr(value, attr)[residue_starts]
                else:
                    atom_array_struct[attr] = getattr(value, attr)
        if self.with_bonds:
            bonds_array = value.bond_list.as_array()
            assert bonds_array.ndim == 2
            assert bonds_array.shape[1] == 3
            atom_array_struct["bond_edges"] = bonds_array[:, :2]
            atom_array_struct["bond_types"] = bonds_array[:, 2]

    def _encode_path(self, value: Union[str, os.PathLike]) -> dict:
        if os.path.exists(value):
            file_type = xsplitext(value)[1][1:].lower()
            return self._encode_example(
                load_structure(value, format=file_type, extra_fields=self.extra_fields)
            )
        raise ValueError(f"Path does not exist: {value}")

    def _encode_bytes(self, value: bytes) -> dict:
        raise NotImplementedError("Not implemented")
        # file_type = infer_bytes_format(value)
        # fhandler = BytesIO(value)
        # return self.encode_example(
        #     load_structure(fhandler, format=file_type, extra_fields=self.extra_fields)
        # )

    def _decode_atoms(self, value, token_per_repo_id=None):
        if not isinstance(value["coords"], (np.ndarray, list)):
            return None

        num_atoms = len(value["coords"])
        if self.all_atoms_present:
            atoms, residue_index = self._decode_complete_atoms(value)
        else:
            atoms, residue_index = self._decode_partial_atoms(value, num_atoms)

        if self.b_factor_is_plddt and "b_factor" in value:
            atoms.set_annotation("b_factor", value.pop("b_factor")[residue_index])
        atoms.coord = value.pop("coords")
        if "bond_edges" in value:
            bonds_array = value.pop("bond_edges")
            bond_types = value.pop("bond_types")
            bonds_array = np.concatenate([bonds_array, bond_types[:, None]], axis=1)
            bonds = bs.BondList(num_atoms, bonds_array)
            atoms.bond_list = bonds

        for key, val in value.items():
            atoms.set_annotation(key, val)
        return atoms

    def _decode_complete_atoms(self, value):
        restype_index = value.pop("restype_index")
        chain_id = value.pop("chain_id")
        atoms, residue_starts, _ = create_complete_atom_array_from_restype_index(
            restype_index,
            residue_dictionary=self.residue_dictionary,
            chain_id=chain_id,
            backbone_only=self.backbone_only,
        )
        residue_index = np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1
        return atoms, residue_index

    def _decode_partial_atoms(self, value, num_atoms):
        atoms = bs.AtomArray(num_atoms)
        residue_starts = value.pop("residue_starts")
        residue_index = np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1

        self._set_residue_annotations(value, atoms, residue_index)

        return atoms, residue_index

    def _set_residue_annotations(self, value, atoms, residue_index):
        if "res_id" in value:
            atoms.set_annotation("res_id", value.pop("res_id")[residue_index])
        else:
            atoms.set_annotation("res_id", residue_index + 1)

        if self.residue_dictionary is not None:
            atoms.set_annotation(
                "restype_index", value.pop("restype_index")[residue_index]
            )
            atoms.set_annotation(
                "res_name",
                np.array(self.residue_dictionary.residue_names)[atoms.restype_index],
            )
        else:
            atoms.set_annotation("res_name", value.pop("res_name")[residue_index])

        atoms.set_annotation("atom_name", value.pop("atom_name"))

        if "chain_id" in value:
            atoms.set_annotation("chain_id", value.pop("chain_id")[residue_index])
        elif self.chain_id is not None:
            atoms.set_annotation("chain_id", np.full(len(atoms), self.chain_id))
        if self.with_element:
            atoms.set_annotation("element", value.pop("element"))
        else:
            raise ValueError("with_element must be True if all_atoms_present is False")

    def _decode_example(
        self, value: dict, token_per_repo_id=None
    ) -> Union["bs.AtomArray", None]:
        atoms = self._decode_atoms(value, token_per_repo_id=token_per_repo_id)

        constructor_kwargs = self.constructor_kwargs or {}
        if self.load_as == "biotite":
            return atoms
        elif self.load_as == "biomolecule":
            residue_dict = self.residue_dictionary or ResidueDictionary.from_ccd_dict()
            return Biomolecule(atoms, residue_dict, **constructor_kwargs)
        elif self.load_as == "chain":
            return BiomoleculeChain(atoms, residue_dict, **constructor_kwargs)
        elif self.load_as == "complex":
            return BiomoleculeComplex.from_atoms(
                atoms, residue_dictionary=self.residue_dictionary, **constructor_kwargs
            )
        else:
            raise ValueError(f"Unsupported load_as: {self.load_as}")


def file_type_from_path(path: str) -> str:
    if os.path.splitext(path)[1] == ".gz":
        base = os.path.splitext(os.path.splitext(path)[0])[1][1:]
        return base + ".gz"
    else:
        return os.path.splitext(path)[1][1:]


@dataclass
class StructureFeature(CustomFeature):
    """Structure [`Feature`] to read (bio)molecular atomic structure data from supported file types.
    The file contents are serialized as bytes, file path and file type within an Arrow table.
    The file contents are automatically decoded to a biotite AtomArray (if mode=="array") or a
    Biopython structure (if mode=="structure") when loading data from the dataset.

    This is similar to the Image/Audio features in the HF datasets library.

    - AtomArray documentation: https://www.biotite-python.org/latest/apidoc/biotite.structure.AtomArray.html#biotite.structure.AtomArray
    - Structure documentation: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ#the-structure-object

    Input: The StructureFeature accepts as (encodeable) input (e.g. as structure values in the outputs of dataset_builder.generate_examples()):
    - A `str`: Absolute path to the structure file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the structure file to the archive file.
        - `bytes`: Bytes of the structure file.

      This is useful for archived files with sequential access.
    - A `biotite.structure.AtomArray` object.
    - TODO: a Biopython structure object
    - TODO: a file handler or file contents string?

    N.B. foldcomp only supports monomer protein chains - should we somehow enforce this?

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the structure data. If `False`,
            returns the underlying dictionary in the format `{"path": structure_path, "bytes": structure_bytes, "type": structure_type}`.
    """

    requires_encoding: bool = True
    requires_decoding: bool = True
    decode: bool = True
    load_as: str = "biotite"  # biomolecule or chain or complex or biotite; if chain must be monomer
    constructor_kwargs: dict = None
    with_occupancy: bool = False
    with_b_factor: bool = False
    with_atom_id: bool = False
    with_charge: bool = False
    encode_with_foldcomp: bool = False
    compression: Optional[str] = None  # "gzip" or "foldcomp" or None
    pa_type: ClassVar[Any] = pa.struct(
        {"bytes": pa.binary(), "path": pa.string(), "type": pa.string()}
    )
    _type: str = field(default="StructureFeature", init=False, repr=False)

    def __post_init__(self):
        if self.load_as in ["chain", "complex"]:
            assert "residue_dictionary" in (
                self.constructor_kwargs or {}
            ), "residue_dictionary must be provided if load_as is chain or complex"

    def __call__(self):
        return self.pa_type

    def fallback_feature(self):
        return {
            "bytes": Value("binary"),
            "path": Value("string"),
            "type": Value("string"),
        }

    @property
    def extra_fields(self):
        # values that can be passed to biotite load_structure
        extra_fields = []
        if self.with_occupancy:
            extra_fields.append("occupancy")
        if self.with_b_factor:
            extra_fields.append("b_factor")
        if self.with_atom_id:
            extra_fields.append("atom_id")
        if self.with_charge:
            extra_fields.append("charge")
        return extra_fields

    def _encode_dict(self, value: dict) -> dict:
        if value.get("path") is not None and os.path.isfile(value["path"]):
            path = value["path"]
            file_type = value.get("type") or file_type_from_path(path)
            # we set "bytes": None to not duplicate the data if they're already available locally; embedding happens later
            # (this assumes invocation in what context?)
            return {"bytes": None, "path": path, "type": file_type}
        elif value.get("bytes") is not None:
            # store the Structure bytes, and path is optionally used to infer the Structure format using the file extension
            path = value.get("path")
            file_type = value.get("type")
            if file_type is None and path is not None:
                file_type = os.path.splitext(path)[1][1:].lower()
            if self.compression == "gzip":
                assert not file_type.endswith(
                    ".gz"
                ), "Gzipped files should not be compressed again"
                value["bytes"] = gzip.compress(value["bytes"])
                value["type"] = value["type"] + ".gz"
            elif self.compression == "foldcomp":
                assert (
                    file_type == "pdb"
                ), "foldcomp compression only supported for PDB files"
                value["bytes"] = foldcomp.compress(value["bytes"])
                value["type"] = "fcz"
            elif self.compression is not None:
                raise ValueError(f"Unsupported compression: {self.compression}")
            return {"bytes": value["bytes"], "path": path, "type": file_type}
        else:
            raise ValueError(
                f"A structure sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def _encode_example(self, value: Union[str, bytes, bs.AtomArray]) -> dict:
        """Encode example into a format for Arrow.

        Similar to the built-in Image/Audio features, we allow for file contents to be written
        to the dataset only when pushing to the Hub, to avoid duplicating locally stored files.

        This determines what gets written to the Arrow file.
        """
        if isinstance(value, str):
            return self._encode_dict({"path": value})
        elif isinstance(value, bytes):
            # just assume pdb format for now
            return self._encode_dict({"bytes": value})
        elif isinstance(value, bs.AtomArray):
            if self.load_as == "chain":
                chain_ids = np.unique(value.chain_id)
                assert (
                    len(chain_ids) == 1
                ), "Only single chain supported when `load_as` == 'chain'"
            return {
                "path": None,
                "bytes": encode_biotite_atom_array(
                    value,
                    encode_with_foldcomp=self.encode_with_foldcomp,
                ),
                "type": "pdb" if not self.encode_with_foldcomp else "fcz",
            }
        elif isinstance(value, dict):
            return self._encode_dict(value)
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def _decode_atoms(self, value: dict, token_per_repo_id=None):
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use Structure(decode=True) instead."
            )

        atoms = load_structure_from_file_dict(
            value, token_per_repo_id=token_per_repo_id, extra_fields=self.extra_fields
        )
        return atoms

    def _decode_example(
        self, value: dict, token_per_repo_id=None
    ) -> Union["bs.AtomArray", None]:
        """Decode example structure file into AtomArray data.

        Args:
            value (`str` or `dict`):
                A string with the absolute structure file path, a dictionary with
                keys:

                - `path`: String with absolute or relative structure file path.
                - `bytes`: The bytes of the structure file.
                - `type`: The type of the structure file (e.g. "pdb", "cif", "fcz")
                Must be not None.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                structure files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `biotite.AtomArray`
        """
        atoms = self._decode_atoms(value, token_per_repo_id=token_per_repo_id)
        constructor_kwargs = self.constructor_kwargs or {}
        if self.load_as == "biotite":
            return atoms
        if self.load_as == "biomolecule":
            return Biomolecule(atoms, **constructor_kwargs)
        elif self.load_as == "chain":
            return BiomoleculeChain(atoms, **constructor_kwargs)
        elif self.load_as == "complex":
            return BiomoleculeComplex.from_atoms(
                atoms,
                **constructor_kwargs,
            )
        else:
            raise ValueError(f"Unsupported load_as: {self.load_as}")

    def cast_storage(self, storage: pa.StructArray) -> pa.StructArray:
        if pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array, storage.field("type")],
                names=["bytes", "path", "type"],
                mask=storage.is_null(),
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage.type}")
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed the file contents into the Arrow table.

        Configured by the embed_external_files flag in Dataset.push_to_hub / DatasetsDict
        TODO: check this is working as expected
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            # could be gzipped or foldcomp compressed already
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            if self.compression == "gzip" and not path.endswith(".gz"):
                bytes_ = gzip.compress(bytes_)
            elif self.compression == "foldcomp" and not path.endswith(".fcz"):
                assert path.endswith(
                    ".pdb"
                ), "foldcomp compression only supported for PDB files"
                bytes_ = foldcomp.compress(bytes_)
            elif self.compression is not None:
                raise ValueError(f"Unsupported compression: {self.compression}")
            return bytes_

        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"])
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                os.path.basename(path) if path is not None else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        type_array = storage.field("type")
        storage = pa.StructArray.from_arrays(
            [bytes_array, path_array, type_array],
            ["bytes", "path", "type"],
            mask=bytes_array.is_null(),
        )
        return array_cast(storage, self.pa_type)


@dataclass
class ProteinStructureFeature(StructureFeature):
    """Protein-specific structure feature.

    Advantages of protein-specific features:
    - we can enforce absence of any non-protein atoms,
    - we can use protein-specific residue dictionaries,
    - we can use protein-specific storage / compression formats,
    - we can return a Protein-specific object.

    TODO: improve foldcomp support - e.g. auto-compression of PDB files.
    N.B. ignores load_as
    """

    load_as: str = "complex"  # biomolecule or chain or complex or biotite; if chain must be monomer
    _type: str = field(default="ProteinStructureFeature", init=False, repr=False)

    def __post_init__(self):
        # residue_dictionary will be set to default if not provided in constructor_kwargs
        if "residue_dictionary" not in (
            self.constructor_kwargs or {}
        ) and self.load_as in ["chain", "complex"]:
            logger.info(
                "No residue_dictionary provided for ProteinStructureFeature, default ProteinDictionary will be used to decode."
            )

    def encode_example(self, value: Union[ProteinMixin, dict, bs.AtomArray]) -> dict:
        if isinstance(value, bs.AtomArray):
            value = value[filter_amino_acids(value)]
            value = value[~np.isin(value.element, ["H", "D"])]
        return super().encode_example(value)

    def _decode_example(
        self, encoded: dict, token_per_repo_id=None
    ) -> Union["ProteinChain", "ProteinComplex", None]:
        atoms = self._decode_atoms(encoded, token_per_repo_id=token_per_repo_id)
        # TODO: filter amino acids in encode_example also where possible
        constructor_kwargs = self.constructor_kwargs or {}
        if self.load_as == "biotite":
            return atoms
        elif self.load_as == "biomolecule":
            raise ValueError(
                "Returning biomolecule for protein-specific feature not supported."
            )
        elif self.load_as == "chain":
            return ProteinChain(atoms, **constructor_kwargs)
        elif self.load_as == "complex":
            return ProteinComplex.from_atoms(atoms, **constructor_kwargs)
        else:
            raise ValueError(f"Unsupported load_as: {self.load_as}")


@dataclass
class ProteinAtomArrayFeature(AtomArrayFeature):

    """Decodes to a `bio_datasets.protein.Protein` or `bio_datasets.protein.ProteinComplex` object.

    Advantages of protein-specific features:
    - we can enforce absence of any non-protein atoms,
    - we can use protein-specific residue dictionaries,
    - we can use protein-specific storage / compression formats,
    - we can return a Protein-specific object.

    Assumes standard set of amino acids for now.

    These objects have standardised atoms (with nans for any missing atoms),
    and are guaranteed to contain no HETATMs or hydrogens.

    For generic storage of atom arrays without standardisation, see AtomArrayFeature
    """

    all_atoms_present: bool = False
    backbone_only: bool = False
    residue_dictionary: ProteinDictionary = field(
        default_factory=functools.partial(ProteinDictionary.from_preset, "protein")
    )
    load_as: str = "complex"  # biomolecule or chain or complex or biotite; if chain must be monomer
    internal_coords_type: str = None  # foldcomp, idealised, or pnerf
    _type: str = field(
        default="ProteinAtomArrayFeature", init=False, repr=False
    )  # registered feature name

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.residue_dictionary is not None
        ), "residue_dictionary must be provided"

    def deserialize(self):
        if isinstance(self.residue_dictionary, dict):
            self.residue_dictionary = ProteinDictionary(**self.residue_dictionary)
        elif self.all_atoms_present:
            assert isinstance(
                self.residue_dictionary, ProteinDictionary
            ), "residue_dictionary must be a ProteinDictionary"

    @classmethod
    def from_preset(cls, preset: str, **kwargs):
        if preset == "afdb":
            residue_dictionary = ProteinDictionary.from_preset("protein")
            return cls(
                residue_dictionary=residue_dictionary,
                with_b_factor=True,
                b_factor_is_plddt=True,
                # b_factor_dtype="uint8"
                b_factor_dtype="float16",
                coords_dtype="float16",
                all_atoms_present=True,
                with_element=False,
                with_hetero=False,
                **kwargs,
            )
        elif preset == "pdb":
            residue_dictionary = ProteinDictionary.from_preset("protein")
            return cls(
                residue_dictionary=residue_dictionary,
                with_b_factor=False,
                coords_dtype="float16",
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def _encode_example(
        self,
        value: Union[ProteinMixin, dict, bs.AtomArray],
    ) -> dict:
        if isinstance(value, dict) and "sequence" in value:
            value = protein_atom_array_from_dict(
                value,
                self.residue_dictionary.backbone_atoms
                if self.residue_dictionary
                else ["N", "CA", "C", "O"],
            )
            return self.encode_example(value)
        if isinstance(value, ProteinMixin):
            # TODO: switch to extracting backbone.
            if self.backbone_only:
                value = value.backbone()
            return super()._encode_example(
                value.atoms, is_standardised=value.is_standardised
            )
        if isinstance(value, bs.AtomArray):
            value = value[~np.isin(value.element, ["H", "D"])]
            value = value[filter_amino_acids(value)]
            if not self.residue_dictionary.keep_oxt:
                value = value[value.atom_name != "OXT"]
            if self.backbone_only:
                backbone_mask = np.isin(
                    value.atom_name, self.residue_dictionary.backbone_atoms
                )
                value = value[backbone_mask]
            return super()._encode_example(value)
        return super()._encode_example(value)

    def _decode_example(
        self, encoded: dict, token_per_repo_id=None
    ) -> Union["ProteinChain", "ProteinComplex", None]:
        atoms = self._decode_atoms(encoded, token_per_repo_id=token_per_repo_id)
        if atoms is None:
            return None
        constructor_kwargs = self.constructor_kwargs or {}
        if self.load_as == "biotite":
            return atoms
        elif self.load_as == "biomolecule":
            raise ValueError(
                "Returning biomolecule for protein-specific feature not supported."
            )
        elif self.load_as == "chain":
            return ProteinChain(
                atoms, residue_dictionary=self.residue_dictionary, **constructor_kwargs
            )
        elif self.load_as == "complex":
            return ProteinComplex.from_atoms(
                atoms, residue_dictionary=self.residue_dictionary, **constructor_kwargs
            )
        else:
            raise ValueError(f"Unsupported load_as: {self.load_as}")


register_bio_feature(StructureFeature)
register_bio_feature(AtomArrayFeature)
register_bio_feature(ProteinAtomArrayFeature)
register_bio_feature(ProteinStructureFeature)
