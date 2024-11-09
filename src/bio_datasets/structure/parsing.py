import gzip
import os
from os import PathLike
from typing import Optional

import numpy as np

from bio_datasets import config as bio_config

if bio_config.FASTPDB_AVAILABLE:
    import fastpdb

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

from biotite import structure as bs
from biotite.structure.filter import (
    filter_first_altloc,
    filter_highest_occupancy_altloc,
)
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx.convert import (
    _filter_model,
    _get_block,
    _get_model_starts,
)
from biotite.structure.residues import get_residue_starts

from .residue import (
    ResidueDictionary,
    create_complete_atom_array_from_restype_index,
    get_residue_starts_mask,
)

FILE_TYPE_TO_EXT = {
    "pdb": "pdb",
    "PDB": "pdb",
    "CIF": "cif",
    "cif": "cif",
    "bcif": "bcif",
    "FCZ": "fcz",
    "fcz": "fcz",
    "foldcomp": "fcz",
}


def is_open_compatible(file):
    return isinstance(file, (str, PathLike))


def fill_missing_polymer_chain_residues(
    chain_atoms, complete_res_ids, restype_index, residue_dict, chain_id
):
    """Fill in missing residues for a single polymer chain."""
    missing_res_mask = ~np.isin(complete_res_ids, chain_atoms.res_id)
    missing_atoms, _, _ = create_complete_atom_array_from_restype_index(
        restype_index[missing_res_mask],
        residue_dict,
        chain_id,
        res_id=complete_res_ids[missing_res_mask],
    )

    missing_atoms.set_annotation(
        "altloc_id", np.full(len(missing_atoms), ".").astype("str")
    )
    missing_atoms.set_annotation(
        "auth_chain_id",
        np.full(len(missing_atoms), chain_id).astype("str"),
    )
    missing_atoms.set_annotation(
        "auth_res_id", np.full(len(missing_atoms), -1).astype(int)
    )
    if "occupancy" in chain_atoms._annot:
        raise NotImplementedError("occupancy not supported yet")
    complete_atoms = chain_atoms + missing_atoms
    annots_to_concat = [
        "altloc_id",
        "auth_chain_id",
        "auth_res_id",
    ]
    for annot in annots_to_concat:
        complete_atoms.set_annotation(
            annot,
            np.concatenate([chain_atoms._annot[annot], missing_atoms._annot[annot]]),
        )

    residue_starts = get_residue_starts(complete_atoms)

    res_perm = np.argsort(complete_atoms.res_id[residue_starts])
    residue_sizes = np.diff(np.append(residue_starts, len(complete_atoms)))

    permuted_residue_starts = residue_starts[res_perm]
    permuted_residue_sizes = residue_sizes[res_perm]

    permuted_residue_starts_atom = np.repeat(
        permuted_residue_starts, permuted_residue_sizes
    )
    post_perm_res_changes = (
        permuted_residue_starts_atom[1:] != permuted_residue_starts_atom[:-1]
    )
    post_perm_residue_starts = np.concatenate(
        [[0], np.where(post_perm_res_changes)[0] + 1]
    )
    _post_perm_res_index = (
        np.cumsum(get_residue_starts_mask(complete_atoms, post_perm_residue_starts)) - 1
    )

    permuted_relative_atom_index = (
        np.arange(len(complete_atoms)) - post_perm_residue_starts[_post_perm_res_index]
    )

    atom_perm = permuted_residue_starts_atom + permuted_relative_atom_index

    complete_atoms = complete_atoms[atom_perm]
    return complete_atoms


def fill_missing_polymer_residues(
    structure, entity_poly_seq, poly_entity_ids, poly_chain_ids
):
    """Fill in missing residues for polymer entities."""
    processed_chain_atoms = []
    residue_dict = ResidueDictionary.from_ccd_dict()
    for entity_chain_ids, entity_id in zip(poly_chain_ids, poly_entity_ids):
        poly_seq_entity_mask = (
            entity_poly_seq["entity_id"].as_array(int, -1) == entity_id
        )
        if not poly_seq_entity_mask.any():
            for chain_id in entity_chain_ids.split(","):
                processed_chain_atoms.append(
                    structure[structure.auth_chain_id == chain_id]
                )
        else:
            complete_res_ids = entity_poly_seq["num"].as_array(int, -1)[
                poly_seq_entity_mask
            ]
            entity_res_name = entity_poly_seq["mon_id"].as_array(str)[
                poly_seq_entity_mask
            ]
            entity_restype_index = residue_dict.res_name_to_index(entity_res_name)

            for chain_id in entity_chain_ids.split(","):
                chain_atoms = structure[
                    (structure.auth_chain_id == chain_id)
                    & (structure.entity_id == entity_id)
                ]
                complete_atoms = fill_missing_polymer_chain_residues(
                    chain_atoms,
                    complete_res_ids,
                    entity_restype_index,
                    residue_dict,
                    chain_id,
                )
                complete_atoms.set_annotation(
                    "entity_id", np.full(len(complete_atoms), entity_id).astype(int)
                )

                processed_chain_atoms.append(complete_atoms)
    return processed_chain_atoms


def _fill_missing_residues(structure: bs.AtomArray, block):
    processed_chain_atoms = []
    entity_poly = block["entity_poly"]
    entity_poly_seq = block["entity_poly_seq"]
    poly_chain_ids = entity_poly["pdbx_strand_id"].as_array(str)
    poly_entity_ids = entity_poly["entity_id"].as_array(int, -1)
    entity_ids = block["entity"]["id"].as_array(int, -1)

    nonpoly_entity_mask = ~np.isin(entity_ids, poly_entity_ids)
    nonpoly_entity_ids = entity_ids[nonpoly_entity_mask]
    for entity_id in nonpoly_entity_ids:
        processed_chain_atoms.append(structure[structure.entity_id == entity_id])

    processed_chain_atoms += fill_missing_polymer_residues(
        structure, entity_poly_seq, poly_entity_ids, poly_chain_ids
    )

    filled_structure = sum(processed_chain_atoms, bs.AtomArray(length=0))
    for key in structure._annot.keys():
        if key not in filled_structure._annot:
            filled_structure.set_annotation(
                key,
                np.concatenate(
                    [chain_atoms._annot[key] for chain_atoms in processed_chain_atoms]
                ),
            )
    return filled_structure


def _load_cif_structure(
    fpath_or_handler,
    file_type,
    model=1,
    extra_fields=None,
    fill_missing_residues=False,
    altloc="first",
):
    """Load a structure from cif or binary cif format.

    Cif files contain canonical labelling of res id chain id etc.
    as well as 'auth' labelling, which is what is shown in the pdb file.

    Optionally fill in missing residues with nan coordinates and standard atom names,
    by cross-referencing the entity_poly_seq header with the atom_site information and
    the CCD dictionary.

    TODO: an alternative to standardising here would be standardising within standardise_atoms
    if an additional kwarg (some map from res id to label res id) is provided.

    This would then generalise to e.g. aligning to uniprot as well, which would be extremely nice.
    Would be good to write some generic residue mapping utilites to allow this.
    """
    # we use filter_altloc all to make it easier to get the chain id mapping
    if file_type == "cif":
        pdbxf = pdbx.CIFFile.read(fpath_or_handler)
    else:
        pdbxf = pdbx.BinaryCIFFile.read(fpath_or_handler)
    extra_fields = extra_fields or (["occupancy"] if altloc == "occupancy" else [])
    if "occupancy" not in extra_fields and altloc == "occupancy":
        extra_fields.append("occupancy")
    structure = pdbx.get_structure(
        pdbxf,
        model=model,
        extra_fields=extra_fields,
        use_author_fields=False,  # be careful with this...
        altloc="all",  # handle later so that atom site lines up
    )
    # auth_chain_id -> chain_id mapping from atom_site
    block = _get_block(pdbxf, None)
    atom_site = block["atom_site"]
    models = atom_site["pdbx_PDB_model_num"].as_array(np.int32)
    model_starts = _get_model_starts(models)
    atom_site = _filter_model(atom_site, model_starts, model)
    structure.set_annotation("auth_chain_id", atom_site["auth_asym_id"].as_array(str))
    structure.set_annotation("auth_res_id", atom_site["auth_seq_id"].as_array(int, -1))
    structure.set_annotation(
        "entity_id", atom_site["label_entity_id"].as_array(int, -1)
    )

    if not fill_missing_residues:
        filled_structure = structure
    else:
        filled_structure = _fill_missing_residues(structure, block)

    if altloc == "occupancy":
        return filled_structure[
            filter_highest_occupancy_altloc(
                filled_structure, filled_structure.altloc_id
            )
        ]
    elif altloc == "first":
        return filled_structure[
            filter_first_altloc(filled_structure, filled_structure.altloc_id)
        ]
    elif altloc == "all":
        return filled_structure
    else:
        raise ValueError(f"'{altloc}' is not a valid 'altloc' option")


def _load_pdb_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
):
    if bio_config.FASTPDB_AVAILABLE:
        pdbf = fastpdb.PDBFile.read(fpath_or_handler)
    else:
        pdbf = PDBFile.read(fpath_or_handler)
    structure = pdbf.get_structure(
        model=model,
        extra_fields=extra_fields,
    )
    return structure


def _load_foldcomp_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
):
    if not bio_config.FOLDCOMP_AVAILABLE:
        raise ImportError(
            "Foldcomp is not installed. Please install it with `pip install foldcomp`"
        )

    if is_open_compatible(fpath_or_handler):
        with open(fpath_or_handler, "rb") as fcz:
            fcz_binary = fcz.read()
    else:
        raise ValueError("Unsupported file type: expected path or bytes handler")
    (_, pdb_str) = foldcomp.decompress(fcz_binary)
    lines = pdb_str.splitlines()
    pdbf = PDBFile()
    pdbf.lines = lines
    structure = pdbf.get_structure(
        model=model,
        extra_fields=extra_fields,
    )
    return structure


def load_structure(
    fpath_or_handler,
    file_type: Optional[str] = None,
    model: int = 1,
    extra_fields=None,
    fill_missing_residues=False,
):
    """
    TODO: support foldcomp format, binary cif format
    TODO: support model choice / multiple models (multiple conformations)
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if isinstance(fpath_or_handler, (str, PathLike)) and fpath_or_handler.endswith(
        ".gz"
    ):
        file_type = (
            file_type or os.path.splitext(os.path.splitext(fpath_or_handler)[0])[1][1:]
        )
        # https://github.com/biotite-dev/biotite/issues/193
        with gzip.open(fpath_or_handler, "rt") as f:
            return load_structure(
                f,
                file_type=file_type,
                model=model,
                extra_fields=extra_fields,
                fill_missing_residues=fill_missing_residues,
            )

    if file_type is None and isinstance(fpath_or_handler, (str, PathLike)):
        file_type = os.path.splitext(fpath_or_handler)[1][1:]
    assert (
        file_type is not None
    ), "Format must be specified if fpath_or_handler is not a path"

    file_type = FILE_TYPE_TO_EXT[file_type]
    if fill_missing_residues:
        assert file_type in [
            "cif",
            "bcif",
        ], "Fill missing residues only supported for cif files"

    if file_type in ["cif", "bcif"]:
        return _load_cif_structure(
            fpath_or_handler,
            file_type=file_type,
            model=model,
            extra_fields=extra_fields,
            fill_missing_residues=fill_missing_residues,
        )

    elif file_type == "pdb":
        return _load_pdb_structure(
            fpath_or_handler,
            model=model,
            extra_fields=extra_fields,
        )
    elif file_type == "fcz":
        return _load_foldcomp_structure(
            fpath_or_handler,
            model=model,
            extra_fields=extra_fields,
        )
    else:
        raise ValueError(f"Unsupported file format: {file_type}")
