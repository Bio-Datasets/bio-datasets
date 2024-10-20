"""Script demonstrating how to build the Pinder dataset.

What we really want to upload for each protein is a single ProteinComplex feature,
and two sets of Protein features for each ligand and each receptor
(representing the unbound and - where available - predicted unbound states).
The chains should all be pre-aligned to the native complex.

To store the structures - we could even drop the missing coordinates and rely on the numbering
being the same.

We should also upload a raw version where we don't do any alignment or masking.

TODO: handle 'canonical' vs non-canonical apo conformation:
https://github.com/pinder-org/pinder/blob/9c70a92119b844d0d20e35f483b4f1f26b2899c4/src/pinder-core/pinder/core/index/system.py#L89

Q: what is difference between create_complex and create_masked_bound_unbound_complexes?

N.B. can use PinderSystem.download_entry to download files for single system.

Oligomeric state of the protein complex (homodimer, heterodimer, oligomer or higher-order complexes)
Structure determination method (X-Ray, CryoEM, NMR)
Resolution
Interfacial gaps, defined as structurally-unresolved segments on PPI interfaces
Number of distinct atom types. Many earlier Cryo-EM structures contain only a few atom-types such as only Cα or backbone atoms
Whether the interface is likely to be a physiological or crystal contact, annotated using Prodigy
Structural elongation, defined as the maximum variance of coordinates projected onto the largest principal component. This allows detection of long end-to-end stacked complexes, likely to be repetitive with small interfaces
Planarity, defined as deviation of interfacial Cα atoms from the fitted plane. This interface characteristic quantifies interfacial shape complementarity. Transient complexes have smaller and more planar interfaces than permanent and structural scaffold complexes
Number of components, defined as the number of connected components of a 10Å Cα radius graph. This allows detection of structurally discontinuous domains
Intermolecular contacts (labeled as polar or apolar)
"""
import argparse
import pathlib
from typing import List, Optional

import biotite.sequence.align as align
import numpy as np
from biotite import structure as bs
from biotite.structure.residues import get_residue_starts
from datasets import Array1D, Dataset, Features, Value
from pinder.core import PinderSystem, get_index, get_metadata
from pinder.core.index.utils import IndexEntry
from pinder.core.loader.structure import Structure
from pinder.core.structure import surgery
from pinder.core.structure.atoms import (
    _align_and_map_sequences,
    _get_seq_aligned_structures,
    _get_structure_and_res_info,
    apply_mask,
    get_seq_alignments,
    mask_from_res_list,
)

from bio_datasets import ProteinStructureFeature


def mask_structure(structure: Structure, mask: np.ndarray) -> Structure:
    """Apply mask to structure."""
    return Structure(
        filepath=structure.filepath,
        uniprot_map=structure.uniprot_map,
        pinder_id=structure.pinder_id,
        atom_array=apply_mask(structure.atom_array.copy(), mask),
    )


def align_sequence_to_ref(
    ref_seq: str,
    subject_seq: str,
    ref_numbering: list[int] | None = None,
    subject_numbering: list[int] | None = None,
) -> tuple[str, str, list[int], list[int]]:
    """Modified from pinder.core.structure.atoms.align_sequences, to retain all ref residues,
    so that only subject residues are dropped.

    Pinder might have chosen not to do this to avoid having to represent insertions.

    Aligns two sequences and returns a tuple containing the aligned sequences
    and their respective numbering.

    Parameters
    ----------
    ref_seq : (str)
        The reference sequence to align to.
    subject_seq : (str)
        The subject sequence to be aligned.
    ref_numbering : (list[int], optional)
        List of residue numbers for the reference sequence.
    subject_numbering : (list[int], optional)
        List of residue numbers for the subject sequence.

    Returns:
        tuple: A tuple containing:
            - Aligned reference sequence (str)
            - Aligned subject sequence (str)
            - Numbering of the aligned reference sequence (list[int])
            - Numbering of the aligned subject sequence (list[int])

    Raises
    ------
        ValueError if the sequences cannot be aligned.
    """
    alignments = get_seq_alignments(ref_seq, subject_seq)
    aln = alignments[0]
    s = align.alignment.get_symbols(aln)

    if ref_numbering is None:
        ref_numbering = list(range(1, len(ref_seq) + 1))

    if subject_numbering is None:
        subject_numbering = list(range(1, len(subject_seq) + 1))

    # assigning reference numbering to all residues in a given sequence
    # (e.g. PDB chain)
    # p = subject protein sequence, u = ref sequence
    ref_numbering_mapped = []
    subject_numbering_mapped = []
    subject_sequence_mapped = ""
    ref_sequence_mapped = ""
    ui = -1
    pj = -1
    for p, u in zip(s[0], s[1]):
        if u:
            ui += 1
        if p:
            pj += 1
        if u:
            # ref_numbering_mapped.append(ui + 1)  # 1-based
            ref_numbering_mapped.append(ref_numbering[ui])
            subject_numbering_mapped.append(subject_numbering[pj])
            ref_sequence_mapped += u
            subject_sequence_mapped += p
    return (
        ref_sequence_mapped,
        subject_sequence_mapped,
        ref_numbering_mapped,
        subject_numbering_mapped,
    )


def get_subject_positions_in_ref_masks(
    ref_at,
    target_at,
    pdb_engine: str = "fastpdb",
):
    """Whereas _get_seq_aligned_structures computes masks for the parts
    of the two sequences that are mutually alignable, this returns the positions
    of the subject sequence in the reference sequence, allowing us to map subject
    coords onto the reference by doing ref_coords[subject_mask_in_ref], subj_coords[subj_mask].
    """
    ref_info = _get_structure_and_res_info(ref_at, pdb_engine)
    subj_info = _get_structure_and_res_info(target_at, pdb_engine)

    # Aligns two sequences and maps the numbering
    # align_sequences aligns a subject to a reference
    # then returns the alignable parts of the sequences together with their numbering
    subj_resid_map, alns = _align_and_map_sequences(ref_info, subj_info)

    # Need to remove ref residues in order to match subject
    ref_structure, _, _ = ref_info
    subj_structure, _, _ = subj_info

    assert isinstance(ref_structure, (bs.AtomArray, bs.AtomArrayStack))
    assert isinstance(subj_structure, (bs.AtomArray, bs.AtomArrayStack))

    # values are positions in ref that subject aligns to
    subj_mask_in_ref = mask_from_res_list(ref_structure, list(subj_resid_map.values()))
    subj_mask = mask_from_res_list(subj_structure, list(subj_resid_map.keys()))
    return subj_mask_in_ref, subj_mask


class PinderDataset:
    def __init__(
        self,
        index,
        metadata,
        download: bool = False,
        dataset_path: Optional[str] = None,
        remove_differing_atoms: bool = True,
    ):
        self.index = index
        self.metadata = metadata
        self.download = download
        self.dataset_path = (
            pathlib.Path(dataset_path) if dataset_path is not None else None
        )
        self.remove_differing_atoms = remove_differing_atoms

    def get_aligned_structures(
        self,
        ref_struct,
        target_struct,
        mode: str = "ref",  # "ref" or "intersection"
    ):
        """TODO: test that we get same result as applying get_seq_aligned_structure

        Source: https://github.com/pinder-org/pinder/blob/8ad1ead7a174736635c13fa7266d9ca54cf9f44e/src/pinder-core/pinder/core/loader/structure.py#L146
        """
        # N.B. pinder utils have stuff for handling multi-chain cases, so we need to assert that these are single-chain structures.
        ref_chains = bs.get_chains(ref_struct.atom_array)
        target_chains = bs.get_chains(target_struct.atom_array)
        assert len(set(target_chains)) == 1
        assert len(set(ref_chains)) == 1

        ref_at = ref_struct.atom_array.copy()
        target_at = target_struct.atom_array.copy()

        if mode == "ref":
            # We drop any ref residues that aren't present in the target.
            subj_mask_in_ref, subj_mask = get_subject_positions_in_ref_masks(
                ref_at, target_at
            )

            # the below also automatically handles renumbering.
            aligned_target_at = ref_at.copy()
            # We'll assume that the sequence is the same at positions that don't align, so only coords need to be masked
            aligned_target_at[~subj_mask_in_ref].coord = np.nan
            if "b_factor" in ref_at._annot:
                aligned_target_at[~subj_mask_in_ref].b_factor = np.nan
            aligned_target_at[subj_mask_in_ref].coord = target_at[subj_mask].coord
            target_at = aligned_target_at

        elif mode == "intersection":
            # handles masking and renumbering
            ref_at, target_at = _get_seq_aligned_structures(ref_at, target_at)

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # make copies so we don't modify the original
        ref_struct = Structure(
            filepath=ref_struct.filepath,
            uniprot_map=ref_struct.uniprot_map,
            pinder_id=ref_struct.pinder_id,
            atom_array=ref_at,
            pdb_engine=ref_struct.pdb_engine,
        )
        target_struct = Structure(
            filepath=target_struct.filepath,
            uniprot_map=target_struct.uniprot_map,
            pinder_id=target_struct.pinder_id,
            atom_array=target_at,
            pdb_engine=target_struct.pdb_engine,
        )

        return ref_struct, target_struct

    def _make_atoms_consistent(self, ref_struct, *structs):
        # An atom is defined as existent in the second array, if there is an
        # atom in the second array that has the same annotation values in all
        # categories that exists in both arrays.

        # So pinder handles this by first modifying annotations to match,
        # then applying the atom mask, then optionally preserving the modified annotations.

        # It's clear that we don't need to preserve matched b_factors, for example.
        # although this does require care when applying filter_intersection downstream.

        # Even if atom counts are identical, annotation categories must be the same
        # First modify annotation arrays to use struc.filter_intersection,
        # then filter original structure with annotations to match res_id, res_name, atom_name
        # of intersecting structure
        ref_at_mod = ref_struct.atom_array.copy()
        target_at_mods = []
        ref_target_mask = np.ones(len(ref_at_mod), dtype=bool)
        for struct in structs:
            target_at_mod = struct.atom_array.copy()
            ref_target_mask &= bs.filter_intersection(ref_at_mod, target_at_mod)
            target_at_mods.append(target_at_mod)
            # just get rid of any annotations that don't match
            ref_at_mod, target_at_mod = surgery.fix_annotation_mismatch(
                ref_at_mod, target_at_mod, ["element", "ins_code", "b_factor"]
            )

        ref_struct.atom_array = ref_struct.atom_array[ref_target_mask]
        for target_at_mod, struct in zip(target_at_mods, structs):
            target_ref_mask = bs.filter_intersection(target_at_mod, ref_at_mod)
            struct.atom_array = struct.atom_array[target_ref_mask].copy()

    def make_structures(self, system: PinderSystem):
        """We have to choose which reference to align to: choices are complex, apo (unbound) or predicted (unbound).

        Bound makes most sense I think.
        """
        has_apo = system.entry.apo_R and system.entry.apo_L
        has_pred = system.entry.predicted_R and system.entry.predicted_L

        # apo_complex = system.create_apo_complex() - this superimposes structures, which gives info away about interaction
        # https://github.com/pinder-org/pinder/blob/8ad1ead7a174736635c13fa7266d9ca54cf9f44e/examples/pinder-system.ipynb
        if has_apo:
            apo_R, apo_L = system.apo_receptor, system.apo_ligand
        else:
            apo_R, apo_L = None, None
        if has_pred:
            pred_R, pred_L = system.pred_receptor, system.pred_ligand
        else:
            pred_R, pred_L = None, None

        if apo_R is not None:
            _, apo_R = self.get_aligned_structures(
                system.native_R,
                apo_R,
                mode="ref",
            )
            _, apo_L = self.get_aligned_structures(
                system.native_L,
                apo_L,
                mode="ref",
            )

        if pred_R is not None:
            _, pred_R = self.get_aligned_structures(
                system.native_R,
                pred_R,
                mode="ref",
            )
            _, pred_L = self.get_aligned_structures(
                system.native_L,
                pred_L,
                mode="ref",
            )

        if self.remove_differing_atoms:
            self._make_atoms_consistent(system.native_R, system.apo_R, system.pred_R)
            self._make_atoms_consistent(system.native_L, system.apo_L, system.pred_L)

        native = system.native_R + system.native_L
        # TODO: add uniprot seq and mapping to native
        assert len(system.holo_receptor.atom_array) == len(system.native_R.atom_array)
        assert len(system.holo_ligand.atom_array) == len(system.native_L.atom_array)
        return {
            "complex": native,
            "apo_receptor": apo_R,
            "apo_ligand": apo_L,
            "pred_receptor": pred_R,
            "pred_ligand": pred_L,
            "holo_receptor": system.holo_receptor,
            "holo_ligand": system.holo_ligand,
        }

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        metadata = self.metadata[self.metadata["id"] == row["id"]].iloc[0]
        # n.b. PinderSystem will automatically download if entry can't be found locally
        # TODO: if necessary, renumber reference ids to always be contiguous (before alignment)
        system = PinderSystem(
            entry=IndexEntry(**row.to_dict()), dataset_path=self.dataset_path
        )
        if system.entry.predicted_R:
            uniprot_seq_R = system.pred_receptor.sequence
        else:
            uniprot_seq_R = None
        if system.entry.predicted_L:
            uniprot_seq_L = system.pred_ligand.sequence
        else:
            uniprot_seq_L = None
        structures = self.make_structures(system)
        receptor_res_starts = get_residue_starts(system.holo_receptor.atom_array)
        ligand_res_starts = get_residue_starts(system.holo_ligand.atom_array)
        structures["receptor_uniprot_resids"] = [
            structures.pop("holo_receptor").resolved_pdb2uniprot[res_id]
            for res_id in system.holo_receptor.atom_array[receptor_res_starts].res_id
        ]
        structures["ligand_uniprot_resids"] = [
            structures.pop("holo_ligand").resolved_pdb2uniprot[res_id]
            for res_id in system.holo_ligand.atom_array[ligand_res_starts].res_id
        ]
        structures["receptor_uniprot_accession"] = system.entry.uniprot_R
        structures["ligand_uniprot_accession"] = system.entry.uniprot_L
        # TODO: get metadata
        structures["receptor_uniprot_seq"] = uniprot_seq_R
        structures["ligand_uniprot_seq"] = uniprot_seq_L
        for key in [
            "cluster_id",
            "cluster_id_R",
            "cluster_id_L",
        ]:
            structures[key] = metadata[key]
        for metadata_key in [
            "method",
            "resolution",
            "probability",  # probability that the protein complex is a true biological complex
            "oligomeric_count",
            "ECOD_names_R",
            "ECOD_names_L",
        ]:
            structures[metadata_key] = metadata[metadata_key]
        return structures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_length_monomer", type=int, default=None)
    parser.add_argument("--keep_representatives_only", action="store_true")
    parser.add_argument("--system_ids", type=List[str], default=None)
    args = parser.parse_args()

    index = get_index()
    metadata = get_metadata()

    if args.system_ids is not None:
        index = index[index["id"].isin(args.system_ids)]
    else:
        if args.subset is not None:
            assert index["split"] == "test"
            index = index[(index[f"{args.subset}"]) & (index["split"] == args.split)]
        else:
            index = index[index["split"] == args.split]
        index = index[
            (index["length1"] <= args.max_length_monomer)
            & (index["length2"] <= args.max_length_monomer)
        ]
        if args.keep_representatives_only:
            index = index.groupby("cluster_id").head(1)

    cluster_ids = list(index.cluster_id.unique())
    print(f"Pinder dataset: {len(cluster_ids)} clusters; {len(index)} systems")

    # TODO: decide on appropriate metadata (probably most of stuff from metadata...)
    features = Features(
        {
            "id": Value("string"),
            "cluster_id": Value("string"),
            "pdb_id": Value("string"),
            "complex": ProteinStructureFeature(),
            "apo_receptor": ProteinStructureFeature(),
            "apo_ligand": ProteinStructureFeature(),
            "pred_receptor": ProteinStructureFeature(),
            "pred_ligand": ProteinStructureFeature(),
            "receptor_uniprot_id": Value("string"),
            "ligand_uniprot_id": Value("string"),
            "receptor_uniprot_seq": Value("string"),
            "ligand_uniprot_seq": Value("string"),
            "receptor_uniprot_resids_with_structure": Array1D("uint16"),
            "ligand_uniprot_resids_with_structure": Array1D("uint16"),
            # metadata
            "oligomeric_count": Value("uint16"),
            "resolution": Value("float16"),
            "probability": Value("float16"),
            "method": Value("string"),
            # "chain_1": Value("string"),
            # "chain_2": Value("string"),
            # "assembly": Value("uint8"),
            # "asym_id_1": Value("string"),
            # "asym_id_2": Value("string"),
            # "ECOD_names_R": Array1D(Value("string")),
            # "ECOD_names_L": Array1D(Value("string")),
            # "link_density": Value("float16"),
            # "planarity": Value("float16"),
            # "n_residue_pairs": Value("uint16"),
            # "n_residues": Value("uint16"),
            # "buried_sasa": Value("float16"),
            # "intermolecular_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
            # "charged_charged_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
            # "charged_polar_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
            # "charged_apolar_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
            # "polar_polar_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
            # "apolar_polar_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
            # "apolar_apolar_contacts": Value("uint16"),  # total number of pair residues with any atom within a %A distance cutoff
        }
    )
    dataset = Dataset.from_generator(
        iter(PinderDataset(index, metadata, download=args.download)), features=features
    )