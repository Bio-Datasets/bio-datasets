"""Python script for converting cif to bcif.

molstar has a js script: https://molstar.org/docs/data-access-tools/convert-to-bcif/
but python probably nicer.
"""
import argparse
import gzip
import json
import os

from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, compress
from biotite.structure.io.pdbx.convert import _get_block, _get_or_create_block
from joblib import Parallel, delayed

# inferred from 1aq1 binary cif file provided by pdb.
with open(os.path.join(os.path.dirname(__file__), "bcif_dtypes.json"), "r") as f:
    BCIF_FILE_DTYPES = json.load(f)


# just anything required for reconstructing the assembly.
LITE_COLUMNS_TO_KEEP = [
    # asym coords
    "atom_site",
    # required for fill missing residues
    "entity",
    "entity_poly",
    "entity_poly_seq",
    # required for assembly reconstruction
    "cell",
    "struct_asym",
    "pdbx_struct_assembly",  # needed?
    "pdbx_struct_assembly_gen",
    "pdbx_struct_oper_list",
    "symmetry",
]


def single_cif_to_bcif(
    input_file: str,
    output_file: str,
    lite: bool = False,
    float_rtol: float = 1e-6,
    compress_bcif: bool = False,
):
    if input_file.endswith(".gz"):
        with gzip.open(input_file, "rt") as f:
            inf = CIFFile.read(f)
    else:
        inf = CIFFile.read(input_file)
    outf = BinaryCIFFile()
    out_block = _get_or_create_block(outf, "1aq1")
    Category = out_block.subcomponent_class()
    in_block = _get_block(inf, None)
    for key, in_category in in_block.items():
        if lite and key not in LITE_COLUMNS_TO_KEEP:
            continue
        out_category = Category()
        for in_column, in_data in in_category.items():
            try:
                # exptl_crystal is causing type coercion issues. TODO: fix
                arr = in_data.as_array(BCIF_FILE_DTYPES[key][in_column])
            except Exception:
                arr = in_data.as_array()

            out_category[in_column] = arr
        out_block[key] = out_category
    outf = compress(outf, float_tolerance=float_rtol)
    if compress_bcif:
        with gzip.open(output_file + ".gz", "wb") as f:
            outf.write(f)
    else:
        outf.write(output_file)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--float_rtol", type=float, default=1e-6)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--compress", action="store_true")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    single_cif_to_bcif(
        args.input_path,
        args.output_path,
        lite=args.lite,
        float_rtol=args.float_rtol,
        compress_bcif=args.compress,
    )


def process_file(file, output_path, lite, float_rtol=1e-6, compress_bcif=False):
    assert file.endswith(".cif") or file.endswith(
        ".cif.gz"
    ), f"file must end with .cif or .cif.gz: {file}"
    if file.endswith(".gz"):
        new_file = os.path.splitext(os.path.splitext(file)[0])[0]
    else:
        new_file = os.path.splitext(file)[0]
    output_file = os.path.join(output_path, os.path.basename(new_file) + ".bcif")
    single_cif_to_bcif(
        file,
        output_file,
        lite=lite,
        float_rtol=float_rtol,
        compress_bcif=compress_bcif,
    )


def dir_main():
    parser = create_parser()
    args = parser.parse_args()
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(
            os.path.join(args.input_path, file),
            args.output_path,
            args.lite,
            args.float_rtol,
            args.compress,
        )
        for file in os.listdir(args.input_path)
    )


if __name__ == "__main__":
    main()
