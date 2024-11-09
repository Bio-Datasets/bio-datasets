"""We upload asymmetric units.

Ultimately what we want to be able to do is to infer the assembly from the coordinates for a single repeating unit.

Before running this script, download the PDB data to the directory specified by `--pdb_download_dir`.

e.g. with:

```
aws s3 cp --recursive --no-sign-request s3://pdbsnapshots/20240101/pub/pdb/data/structures/divided/mmCIF/ <path>
```
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile

from bio_datasets import Dataset, Features, NamedSplit, Value
from bio_datasets.features import AtomArrayFeature, StructureFeature


def get_pdb_id(assembly_file):
    return os.path.basename(assembly_file).split("-")[0]


def examples_generator(pair_codes, pdb_download_dir, compress):
    if pair_codes is None:
        result = subprocess.check_output(
            [
                "aws",
                "s3",
                "ls",
                "--no-sign-request",
                "s3://pdbsnapshots/20240101/pub/pdb/data/structures/divided/mmCIF/",
            ],
            text=True,
        )
        pair_codes = [
            line.split()[1][:-1] for line in result.splitlines() if "PRE" in line
        ]

    for pair_code in pair_codes:
        shutil.rmtree(os.path.join(pdb_download_dir, pair_code), ignore_errors=True)
        os.makedirs(os.path.join(pdb_download_dir, pair_code), exist_ok=True)
        # download from s3
        # TODO use boto3
        if not os.path.exists(os.path.join(pdb_download_dir, pair_code)):
            # intended that the directory is already downloaded - this is a backup
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--recursive",
                    "--no-sign-request",
                    f"s3://pdbsnapshots/20240101/pub/pdb/data/structures/divided/mmCIF/{pair_code}",
                    os.path.join(pdb_download_dir, pair_code),
                ],
                check=True,
            )

        if glob.glob(os.path.join(pdb_download_dir, pair_code, "*.cif.gz")):
            converter_args = [
                "cifs2bcifs",
                os.path.join(pdb_download_dir, pair_code),
                os.path.join(pdb_download_dir, pair_code),
                "--lite",
            ]
            if compress:
                converter_args.append("--compress")
            subprocess.run(
                converter_args,
                check=True,
            )

        assert not glob.glob(
            os.path.join(pdb_download_dir, pair_code, "*.bcif.gz")
        ), "bCIF.GZ files should not exist"

        downloaded_assemblies = glob.glob(
            os.path.join(
                pdb_download_dir, pair_code, "*.bcif.gz" if compress else "*.bcif"
            )
        )
        assert len(downloaded_assemblies) > 0, f"No assemblies found for {pair_code}"
        for assembly_file in downloaded_assemblies:
            yield {
                "id": get_pdb_id(assembly_file),
                "structure": {
                    "path": assembly_file,
                    "type": "bcif.gz" if compress else "bcif",
                },
            }
            os.remove(
                assembly_file.replace(".bcif.gz" if compress else ".bcif", ".cif.gz")
            )


def main(args):
    features = Features(
        id=Value("string"),
        structure=AtomArrayFeature() if args.as_array else StructureFeature(),
    )

    with tempfile.TemporaryDirectory(dir=args.temp_dir) as temp_dir:
        ds = Dataset.from_generator(
            examples_generator,
            gen_kwargs={
                "pair_codes": args.pair_codes,
                "pdb_download_dir": args.pdb_download_dir,
                "compress": args.compress,
            },
            features=features,
            cache_dir=temp_dir,
            split=NamedSplit("train"),
        )
        ds.push_to_hub("biodatasets/pdb", config_name=args.config_name or "default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument(
        "--pair_codes", nargs="+", help="PDB 2-letter codes", default=None
    )
    parser.add_argument(
        "--backbone_only", action="store_true", help="Whether to drop sidechains"
    )
    parser.add_argument(
        "--as_array", action="store_true", help="Whether to return an array"
    )
    parser.add_argument(
        "--pdb_download_dir",
        type=str,
        default="data/pdb",
        help="Directory to download PDBs to",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory (for caching built dataset)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Whether to compress the compressed bcif with gzip",
    )
    args = parser.parse_args()
    os.makedirs(args.pdb_download_dir, exist_ok=True)
    if args.temp_dir is not None:
        os.makedirs(args.temp_dir, exist_ok=True)

    main(args)
