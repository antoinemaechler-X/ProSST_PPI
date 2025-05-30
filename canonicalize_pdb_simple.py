#!/usr/bin/env python3
"""
canonicalize_and_refine.py

For each .pdb in --in-dir, produce a file in --out-dir that:
  • keeps only ATOM lines
  • skips hydrogens (atom names starting with H or element H)
  • renumbers atoms 0..N-1
  • renumbers residues 0..M-1
  • rounds coords to 0.001 Å
  • resets occupancy→1.00, bfactor→0.00
  • preserves the original chain ID
"""
import argparse
from pathlib import Path

def is_hydrogen(atom_name: str, element: str) -> bool:
    """Return True if this atom should be considered a hydrogen."""
    if atom_name.strip().startswith("H"):
        return True
    if element.strip().upper() == "H":
        return True
    return False

def process_pdb(inp_path: Path, out_path: Path):
    out_lines = []
    atom_idx   = 0
    last_res   = None
    res_idx    = -1

    with inp_path.open() as f:
        for L in f:
            # only ATOM lines
            if not L.startswith("ATOM"):
                continue

            atom_name = L[12:16]
            element   = L[76:78]
            chain_id  = L[21]        # original chain ID (column 22)
            try:
                resnum  = int(L[22:26])
                x, y, z = map(float, (L[30:38], L[38:46], L[46:54]))
            except ValueError:
                continue  # skip bad lines

            # drop hydrogens
            if is_hydrogen(atom_name, element):
                continue

            # new zero-based residue index
            if resnum != last_res:
                res_idx += 1
                last_res = resnum

            # rebuild a clean ATOM line preserving chain
            new_line = (
                f"ATOM  "
                f"{atom_idx:5d} "
                f"{atom_name:4s}"
                f"{L[17:20]:>3s} {chain_id}"
                f"{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}           "
                f"{atom_name.strip():>2s}\n"
            )
            out_lines.append(new_line)
            atom_idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Canonicalize + refine PyRosetta PDBs"
    )
    parser.add_argument(
        "--in-dir",  type=Path, required=True,
        help="Directory of original PyRosetta PDBs"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Where to write the refined/canonicalized PDBs"
    )
    args = parser.parse_args()

    pdbs = sorted(args.in_dir.glob("*.pdb"))
    if not pdbs:
        print(f"No PDBs found in {args.in_dir}")
        return

    print(f"Processing {len(pdbs)} PDB(s)…")
    for pth in pdbs:
        outp = args.out_dir / pth.name
        print(f" • {pth.name}")
        process_pdb(pth, outp)

    print(f"\nDone! Refined PDBs are in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
