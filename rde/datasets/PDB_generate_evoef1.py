#!/usr/bin/env python3
"""
Script to generate wildtype and mutant complex structures using EvoEF1,
modeled after the FoldX pipeline. Run from rde/datasets/.
"""
import os
import glob
import shutil
import sys
import pandas as pd

from Bio.PDB import PDBParser


def main():
    subset = "M1340"
    csv_path = os.path.join("..", "..", "data", "SKEMPI2", f"{subset}.csv")
    df = pd.read_csv(csv_path)

    pdb_dir = os.path.abspath(os.path.join("..", "..", "data", "SKEMPI2", "PDBs"))
    workdir = os.path.abspath(os.path.join("..", "..", "data", "SKEMPI2", f"{subset}_cache"))
    evoef_bin = os.path.abspath(os.path.join("..", "..", "data", "SKEMPI2", "evoef1", "EvoEF"))

    wildtype_dir = os.path.join(workdir, "wildtype_evoef1")
    optimized_dir = os.path.join(workdir, "optimized_evoef1")
    os.makedirs(wildtype_dir, exist_ok=True)
    os.makedirs(optimized_dir, exist_ok=True)

    errors = []
    for i, row in df.iterrows():
        pdbcode = row["#Pdb_origin"]
        mut_clean = row["Mutation(s)_cleaned"]
        print(f"Processing {i}: {pdbcode} | Mutations: {mut_clean}")

        # default model filename for cleanup
        model = "current_input_Model_0001.pdb"

        if pdbcode == "1KBH":
            print(f"Skipped locked PDB: {pdbcode}")
            continue

        orig_cwd = os.getcwd()
        try:
            # locate source PDB
            pattern = os.path.join(pdb_dir, f"*{pdbcode}.pdb")
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No file matching {pattern}")
            exact = [m for m in matches if os.path.basename(m) == f"{pdbcode}.pdb"]
            src_pdb = exact[0] if exact else [m for m in matches if not os.path.basename(m).startswith('._')][0]
            src_pdb = os.path.abspath(src_pdb)

            wild_out = os.path.join(wildtype_dir, f"{i}_{pdbcode}.pdb")
            opt_out = os.path.join(optimized_dir, f"{i}_{pdbcode}.pdb")
            # skip if already done
            if os.path.exists(wild_out) and os.path.exists(opt_out):
                print(f"Both wildtype and mutant exist, skipping: {wild_out}, {opt_out}")
                continue
            opt_out = os.path.join(optimized_dir, f"{i}_{pdbcode}.pdb")
            if os.path.exists(opt_out):
                print(f"Already exists, skipping mutant generation: {opt_out}")
                continue

            # change to workdir and clean old files
            os.chdir(workdir)
            for f in glob.glob("current_input*"):
                try:
                    os.remove(f)
                except OSError:
                    pass

            # repair wildtype
            shutil.copy(src_pdb, "current_input.pdb")
            ret = os.system(f"{evoef_bin} --command=RepairStructure --pdb=current_input.pdb")
            if os.path.exists("current_input_Repair.pdb"):
                os.rename("current_input_Repair.pdb", "current_input.pdb")
            else:
                raise RuntimeError(f"RepairStructure failed (exit {ret})")
            shutil.copy("current_input.pdb", wild_out)

            # write mutation file
            mut_file = "evoef_mut.txt"
            muts = [m.strip() for m in mut_clean.split(",") if m.strip()]
            with open(mut_file, "w") as mf:
                for mut in muts:
                    mf.write(f"{mut};\n")

            # build mutant
            ret = os.system(
                f"{evoef_bin} --command=BuildMutant --pdb=current_input.pdb --mutant_file={mut_file} --num_of_runs=10"
            )
            if not os.path.exists(model):
                raise RuntimeError(f"BuildMutant failed (exit {ret})")
            shutil.copy(model, opt_out)

        except Exception as e:
            print(f"❌ Error processing {pdbcode}: {e}")
            errors.append((pdbcode, str(e)))
        finally:
            # cleanup intermediates
            cleanup_files = ["current_input.pdb", "current_input_Repair.pdb", "evoef_mut.txt"]
            if model and os.path.exists(model):
                cleanup_files.append(model)
            for f in cleanup_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            # return to original directory
            os.chdir(orig_cwd)

    # summary
    if errors:
        print("The following complexes failed:")
        for code, msg in errors:
            print(f" - {code}: {msg}")
    else:
        print("All complexes processed successfully.")

if __name__ == "__main__":
    main()
