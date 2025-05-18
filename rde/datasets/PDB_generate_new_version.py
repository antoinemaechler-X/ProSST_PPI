# -*- coding: utf-8 -*-
import sys
import numpy as np
import os, gc
import csv, glob
import os.path as path
import torch, pickle
from scipy.spatial import distance
#import networkx as nx
import Bio.PDB
from matplotlib import pylab
import  pandas as pd

import scipy.sparse as sp
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def main():
    subset = "SKEMPI2" #before M1340
    df = pd.read_csv(f'../../data/SKEMPI2/{subset}.csv')
    for i, pdb_dict in df.iterrows():

        pdb_dir = '../../data/SKEMPI2/PDBs'
        workdir = f'../../data/SKEMPI2/{subset}_cache'
        pdbcode = pdb_dict["#Pdb_origin"]
        if pdbcode == "1KBH":
            print("1KBH is locked!")
            continue

        graph_out = os.path.join(f"../../data/SKEMPI2/{subset}_cache/optimized", f"{str(i)}_{pdbcode}.pdb")
        os.system("mkdir -p {}".format(os.path.dirname(graph_out)))
        if os.path.exists(graph_out):
            print(f"{str(i)}_{pdbcode}.pdb exist!")
            continue

        pdb_id = pdb_dict["#Pdb_origin"] + ".pdb"

        mutstr = pdb_dict["Mutation(s)_cleaned"]
        mut_list = pdb_dict["Mutation(s)_cleaned"].split(",")
        wild_list = []
        for mut in mut_list:
            wildname = list(mut)[0]
            chainid = list(mut)[1]
            resid = "".join(list(mut)[2:-1])
            mutname = list(mut)[-1]
            wild_list.append("".join([wildname, chainid, resid, wildname]))
        wildstr = ",".join(wild_list) + ";"
        mutstr = ",".join(mut_list) + ";"

        print(f"generating {i}-th file")
        # build the wild-type file
        individual_file = os.path.join(workdir,'individual_list.txt')
        with open(individual_file, 'w') as f:
            cont = wildstr
            f.write(cont)
        
        # Save current directory to return to later
        cwd = os.getcwd()

        # Go to workdir (SKEMPI2_cache)
        os.chdir(workdir)

        # File names
        individual_file = "individual_list.txt"
        pdb_filename = f"{pdbcode}.pdb"
        output_model = f"{pdbcode}_1.pdb"
        renamed_output = f"{i}_{pdbcode}.pdb"

        # Write wildtype mutations (e.g., SI41S;)
        with open(individual_file, 'w') as f:
            f.write(wildstr)

        # Run FoldX
        foldx_command = f"../FoldX --command=BuildModel --pdb={pdb_filename} --mutant-file={individual_file} --output-dir=. --pdb-dir=../PDBs > foldx.log"
        os.system(foldx_command)

        # Return to original working directory
        os.chdir(cwd)

        # Ensure wildtype dir exists
        wildtype_dir = os.path.join(workdir, "wildtype")
        os.makedirs(wildtype_dir, exist_ok=True)

        # Move the generated file to wildtype folder with correct naming
        src_path = os.path.join(workdir, output_model)
        dst_path = os.path.join(wildtype_dir, renamed_output)

        if os.path.exists(src_path):
            os.rename(src_path, dst_path)
        else:
            print(f"❌ Wildtype file not found: {src_path}")

        # === Build the mutant file ===
        cwd = os.getcwd()
        os.chdir(workdir)

        individual_file = "individual_list.txt"
        with open(individual_file, "w") as f:
            f.write(mutstr)

        # Use previously generated wildtype as base
        input_pdb = f"{i}_{pdbcode}.pdb"  # this is '0_1CSE.pdb'
        wildtype_input = os.path.join("wildtype", input_pdb)
        mutant_output = f"{input_pdb.split('.')[0]}_1.pdb"  # → '0_1CSE_1.pdb'
        optimized_result = f"Optimized_{mutant_output}"     # → 'Optimized_0_1CSE_1.pdb'
        optimized_final = os.path.join("optimized", f"{i}_{pdbcode}.pdb")

        # Copy the wildtype structure into current dir so FoldX can use it
        os.system(f"cp {wildtype_input} .")

        # Run FoldX BuildModel for mutant
        foldx_command = f"../FoldX --command=BuildModel --numberOfRuns=1 --pdb={input_pdb} --mutant-file={individual_file} --output-dir=. --pdb-dir=wildtype > foldx.log"
        os.system(foldx_command)

        # === Optimize the mutant ===
        opt_command = f"../FoldX --command=Optimize --pdb={mutant_output} --output-dir=. --pdb-dir=. >> foldx.log"
        os.system(opt_command)

        # Make sure output directory exists
        os.makedirs("optimized", exist_ok=True)

        # Move optimized structure if it exists
        if os.path.exists(optimized_result):
            os.rename(optimized_result, optimized_final)
        else:
            print(f"❌ Optimized file not found: {optimized_result}")

        # === Cleanup ===
        for f in os.listdir('.'):
            if f.endswith('.pdb') or f.endswith('.fxout'):
                os.remove(f)

        # Return to original directory
        os.chdir(cwd)


if __name__ == "__main__":
    main()