# -*- coding: utf-8 -*-
import sys
import numpy as np
import os, gc
import csv, glob
import os.path as path
import torch, pickle
from scipy.spatial import distance
import networkx as nx
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
        pdb_id = pdb_dict["#Pdb_origin"] + ".pdb"

#new
        print("Current working directory:", os.getcwd())
        print("Workdir:", workdir)
        #end

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

        graph_out = os.path.join(f"../../data/SKEMPI2/{subset}_cache/optimized", f"{str(i)}_{pdbcode}.pdb")
        os.system("mkdir -p {}".format(os.path.dirname(graph_out)))
        if os.path.exists(graph_out):
            print(f"{str(i)}_{pdbcode}.pdb exist!")
            continue

        print(f"generating {i}-th file")
        # build the wild-type file
        individual_file = os.path.join(workdir,'individual_list.txt')
        with open(individual_file, 'w') as f:
            cont = wildstr
            f.write(cont)
        
        # Change to workdir so FoldX runs in the right place
        cwd = os.getcwd()
        os.chdir(workdir)

        #new
        # Only use filenames (not full paths) once inside workdir
        pdb_filename = pdbcode + ".pdb"
        mut_file = "individual_list.txt"
        log_file = "foldx.log"
        print(f"Running FoldX from: {os.getcwd()}")
        comm = f'../FoldX --command=BuildModel --pdb={pdb_filename} --mutant-file={mut_file} --output-dir=. --pdb-dir={pdb_dir} > {log_file}'
        os.system(comm)
        #end

        #comm = '../../data/SKEMPI2/FoldX --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(pdb_id, individual_file, workdir, pdb_dir, workdir)
        #os.system(comm)

        #new
        # Move back to original directory
        os.chdir(cwd)

        # Ensure wildtype directory exists
        wildtype_dir = os.path.join(workdir, 'wildtype')
        os.makedirs(wildtype_dir, exist_ok=True)

        # Move the generated wildtype file to named output
        src = os.path.join(workdir, f"{pdbcode}_1.pdb")
        dst = os.path.join(wildtype_dir, f"{i}_{pdbcode}.pdb")
        if os.path.exists(src):
            os.rename(src, dst)
        else:
            print(f"❌ Wildtype file not found: {src}")
        #end

        # wildtype_dir = os.path.join("{}/wildtype".format(workdir))
        # if not os.path.exists(wildtype_dir):
        #     os.system("mkdir -p {}".format(wildtype_dir))
        # os.system(f'mv {workdir}/{pdbcode}_1.pdb {wildtype_dir}/{str(i)}_{pdbcode}.pdb')

        # print("h1")

        # build the mutant file
        individual_file = os.path.join(workdir, 'individual_list.txt')
        with open(individual_file, 'w') as f:
            cont = mutstr
            f.write(cont)
        pdb_id = f"{str(i)}_{pdbcode}.pdb"
        pdb_dir = wildtype_dir
        comm = '../../data/SKEMPI2/FoldX --command=BuildModel --numberOfRuns=1 --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(pdb_id, individual_file, workdir, pdb_dir, workdir)
        os.system(comm)

        # energy optimization
        comm = '../../data/SKEMPI2/FoldX --command=Optimize --pdb={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(f"{str(i)}_" + pdbcode + "_1" + ".pdb", workdir, workdir, workdir)
        os.system(comm)

        Optimized_dir = os.path.join("{}/optimized".format(workdir))

        print("h2")

        if not os.path.exists(Optimized_dir):
            os.system("mkdir -p {}".format(Optimized_dir))
        os.system(f'mv {workdir}/Optimized_{str(i)}_{pdbcode}_1.pdb {Optimized_dir}/{str(i)}_{pdbcode}.pdb') # before Optimized_{str(i)}_{pdbcode}_1.pdb

        os.system("rm {}/*.pdb".format(workdir))
        os.system("rm {}/*.fxout".format(workdir))


if __name__ == "__main__":
    main()