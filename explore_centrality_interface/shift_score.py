#!/usr/bin/env python

"""
shift_score.py

A combined shift scoring script measuring structural changes after optimization.

Tunables:
  - CHI_DEFINITIONS: Side-chain chi angle atom tuples per residue type.
  - CONTACT_CUTOFF: Distance (Å) for defining heavy-atom contacts.
  - w_dihedral: Weight for dihedral score (CLI: --w_dihedral).
  - w_contact: Weight for contact score (CLI: --w_contact).
"""

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.vectors import calc_dihedral, Vector

# ============================ Tunable parameters ============================
# Define which dihedrals to track per residue type
CHI_DEFINITIONS = {
    'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
    'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
    'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
    'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
    'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
    'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD')],
    'SER': [('N','CA','CB','OG')],
    'THR': [('N','CA','CB','OG1')],
    'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'VAL': [('N','CA','CB','CG1')]
}
CONTACT_CUTOFF = 10  # Å; cutoff for heavy-atom contact definition
# ===========================================================================


def _get_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('struct', pdb_path)


def _get_mutation_residue(structure, mutation):
    chain_id = mutation[1]
    res_pos = int(mutation[2:-1])
    for model in structure:
        chain = model[chain_id]
        return chain[(" ", res_pos, " ")]
    raise KeyError(f"Residue {mutation} not found in {pdb_path}")


def compute_dihedral_score(pdb_wt, pdb_opt, mutation):
    """
    Compute sum of absolute chi-angle changes (°) for all side chains except the mutated residue.
    """
    wt, opt = _get_structure(pdb_wt), _get_structure(pdb_opt)
    mut_res = _get_mutation_residue(wt, mutation)
    total_delta = 0.0
    # Compare only first model
    for res_wt, res_opt in zip(wt[0].get_residues(), opt[0].get_residues()):
        if res_wt.id[0] != ' ' or res_wt == mut_res:
            continue
        name = res_wt.get_resname()
        if name not in CHI_DEFINITIONS:
            continue
        for atom_names in CHI_DEFINITIONS[name]:
            try:
                coords_wt = [Vector(res_wt[a].get_coord()) for a in atom_names]
                coords_opt = [Vector(res_opt[a].get_coord()) for a in atom_names]
                chi_wt = np.degrees(calc_dihedral(*coords_wt))
                chi_opt = np.degrees(calc_dihedral(*coords_opt))
                diff = abs(chi_opt - chi_wt) % 360
                delta = min(diff, 360 - diff)
                # ignore symmetric 180° flips in aromatics
                if name in ('PHE','TYR','HIS') and abs(delta - 180) < 1e-3:
                    continue
                total_delta += delta
            except KeyError:
                continue
    return total_delta/1000


def compute_contact_score(pdb_wt, pdb_opt, mutation):
    """
    Compute sum of distance changes (Å) for all gained or lost residue-residue contacts.
    """
    wt_model, opt_model = _get_structure(pdb_wt)[0], _get_structure(pdb_opt)[0]
    # heavy atoms only
    atoms_wt = [a for a in wt_model.get_atoms() if a.element != 'H']
    atoms_opt = [a for a in opt_model.get_atoms() if a.element != 'H']
    ns_wt, ns_opt = NeighborSearch(atoms_wt), NeighborSearch(atoms_opt)
    def res_id(atom): return (atom.get_parent().get_parent().id, atom.get_parent().id[1])
    contacts = {}
    for label, atoms, ns in [('wt', atoms_wt, ns_wt), ('opt', atoms_opt, ns_opt)]:
        contacts[label] = set()
        for atom in atoms:
            for nbr in ns.search(atom.get_coord(), CONTACT_CUTOFF):
                if nbr is atom: continue
                id1, id2 = res_id(atom), res_id(nbr)
                if id1 == id2: continue
                contacts[label].add(tuple(sorted((id1, id2))))
    gained = contacts['opt'] - contacts['wt']
    lost   = contacts['wt']  - contacts['opt']
    score = 0.0
    # weight by change in minimal inter-atomic distance
    for pair in gained.union(lost):
        pts_wt = [a.get_coord() for a in atoms_wt if res_id(a) in pair]
        pts_opt= [a.get_coord() for a in atoms_opt if res_id(a) in pair]
        d_wt = min(np.linalg.norm(a-b) for a in pts_wt for b in pts_wt if not np.allclose(a,b))
        d_opt= min(np.linalg.norm(a-b) for a in pts_opt for b in pts_opt if not np.allclose(a,b))
        score += abs(d_opt - d_wt)
    return 1000 * score


def combined_shift_score(pdb_wt: str,
                         pdb_opt: str,
                         mutation: str,
                         w_dihedral: float = 1.0,
                         w_contact: float = 1.0) -> float:
    """Return weighted sum: w_dihedral*dihedral_score + w_contact*contact_score."""
    d = compute_dihedral_score(pdb_wt, pdb_opt, mutation)
    c = compute_contact_score(pdb_wt, pdb_opt, mutation)
    return w_dihedral * d + w_contact * c


def combined_shift_scores(pdb_wt: str,
                          pdb_opt: str,
                          mutations: list,
                          w_dihedral: float = 1.0,
                          w_contact: float = 1.0) -> list:
    return [combined_shift_score(pdb_wt, pdb_opt, m, w_dihedral, w_contact)
            for m in mutations]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compute dihedral, contact, and combined scores for mutations.")
    p.add_argument("wildtype_pdb", nargs='?', default="data/SKEMPI2/SKEMPI2_cache/wildtype/174_1BRS.pdb",
                   help="Path to wildtype PDB file")
    p.add_argument("optimized_pdb", nargs='?', default="data/SKEMPI2/SKEMPI2_cache/optimized/174_1BRS.pdb",
                   help="Path to optimized PDB file")
    p.add_argument("mutations", nargs='*', default=["HA100L"],
                   help="Mutation strings, e.g. HA100L")
    p.add_argument("--w_dihedral", type=float, default=1.0,
                   help="Weight for dihedral score")
    p.add_argument("--w_contact", type=float, default=2.0,
                   help="Weight for contact score")
    args = p.parse_args()
    if not args.mutations:
        p.error("At least one mutation must be provided")
    for m in args.mutations:
        d_score = compute_dihedral_score(args.wildtype_pdb, args.optimized_pdb, m)
        c_score = compute_contact_score(args.wildtype_pdb, args.optimized_pdb, m)
        combined = args.w_dihedral * d_score + args.w_contact * c_score
        print(f"{m}\tDihedral: {d_score:.3f}\tContact: {c_score:.3f}\tCombined: {combined:.3f}")
