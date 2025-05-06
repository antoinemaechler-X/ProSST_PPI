from prosst.structure.quantizer import PdbQuantizer
processor = PdbQuantizer(structure_vocab_size=2048) # can be 20, 128, 512, 1024, 2048, 4096
result = processor("example_data/p1.pdb", return_residue_seq=False)