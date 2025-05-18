import numpy as np

data_mt = np.load("data/SKEMPI2/SKEMPI2_cache/embedding_optimized_2048/0_1CSE_B_embeddings.npy")
data_wt = np.load("data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_2048/0_1CSE_B_embeddings.npy")

print(data_mt.shape)
print("mt")
print(data_mt)

print(data_wt.shape)
print("wt")
print(data_wt)

print("diff")
print(data_mt - data_wt)

print("Indices where diff is non zero")
print(np.sum(np.nonzero(data_mt - data_wt)))

