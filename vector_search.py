import numpy as np
import faiss

d = 128 
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# index = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(
    faiss.IndexFlatL2(d), d, 100
)
index.train(xb)

index.add(xb)

D, I = index.search(xq, 5)

print(I)