import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd

df = pd.read_csv('dataset/rule_embeddings.csv')

tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(df)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
plt.scatter(X_norm[:, 0], X_norm[:, 1])
plt.show()