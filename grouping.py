import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

df = pd.read_csv('dataset/rule_embeddings.csv')

SSE = []
candidates = [30, 40, 50, 60, 70, 80, 90, 100]

for k in candidates:
    print('candidate: {}'.format(k))
    KMeans_ = KMeans(n_clusters=k, random_state=42)  
    KMeans_.fit(df)
    SSE.append(KMeans_.inertia_)
    print(SSE[-1])

plt.xlabel('群數')
plt.ylabel('SSE')
plt.title('不同k值之SSE')
plt.plot(candidates , SSE, 'o-')
plt.show()