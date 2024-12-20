import tensorflow_hub as hub
import pandas as pd
import numpy as np

embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
# embeddings = embed([
#     "The quick brown fox jumps over the lazy dog."])

df = pd.read_csv('dataset/train.csv')
df = df['EnglishRules']

arr = np.zeros([df.shape[0], 512])
for idx, row in enumerate(df):
    if idx % 100 == 0:
        print('{} completed'.format(idx + 1))
    arr[idx] = embed([row])[0].numpy()


np.savetxt('dataset/rule_embeddings.csv', arr, delimiter=',')
# print(embeddings[0].numpy().tolist())

# The following are example embedding output of 512 dimensions per sentence
# Embedding for: The quick brown fox jumps over the lazy dog.
# [-0.03133016 -0.06338634 -0.01607501, ...]
# Embedding for: I am a sentence for which I would like to get its embedding.
# [0.05080863 -0.0165243   0.01573782, ...]