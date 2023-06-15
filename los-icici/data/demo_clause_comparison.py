import torch

import pandas as pd
import itertools
from sentence_transformers import SentenceTransformer, util

import docx2txt

def read_file(file):
    text = docx2txt.process(file)
    
    #Prints output after converting
 #   print ("After converting text is ",text)
    
    content = []
    for line in text.splitlines():
      #This will ignore empty/blank lines. 
      if line != '':
        #Append to list
        content.append(line)
    
 #   print (content)
    return content

embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus = read_file(r"RTL-High/RTL FA for High Rated Borrower_Sample 1.docx")


corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
    
df = pd.read_excel("High Rated RTL FA_Standard Clause library_v6.xlsx", usecols='F:F')
out = df.dropna()
out = out.values.tolist()
queries = list(itertools.chain(*out))[:10]

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(2, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 2 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))