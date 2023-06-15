import pandas as pd
import itertools
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

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

total_doc = read_file("RTL-High/RTL FA for High Rated Borrower_Sample 1.docx")

df = pd.read_excel("High Rated RTL FA_Standard Clause library_v6.xlsx", usecols='F:F')
out = df.dropna()
out = out.values.tolist()
merged = list(itertools.chain(*out))

# Two lists of sentences
sentences1 = merged

sentences2 = total_doc



#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.cos_sim(embeddings1, embeddings2)

res = []
#Output the pairs with their score
for i in range(len(sentences1)):
    
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[0], cosine_scores[i][0]))
    res.append((sentences1[i], sentences2[0], "{:.4f}".format(cosine_scores[i][0])))
    
res.sort(key = lambda x: x[2], reverse = True)

print(res[0:3])


    
    





