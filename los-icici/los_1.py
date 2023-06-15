from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('stsb-roberta-large')

# Two lists of sentences
sentences1 = ['“IBC” means the Insolvency and Bankruptcy Code, 2016, excluding all replacements and amendments made thereto and all rules and regulations framed thereunder. ']

sentences2 = ["“IBC” means the Insolvency and Bankruptcy Code, 2016, excluding all replacements and amendments made thereto and all rules and regulations framed thereunder." ]

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))