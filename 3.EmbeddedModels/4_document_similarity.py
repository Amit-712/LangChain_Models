# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=300)

# documents=[
#     "Virat Kohli is an Indain Crickter",
#     "Rohit Sharma is former Indian captain famous",
#     "Sachin Tendulkar also known as 'God of cricket'",
#     "Jasprit Bumrah is an Indian Fast bowler"
# ]

# query = "tell me about virat kohli"

# doc_embeddings = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# print(cosine_similarity([query_embedding], doc_embeddings))





from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents=[
    "Virat Kohli is an Indain Crickter",
    "Rohit Sharma is former Indian captain famous",
    "Sachin Tendulkar also known as 'God of cricket'",
    "Jasprit Bumrah is an Indian Fast bowler"
]

query = "tell me about rohit kohli"


doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is:", score)
