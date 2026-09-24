from sentence_transformers import util
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Google OAuth allows users to sign in using their Google account.",
    "PostgreSQL indexes make database queries faster.",
    "Supabase Storage is used to store files and images.",
    "JWT tokens can be used to authenticate API requests.",
    "Docker packages applications into portable containers."
]

embeddings = model.encode(documents)

question = "How can I let users log in with Google?"

question_embedding = model.encode(question)

print(question_embedding.shape)

scores = util.cos_sim(question_embedding, embeddings)[0]

print(scores)
