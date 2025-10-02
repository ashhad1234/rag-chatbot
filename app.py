import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load documents
def load_docs(path="docs/sample.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

docs = load_docs()

# Function to get embeddings
def get_embedding(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding

# Build vector index
X = np.array([get_embedding(doc) for doc in docs])

# Dynamically choose number of neighbors
k = min(3, len(docs))  # can't request more neighbors than docs available
nn = NearestNeighbors(n_neighbors=k, metric="cosine")
nn.fit(X)

# Streamlit UI
st.title("🔎 RAG Chatbot")

user_query = st.text_input("Ask me something:")

if user_query:
    # Get query embedding
    q_emb = get_embedding(user_query)
    q_emb = np.array(q_emb).reshape(1, -1)

    # Find relevant docs
    distances, indices = nn.kneighbors(q_emb)
    retrieved = [docs[i] for i in indices[0]]

    # Create prompt
    context = "\n".join(retrieved)
    prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQuestion: {user_query}"

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write("**Answer:**", response.choices[0].message.content)
    st.write("---")
    st.write("**Retrieved context:**")
    for r in retrieved:
        st.write("-", r)
