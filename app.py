import streamlit as st
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load API key from secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load documents
def load_docs(path="sample.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return chunks

# Embed documents
def embed_texts(texts):
    embeddings = []
    for t in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Build vector index
docs = load_docs()
doc_embeddings = embed_texts(docs)
nn = NearestNeighbors(n_neighbors=3, metric="cosine")
nn.fit(doc_embeddings)

# Streamlit UI
st.title("📚 Simple RAG Chatbot")

user_query = st.text_input("Ask a question:")

if user_query:
    # Embed query
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_query
    ).data[0].embedding

    # Find relevant docs
    q_emb = np.array(q_emb).reshape(1, -1)  # ensure shape (1, embedding_dim)
    distances, indices = nn.kneighbors(q_emb)
    retrieved = [docs[i] for i in indices[0]]

    # Create prompt
    context = "\n".join(retrieved)
    prompt = f"Answer the question using the context:\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    # Generate answer
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write("### 🤖 Answer:")
    st.write(completion.choices[0].message.content)
