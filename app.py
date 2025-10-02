import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

# Load secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --------------------------
# Load documents
# --------------------------
def load_docs(path="sample.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ["No documents found. Please upload sample.txt."]

docs = load_docs()

# --------------------------
# Embeddings
# --------------------------
def get_embedding(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding

# Precompute embeddings for docs
doc_embeddings = [get_embedding(doc) for doc in docs]

# NearestNeighbors index
nn = NearestNeighbors(n_neighbors=min(3, len(docs)), metric="cosine")
nn.fit(doc_embeddings)

# --------------------------
# Streamlit UI
# --------------------------
st.title("📚 Hybrid RAG Chatbot")

query = st.text_input("Ask me something:")

if query:
    # Embed query
    q_emb = np.array(get_embedding(query)).reshape(1, -1)

    # Retrieve nearest docs
    distances, indices = nn.kneighbors(q_emb)
    retrieved = [docs[i] for i in indices[0]]
    context = "\n".join(retrieved)

    # Best match distance
    best_distance = distances[0][0]
    threshold = 0.6  # tweak between 0.5–0.7

    if best_distance < threshold:
        # RAG mode
        source = "documents"
        prompt = f"""
        You are a helpful assistant.
        Answer the question ONLY using the following context.

        Context:
        {context}

        Question: {query}
        """
    else:
        # General fallback mode
        source = "general knowledge"
        prompt = f"""
        You are a helpful assistant.
        The provided documents are not relevant to the user's query.
        Answer the question from your general knowledge.

        Question: {query}
        """

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Show chat-like UI
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"*(Answered using {source})*")
