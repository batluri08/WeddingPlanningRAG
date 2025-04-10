
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ------------------ Load CSV ------------------
@st.cache_data
def load_data():
    return pd.read_csv("wedding_venues.csv")

# ------------------ Load Embedding Model ------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Load Generator Model ------------------
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# ------------------ Embed Documents ------------------
def embed_documents(df, embedder):
    texts = []
    for _, row in df.iterrows():
        text = (
            f"{row['Venue']} in {row['Location']} can seat {row['SeatingCapacity']} guests. "
            f"It costs ₹{row['Price']} and {'has' if row['OutdoorAvailable'] == 'Yes' else 'does not have'} outdoor seating. "
            f"The venue offers {row['FoodType']} food at ₹{row['PerPlateCost']} per plate. "
            f"Additional info: {row['Notes']}."
        )
        texts.append(text)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    return texts, np.array(embeddings).astype("float32")

# ------------------ Setup FAISS Index ------------------
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ------------------ App Layout ------------------
st.title("💍 Wedding Planner (Lite RAG)")
st.markdown("Search for wedding venues using natural language — fast, smart, deployable!")

# Load models and data
df = load_data()
embedder = load_embedder()
generator = load_generator()

# Embed all text rows
texts, embeddings = embed_documents(df, embedder)
index = create_faiss_index(embeddings)

# Query box
query = st.text_input("Ask something like:", "Outdoor venue under 10L with veg food in Jaipur")
if st.button("Search"):
    query_vec = embedder.encode([query]).astype("float32")
    scores, indices = index.search(query_vec, k=3)
    context = "\n".join([texts[i] for i in indices[0]])
    prompt = f"Answer based on the venue info below.\n\n{context}\n\nQuestion: {query}"
    answer = generator(prompt, max_length=300)[0]["generated_text"]

    st.subheader("💡 Suggested Option:")
    st.write(answer)
