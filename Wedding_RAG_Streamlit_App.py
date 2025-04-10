import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# ✅ Step 1: Load CSV — this is safe to cache
@st.cache_data
def load_data():
    return pd.read_csv("wedding_venues.csv")

# ✅ Step 2: Load FLAN-T5 — also safe to cache
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# ❌ DO NOT CACHE THESE: Documents or FAISS
def create_documents(df):
    docs = []
    for _, row in df.iterrows():
        text = (
            f"{row['Venue']} in {row['Location']} can seat {row['SeatingCapacity']} guests. "
            f"It costs ₹{row['Price']} and {'has' if row['OutdoorAvailable'] == 'Yes' else 'does not have'} outdoor seating. "
            f"The venue offers {row['FoodType']} food at ₹{row['PerPlateCost']} per plate. "
            f"Additional info: {row['Notes']}."
        )
        docs.append(Document(page_content=text))
    return docs

# ✅ UI
st.title("💍 Wedding Planner RAG")
df = load_data()
generator = load_generator()
documents = create_documents(df)

# ❌ Do not cache this
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# ✅ User query input
query = st.text_input("Ask your venue planning question:",
                      "Outdoor wedding venue for 300 guests under ₹10L with veg food")

if st.button("Search"):
    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"Answer this question based on the venues:\n{context}\n\nQuestion: {query}"
    answer = generator(prompt, max_length=300)[0]["generated_text"]
    st.subheader("💡 Suggested Option:")
    st.write(answer)

