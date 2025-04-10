import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# ------------------ Load CSV (safe to cache) ------------------
@st.cache_data
def load_data():
    return pd.read_csv("wedding_venues.csv")

# ------------------ Convert Rows to LangChain Documents ------------------
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

# ------------------ Load Generator Model (Flan-T5) ------------------
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# ------------------ App Layout ------------------
st.title("💍 Wedding Planner RAG")
st.markdown("Ask any wedding planning question — our AI will help you choose venues!")

df = load_data()
documents = create_documents(df)
generator = load_generator()

# Embeddings + VectorStore (do NOT cache this!)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# ------------------ User Query Input ------------------
query = st.text_input("Enter your venue preferences:", 
                      "Outdoor wedding venues under ₹10L with veg food for 300 guests")

if st.button("Find venues"):
    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"Answer this based on the venues:\n{context}\n\nQuestion: {query}"
    answer = generator(prompt, max_length=300)[0]["generated_text"]
    st.subheader("💡 Suggested Option(s):")
    st.write(answer)
