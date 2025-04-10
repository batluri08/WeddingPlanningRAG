
import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("wedding_venues.csv")
    documents = []
    for _, row in df.iterrows():
        text = (
            f"{row['Venue']} in {row['Location']} can seat {row['SeatingCapacity']} guests. "
            f"It costs ₹{row['Price']} and {'has' if row['OutdoorAvailable'] == 'Yes' else 'does not have'} outdoor seating. "
            f"The venue offers {row['FoodType']} food at ₹{row['PerPlateCost']} per plate. "
            f"Additional info: {row['Notes']}."
        )
        documents.append(Document(page_content=text))
    return df, documents

# Load Vector Store
@st.cache_resource
def setup_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Generator model
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# App Layout
st.title("💍 Wedding Planner RAG App")
st.markdown("Ask anything to find your perfect wedding venue!")

# Load
df, documents = load_data()
vectorstore = setup_vectorstore(documents)
generator = load_generator()

# Sidebar Filters
with st.sidebar:
    st.header("🔍 Filters")
    selected_location = st.selectbox("Location", ["Any"] + sorted(df['Location'].unique().tolist()))
    max_budget = st.slider("Max Venue Price (₹)", 500000, 3000000, 1500000, step=50000)
    food_type = st.selectbox("Preferred Food Type", ["Any"] + sorted(df['FoodType'].unique().tolist()))
    min_seating = st.slider("Minimum Seating", 50, 600, 250, step=50)

# User Query
user_query = st.text_input("Enter your wedding query:", "Suggest outdoor venues for 300 guests under 15L with veg food")

# Filtered Search
if st.button("Search"):
    filtered_docs = []
    for _, row in df.iterrows():
        if (selected_location == "Any" or row["Location"] == selected_location) and            (row["Price"] <= max_budget) and            (food_type == "Any" or row["FoodType"] == food_type) and            (row["SeatingCapacity"] >= min_seating):
            text = (
                f"{row['Venue']} in {row['Location']} can seat {row['SeatingCapacity']} guests. "
                f"It costs ₹{row['Price']} and {'has' if row['OutdoorAvailable'] == 'Yes' else 'does not have'} outdoor seating. "
                f"The venue offers {row['FoodType']} food at ₹{row['PerPlateCost']} per plate. "
                f"Additional info: {row['Notes']}."
            )
            filtered_docs.append(Document(page_content=text))

    if filtered_docs:
        vectorstore = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        results = vectorstore.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        prompt = f"Answer the following question using the venues provided.\n\nVenues:\n{context}\n\nQuestion: {user_query}"
        answer = generator(prompt, max_length=300)[0]['generated_text']
        st.subheader("💡 Recommended Option(s):")
        st.write(answer)
    else:
        st.warning("No matching venues found. Try adjusting the filters.")
