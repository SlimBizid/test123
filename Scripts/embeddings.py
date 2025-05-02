import pandas as pd
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
import os

df = pd.read_csv("D:/Downloads/trustpilot_scraped_dataa.csv")

# Get complaints and their category
texts = df["description"].tolist()
categories = df["category"].tolist()

# Initialize embedding model from Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create FAISS vector store with category as metadata
vectorstore = FAISS.from_texts(
    texts,
    embedding=embeddings,
    metadatas=[{"category": cat} for cat in categories]
)

# Save the FAISS index
os.makedirs("embeddings_data" , exist_ok=True)
vectorstore.save_local("embeddings_data")

print("✅ Embeddings stored in FAISS using 'category' as metadata.")

