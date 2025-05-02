from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd

# Step 1: Load CSV directly with pandas to get categories
df = pd.read_csv("C:/Users/yasmi/Downloads/trustpilot_scraped_dataa.csv")

# Step 2: Use CSVLoader to load 'description' as content
loader = CSVLoader(
    file_path="C:/Users/yasmi/Downloads/trustpilot_scraped_dataa.csv",
    source_column="description"
)
docs = loader.load()

# Step 3: Inject 'category' column as metadata
for i, doc in enumerate(docs):
    doc.metadata["category"] = df.loc[i, "category"]

# Step 4: Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
split_docs = text_splitter.split_documents(docs)

# Step 5: Initialize Ollama Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 6: Create a new FAISS vectorstore
vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

# Step 7: Save vectorstore locally
os.makedirs("embedding_data", exist_ok=True)
vectorstore.save_local("embedding_data")

print("✅ FAISS vectorstore created with category metadata and saved.")

