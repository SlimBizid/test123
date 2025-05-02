from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# Reload the FAISS store
#vectorstore = FAISS.load_local("embeddings", embeddings=OllamaEmbeddings(model="nomic-embed-text"))
vectorstore = FAISS.load_local(
    "embeddings_data",
    OllamaEmbeddings(model="nomic-embed-text"),
    allow_dangerous_deserialization=True
)
# Access stored metadata
docs = vectorstore.similarity_search("item not received", k=5)

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:")
    print("Text:", doc.page_content)
    print("Metadata:", doc.metadata)
