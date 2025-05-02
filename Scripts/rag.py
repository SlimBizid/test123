#from langchain.embeddings import OllamaEmbeddings
#from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


#vectorstore = FAISS.load_local("embeddings", OllamaEmbeddings(model="nomic-embed-text"))
vectorstore = FAISS.load_local(
    "embedding_data",
    OllamaEmbeddings(model="nomic-embed-text"),
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

llm = Ollama(model="llama2:7b")  # or another supported LLM

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = "I never recieved my order? It's been 2 weeks!"
result = qa_chain({"query": query})

print("📌 Answer:", result["result"])
