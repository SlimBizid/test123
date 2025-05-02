from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Step 1: Reinitialize embedding model (must match the one used to create the store)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 2: Load FAISS vector store from disk
vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)

# Step 3: Define classification function
def classify_text(input_text, k=3):
    # Retrieve top-k similar texts
    retrieved_docs = vectorstore.similarity_search(input_text, k=k)
    
    # Format examples
    examples = "\n".join([f'"{doc.page_content}" → {doc.metadata["category"]}' for doc in retrieved_docs])

    # Prompt setup
    prompt = PromptTemplate(
        input_variables=["input_text", "examples"],
        template="""
You are a classification assistant.
Based on the input and examples, return only the best matching category.

Input: {input_text}

Examples:
{examples}

Category:"""
    )

    # Use Ollama model for classification
    llm = Ollama(model="llama2")  
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.run({
        "input_text": input_text,
        "examples": examples
    })

    return result.strip()

# Example usage
#query = "I was charged twice and nobody responds to my emails."
#predicted_category = classify_text(query)
#print("Predicted Category:", predicted_category)
