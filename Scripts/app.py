import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Load embeddings and FAISS vectorstore ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)
llm = Ollama(model="llama2:7b")  

# --- Classification function ---
def classify_text(input_text: str, k: int = 3) -> str:
    retrieved_docs = vectorstore.similarity_search(input_text, k=k)
    examples = "\n".join([f'"{doc.page_content}" → {doc.metadata["category"]}' for doc in retrieved_docs])

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

    chain = prompt | llm
    result = chain.invoke({"input_text": input_text, "examples": examples})
    return result.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Text Classifier", layout="centered")
st.title("📂 Complaint Text Classifier")
st.write("Enter a complaint or customer message to predict its category using a vector-based RAG system.")

user_input = st.text_area("Enter text to classify:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            category = classify_text(user_input)
        st.success(f"🧠 Predicted Category: **{category}**")

# Optional: Show retrieved examples
if user_input.strip():
    with st.expander("See similar examples used for classification"):
        retrieved_docs = vectorstore.similarity_search(user_input, k=3)
        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Example {i}**\n\n> {doc.page_content}\n\n**→ {doc.metadata['category']}**")
