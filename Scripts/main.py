from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# === Initialize FastAPI app ===
app = FastAPI(
    title="Text Classification API",
    description="RAG-based complaint classifier using FAISS and Ollama",
    version="1.0"
)

# === Load vector store and embeddings ===
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)
    llm = Ollama(model="llama2")
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

# === Request body schema ===
class ClassifyRequest(BaseModel):
    text: str
    top_k: int = 3  # Optional: how many examples to retrieve

# === Health check endpoint ===
@app.get("/health")
def health_check():
    return {"status": "ok"}

# === POST endpoint for classification ===
@app.post("/classify")
def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        # Retrieve top-k similar examples
        retrieved_docs = vectorstore.similarity_search(req.text, k=req.top_k)

        # Format examples
        examples = "\n".join([
            f'"{doc.page_content}" → {doc.metadata.get("category", "Unknown")}'
            for doc in retrieved_docs
        ])

        # Build prompt
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

        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run({
            "input_text": req.text,
            "examples": examples
        })

        return {
            "category": result.strip(),
            "similar_examples": [
                {
                    "text": doc.page_content,
                    "category": doc.metadata.get("category", "Unknown")
                }
                for doc in retrieved_docs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")

