from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain_ollama.llms import OllamaLLM # Updated import as per deprecation warning
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.embeddings.base import BaseRagasEmbeddings
from typing import List, AsyncGenerator
import asyncio

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load FAISS vector store
vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)

# Custom Ragas Embeddings using Ollama
class OllamaRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name="nomic-embed-text"):
        super().__init__()
        self.embedding_model = OllamaEmbeddings(model=model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def _agenerate_embeddings(self, texts: List[str]) -> AsyncGenerator[List[float], None]:
        embeddings = await self.aembed_documents(texts)
        for embedding in embeddings:
            yield embedding

# Custom Ragas LLM using Ollama
import concurrent.futures

from langchain_core.outputs import Generation, LLMResult

class OllamaRagasLLM:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)

    def set_run_config(self, config):
        self.run_config = config

    async def generate(self, prompts: List[str], n: int = 1, **kwargs) -> LLMResult:
        def call_model(prompt):
            return self.llm.invoke(prompt)

        generations = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(call_model, prompt) for prompt in prompts]
            for future in futures:
                try:
                    output = future.result(timeout=30)
                    generations.append([Generation(text=output)])
                except concurrent.futures.TimeoutError:
                    generations.append([Generation(text="")])

        return LLMResult(generations=generations)


# Classification function using LangChain
def classify_text(input_text: str, k: int = 3) -> str:
    retrieved_docs = vectorstore.similarity_search(input_text, k=k)
    examples = "\n".join([f'"{doc.page_content}" → {doc.metadata.get("category", "unknown")}' for doc in retrieved_docs])

    prompt = PromptTemplate.from_template("""
You are a classification assistant.
Based on the input and examples, return only the best matching category.

Input: {input_text}

Examples:
{examples}

Category:""")

    llm = OllamaLLM(model="mistral")  
    chain = {"input_text": RunnablePassthrough(), "examples": lambda x: examples} | prompt | llm
    return chain.invoke(input_text).strip()

# Async evaluation
async def evaluate_classification_async(test_cases):
    results = []

    for case in test_cases:
        input_text = case["question"]
        ground_truth = case["ground_truth"]

        predicted_category = classify_text(input_text)
        retrieved_docs = vectorstore.similarity_search(input_text, k=3)
        contexts = [doc.page_content for doc in retrieved_docs]

        results.append({
            "question": input_text,
            "answer": predicted_category,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

    dataset = Dataset.from_pandas(pd.DataFrame(results))

    ragas_embeddings = OllamaRagasEmbeddings()
    ragas_llm = OllamaRagasLLM()

    result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        ],
        embeddings=ragas_embeddings,
        llm=ragas_llm,
    )

    return result

# Sync wrapper
def evaluate_classification(test_cases):
    return asyncio.run(evaluate_classification_async(test_cases))

# Test usage
if __name__ == "__main__":
    test_cases = [
        {
            "question": "I didn't receive my order.",
            "ground_truth": "shipping"
        },
        {
            "question": "They didn't answer my calls.",
            "ground_truth": "customer_service"
        },
        {
            "question": "Broken item.",
            "ground_truth": "product_quality"
        }
    ]

    evaluation_results = evaluate_classification(test_cases)
    print(evaluation_results)