from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import Generation, LLMResult

from datasets import Dataset
import pandas as pd
import asyncio
from typing import List


# ---- Embeddings ----
class SimpleOllamaEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name="nomic-embed-text"):
        self.embedding_model = OllamaEmbeddings(model=model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)


# ---- LLM Wrapper for RAGAS ----
class SimpleOllamaLLM:
    def __init__(self, model_name="gemma:2b"):
        self.llm = OllamaLLM(model=model_name)

    async def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            result = self.llm.invoke(prompt)
            generations.append([Generation(text=result)])
        return LLMResult(generations=generations)


# ---- Load Vectorstore ----
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)


# ---- Classification Logic ----
def classify_text(input_text: str, k: int = 3) -> str:
    docs = vectorstore.similarity_search(input_text, k=k)
    examples = "\n".join([f'"{doc.page_content}" → {doc.metadata.get("category", "unknown")}' for doc in docs])

    prompt = PromptTemplate.from_template("""
You are a classification assistant.
Based on the input and examples, return only the best matching category.

Input: {input_text}

Examples:
{examples}

Category:""")
    llm = OllamaLLM(model="gemma:2b")
    chain = {"input_text": RunnablePassthrough(), "examples": lambda x: examples} | prompt | llm
    return chain.invoke(input_text).strip()


# ---- Evaluation ----
async def evaluate_classification_async(test_cases):
    results = []

    for case in test_cases:
        input_text = case["question"]
        ground_truth = case["ground_truth"]
        prediction = classify_text(input_text)
        contexts = [doc.page_content for doc in vectorstore.similarity_search(input_text, k=3)]

        results.append({
            "question": input_text,
            "answer": prediction,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

    dataset = Dataset.from_pandas(pd.DataFrame(results))

    return evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        embeddings=SimpleOllamaEmbeddings(),
        llm=SimpleOllamaLLM(),
    )


def evaluate_classification(test_cases):
    return asyncio.run(evaluate_classification_async(test_cases))


# ---- Example Usage ----
if __name__ == "__main__":
    test_cases = [
        {"question": "I didn't receive my order.", "ground_truth": "shipping"},
        {"question": "They didn't answer my calls.", "ground_truth": "customer_service"},
        {"question": "Broken item.", "ground_truth": "product_quality"},
    ]

    results = evaluate_classification(test_cases)
    print(results)
