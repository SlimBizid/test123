from datasets import Dataset
import pandas as pd

#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas import evaluate


llm = Ollama(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")


vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)

def classify_text(input_text, k=3, return_docs=False):
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
    result = chain.invoke({
        "input_text": input_text,
        "examples": examples
    })  # result is a string, not a dict

    if return_docs:
        return result.strip(), [doc.page_content for doc in retrieved_docs]
    return result.strip()


df = pd.read_csv("D:/Downloads/trustpilot_scraped_dataa.csv").sample(3)

ragas_data = []
for _, row in df.iterrows():
    input_text = row["description"]
    true_category = row["category"]
    predicted, contexts = classify_text(input_text, return_docs=True)

    ragas_data.append({
        "question": input_text,
        "answer": predicted,
        "ground_truths": [true_category],
        "reference": true_category,
        "contexts": contexts

    })


dataset = Dataset.from_list(ragas_data)


results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness],
    llm=llm,
    embeddings=embeddings  
)

print("📊 RAGAS Evaluation Metrics:")
print(results)