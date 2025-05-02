import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,  context_recall,
    answer_similarity, context_entity_recall, answer_correctness
)

from langchain_community.vectorstores import FAISS

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from classify import classify_text

embeddings = OllamaEmbeddings(model="nomic-embed-text")
#df = pd.read_csv("C:/Users/yasmi/Downloads/trustpilot_scraped_dataa.csv").sample(3)
vectorstore = FAISS.load_local("embeddings_data", embeddings, allow_dangerous_deserialization=True)


questions, answers, contexts, ground_truths = [], [], [], []


query = "I didn't receive my order"
true_category = "shipping"
docs = vectorstore.similarity_search(query, k=3)
context_texts = [doc.page_content for doc in docs]
    
examples = "\n".join([f'"{doc.page_content}" → {doc.metadata["category"]}' for doc in docs])
prompt_input = f"""
Input: {query}
    
Examples:
{examples}
    
Category:"""
predicted = classify_text(query)

questions.append(query)
answers.append(predicted)
contexts.append(context_texts)
ground_truths.append(true_category)  # RAGAS expects a list

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})



embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="gemma:2b" , timeout = 60) 

metrics = [
    context_precision,
    context_recall,
    answer_similarity,
    context_entity_recall,
    answer_correctness,
]

score = evaluate(llm=llm, 
                 embeddings=embeddings,
                 dataset=dataset,
                 metrics=metrics,
                 )
print(score)
