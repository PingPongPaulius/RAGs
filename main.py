from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
import spacy
import numpy as np
import glob

load_dotenv()

def get_documents(path: str = "Data") -> list[Document]:

    names = glob.glob(f"{path}/*.pdf")
    nlp = spacy.load("en_core_web_sm")

    text = []
    for name in names:
        loader = PyPDFLoader(name)
        for page in loader.lazy_load():
            splitt = nlp(page.page_content)
            sentences = [sent.text.strip() for sent in splitt.sents]
            for txt in sentences:
                if len(txt) >= 3:
                    text.append(txt)

    return text

def get_BM25(documents, query, LLM_importance_proportion):
    BM25 = BM25Okapi(documents)
    return torch.tensor(BM25.get_scores(query)) * (1 - LLM_importance_proportion)


def get_Model(documents, query, LLM_importance_proportion):
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Runs okay on CPU
    doc_embed = model.encode_document(documents, convert_to_tensor=True)
    query_embed = model.encode(query, convert_to_tensor=True)
    return model.similarity(query_embed, doc_embed)[0] * (LLM_importance_proportion)

def query_RAG():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",
    )

    chat_model = ChatHuggingFace(llm=llm)
    messages = [
        SystemMessage(content="You are a Senior Chinese AI Engineer"),
        HumanMessage(
            content="What type of LLM Projects would you recommend for someone who can not run embedding models due to weak GPU?"
        ),
    ]

    ai_msg = chat_model.invoke(messages)
    print(ai_msg.content)


def main():
    documents = get_documents()

    k = 3
    LLM_importance_proportion = 0.75
    query = "van Edme Boas"

    BM25_scores = get_BM25(documents, query, LLM_importance_proportion)
    similarity_scores = get_Model(documents, query, LLM_importance_proportion)

    hybrid_scores = similarity_scores + BM25_scores
    scores, indices = torch.topk(hybrid_scores, k)

    for s, i in zip(scores, indices):
        print(f"(Score: {s:.4f})", documents[i])

    #query_RAG()
    

if __name__ == "__main__":
    main()
