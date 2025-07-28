#!/usr/bin/env python3
"""quick_test.py – мини-тест работы индекса doc_faiss и модели GPT-4 (Azure).
Запускается командой:
    python quick_test.py "¿Cuál es el par de apriete recomendado?"
Если вопрос не указан, берётся дефолтный.
"""
import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

QUESTION = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "¿Cuál es el par de apriete recomendado?"
)

# --- embeddings & index ------------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    deployment="text-embedding-ada-002",
    api_version="2023-05-15",
)
index = FAISS.load_local("doc_faiss", embeddings, allow_dangerous_deserialization=True)

# --- llm --------------------------------------------------------------------
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4",
    api_version="2024-02-15-preview",
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.as_retriever(search_kwargs={"k": 20}),
)

print("Pregunta:", QUESTION)
print("Respuesta:")
result = chain.invoke({"question": QUESTION, "chat_history": []})
print(result["answer"]) 