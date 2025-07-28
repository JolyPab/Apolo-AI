import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from datetime import datetime
import os
from dotenv import load_dotenv

# --- env --------------------------------------------------------------------
load_dotenv()

# --- embeddings & index -----------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    deployment="text-embedding-ada-002",
    api_version="2023-05-15",
)

INDEX_DIR = "doc_faiss"  # предполагается, что индекс создан заранее
index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# --- llm --------------------------------------------------------------------
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4",
    api_version="2024-02-15-preview",
)

# --- prompt -----------------------------------------------------------------
current_date = datetime.now().strftime("%Y-%m-%d")
TEMPLATE = """
Eres un asistente técnico. Responde únicamente basándote en el contenido del documento proporcionado.
Si la pregunta no está relacionada o la información no se encuentra en el documento, responde educadamente que no dispones de datos.

Fecha actual: {current_date}

Historial del diálogo:
{chat_history}

Contexto del documento:
{context}

Pregunta del usuario: {question}
Respuesta:
"""
PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "current_date", "question"],
    template=TEMPLATE,
).partial(current_date=current_date)

# --- memory & chain ---------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

# --- chat history -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.as_retriever(search_kwargs={"k": 20}),
    memory=st.session_state["memory"],
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

# --- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Asistente Documento", page_icon="📄")
st.title("📄 Asistente del Documento")

# Отображение истории чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода
if prompt := st.chat_input("Pregúntame sobre el documento…"):
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Получаем ответ от AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Показываем индикатор загрузки
        message_placeholder.markdown("🤔 Pensando...")
        
        # Получаем ответ
        try:
            result = qa.invoke({"question": prompt})
            respuesta = result["answer"]
        except Exception as e:
            respuesta = f"❌ Error: {str(e)}\n\nRevisa las claves de Azure OpenAI en los secretos."
        
        # Отображаем ответ
        message_placeholder.markdown(respuesta)
    
    # Добавляем ответ AI в историю
    st.session_state.messages.append({"role": "assistant", "content": respuesta})

# Кнопка очистки истории
if st.sidebar.button("🗑️ Limpiar historial"):
    st.session_state.messages = []
    st.session_state["memory"].clear()
    st.rerun() 