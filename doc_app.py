import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from datetime import datetime
import os
import json
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
IMAGES_DIR = f"{INDEX_DIR}_images"
index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# --- load images info -------------------------------------------------------
images_info = []
images_info_path = os.path.join(IMAGES_DIR, "images_info.json")

if os.path.exists(images_info_path):
    with open(images_info_path, "r", encoding="utf-8") as f:
        images_info = json.load(f)

def get_related_images(question, answer):
    """Определяет, какие изображения связаны с вопросом/ответом"""
    if not images_info:
        return []
    
    text_to_check = (question + " " + answer).lower()
    related_images = []
    
    # Проверяем на точные совпадения сначала
    
    # Проверка на figura 2 (должна быть ПЕРВОЙ, чтобы перехватить точное совпадение)
    if "figura 2" in text_to_check:
        if len(images_info) > 1:
            related_images.append(images_info[1]["filename"])
        return related_images
    
    # Проверка на figura 1
    if "figura 1" in text_to_check:
        related_images.extend([images_info[0]["filename"], images_info[-1]["filename"]])
        return related_images
    
    # Если есть упоминание процедуры GTAW без конкретной фигуры
    if any(word in text_to_check for word in ["gtaw", "proceso", "técnica"]) and "figura" not in text_to_check:
        if len(images_info) > 1:
            related_images.append(images_info[1]["filename"])
        return related_images
    
    # Если есть упоминание биселя без конкретной фигуры
    if any(word in text_to_check for word in ["bisel", "geometría"]) and "figura" not in text_to_check:
        related_images.extend([images_info[0]["filename"], images_info[-1]["filename"]])
        return related_images
    
    # Если спрашивают показать все диаграммы
    if any(word in text_to_check for word in ["todas las figuras", "mostrar todas", "ver todas"]):
        return [img["filename"] for img in images_info]
    
    # Fallback: если AI не может показать, показываем первое изображение
    if any(phrase in answer.lower() for phrase in ["no tengo acceso", "no dispongo", "no puedo mostrar"]):
        if any(word in text_to_check for word in ["figura", "imagen", "diagrama"]):
            related_images.append(images_info[0]["filename"])
    
    return related_images

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
Eres un MAESTRO SOLDADOR EXPERIMENTADO que ayuda a los trabajadores de la planta. Tu trabajo es enseñar y guiar, no solo buscar información.

MENTALIDAD: Eres como un supervisor veterano que conoce todos los procedimientos y ayuda a los nuevos trabajadores.

CUANDO TE PREGUNTEN:
- Explica paso a paso lo que debe hacer el trabajador
- Da consejos prácticos basados en el documento
- Si no tienes detalles exactos, proporciona principios generales de soldadura
- Relaciona la información con la experiencia práctica
- Siempre intenta ayudar y guiar, nunca digas solo "no tengo información"

EJEMPLO DE RESPUESTA BUENA:
"Para la geometría del bisel según Figura 1, debes preparar el material siguiendo estas especificaciones... Aunque no veo todos los detalles específicos, en general para este tipo de trabajo necesitas... Te recomiendo revisar la imagen adjunta y..."

EJEMPLO DE RESPUESTA MALA:
"No tengo información específica sobre la figura."

El documento contiene {num_images} imágenes técnicas que te ayudarán a entender mejor los procedimientos.

Fecha actual: {current_date}

Contexto del documento:
{context}

Historial del diálogo:
{chat_history}

Pregunta del trabajador: {question}

Respuesta del maestro soldador:
"""
PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "current_date", "question", "num_images"],
    template=TEMPLATE,
).partial(current_date=current_date, num_images=len(images_info))

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

# Показать информацию о документе в сайдбаре
with st.sidebar:
    st.markdown("### 📋 Información del documento")
    if images_info:
        st.markdown(f"🖼️ **Imágenes encontradas:** {len(images_info)}")
        
        # Показать превью изображений
        st.markdown("### 🖼️ Imágenes del documento")
        for i, img_info in enumerate(images_info):
            img_path = os.path.join(IMAGES_DIR, img_info["filename"])
            if os.path.exists(img_path):
                # Проверяем, поддерживается ли формат Streamlit
                file_ext = img_info["filename"].lower().split('.')[-1]
                if file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg']:
                    with st.expander(f"Imagen {i+1} ({img_info['size']} bytes)"):
                        try:
                            st.image(img_path, use_container_width=True)
                        except Exception as e:
                            st.error(f"No se puede mostrar la imagen: {img_info['filename']}")
                else:
                    with st.expander(f"Imagen {i+1} ({img_info['size']} bytes) - {file_ext.upper()}"):
                        st.warning(f"Formato {file_ext.upper()} no soportado para vista previa. Archivo: {img_info['filename']}")
    else:
        st.markdown("🖼️ **Imágenes:** No encontradas")
    
    # Кнопка очистки истории
    if st.button("🗑️ Limpiar historial"):
        st.session_state.messages = []
        st.session_state["memory"].clear()
        st.rerun()

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
            
            # Проверяем, нужно ли показать связанные изображения
            related_images = get_related_images(prompt, respuesta)
            
            # Отображаем ответ
            message_placeholder.markdown(respuesta)
            
            # Показываем связанные изображения если есть
            if related_images:
                st.markdown("### 🖼️ Imágenes relacionadas:")
                cols = st.columns(min(len(related_images), 3))  # максимум 3 колонки
                for i, img_filename in enumerate(related_images):
                    img_path = os.path.join(IMAGES_DIR, img_filename)
                    if os.path.exists(img_path):
                        # Проверяем формат файла
                        file_ext = img_filename.lower().split('.')[-1]
                        if file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg']:
                            with cols[i % 3]:
                                try:
                                    st.image(img_path, caption=f"Imagen {i+1}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error mostrando {img_filename}")
                        else:
                            with cols[i % 3]:
                                st.warning(f"Formato {file_ext.upper()} no soportado")
                                st.caption(f"Archivo: {img_filename}")
                            
        except Exception as e:
            respuesta = f"❌ Error: {str(e)}\n\nRevisa las claves de Azure OpenAI en los secretos."
            message_placeholder.markdown(respuesta)
    
    # Добавляем ответ AI в историю
    st.session_state.messages.append({"role": "assistant", "content": respuesta}) 