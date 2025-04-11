import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
import json
from langchain.prompts import PromptTemplate
from datetime import datetime
import re
from twilio.rest import Client

# === Twilio Sandbox конфигурация ===

TWILIO_ACCOUNT_SID=st.secrets["TWILIO_SID"]
TWILIO_AUTH_TOKEN=st.secrets["TWILIO_TOKEN"]
TWILIO_WHATSAPP_SANDBOX=st.secrets["WHATSAPP_SANDBOX"]

# ✅ Кому отправлять (временно только себе)
AGENT_WHATSAPP_NUMBERS = [
    "whatsapp:+79110057195"
]

def enviar_whatsapp_agentes(mensaje):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    for numero in AGENT_WHATSAPP_NUMBERS:
        client.messages.create(
            from_=TWILIO_WHATSAPP_SANDBOX,
            body=mensaje,
            to=numero
        )

# Получаем текущую дату
current_date = datetime.now().strftime("%Y-%m-%d")

# === Конфигурация Azure ===
embeddings = AzureOpenAIEmbeddings(
    api_key=st.secrets["AZURE_EMBEDDINGS_API_KEY"],
    azure_endpoint=st.secrets["AZURE_EMBEDDINGS_ENDPOINT"],
    deployment="text-embedding-ada-002",
    api_version="2023-05-15"
)

llm = AzureChatOpenAI(
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4",
    api_version="2024-02-15-preview"
)

# === PROMPT ===
template = """
Eres un asistente virtual para la selección de bienes raíces. Tu tarea es ayudar al cliente a elegir una propiedad que se ajuste lo máximo posible a sus deseos y necesidades.

Fecha actual: {current_date}

Tus responsabilidades:
- Asegúrate de verificar la fecha actual ({current_date}) al ofrecer información, especialmente en casos de propiedades en renta o eventos limitados en el tiempo.
- Mantén una conversación profesional y amigable, como un agente inmobiliario experimentado.
- Pregunta al cliente detalles importantes: presupuesto, ubicación, tipo de propiedad, cantidad de habitaciones, características de infraestructura, preferencias de estilo y cualquier otro requisito adicional.
- Recuerda las preferencias del cliente y tómalas en cuenta en futuras recomendaciones.
- Si el cliente pregunta sobre una propiedad específica, proporciona una descripción detallada, incluyendo el precio, si está disponible.
- Si el precio no está disponible, informa claramente sobre ello y ofrece una alternativa con precio conocido o pide al cliente que precise sus preferencias.
- Responde exclusivamente con base en la información proporcionada, sin inventar detalles adicionales.
- Si la información es insuficiente o poco clara, formula preguntas aclaratorias.
- Actúa proactivamente, ofreciendo alternativas y recomendaciones que puedan interesar al cliente, basadas en sus solicitudes previas.

⚠️ Si el cliente demuestra un interés claro en una propiedad (por ejemplo, expresa "me interesa", "quiero agendar", o comparte su nombre, teléfono o email),
PERO no ha proporcionado nombre, teléfono o email, ENTONCES solicita esos datos explícitamente.
SOLO cuando el cliente haya mostrado un interés claro Y haya proporcionado al menos un dato de contacto (nombre, teléfono o email), responde con el siguiente JSON...
{{
  "lead_detected": true,
  "nombre": "Nombre del cliente (si lo proporciona, si no deja vacío)",
  "telefono": "Número del cliente (si lo proporciona, si no deja vacío)",
  "email": "Email del cliente (si lo proporciona, si no deja vacío)",
  "mensaje": "Texto breve del interés del cliente en la propiedad"
}}

Historial del diálogo:
{chat_history}

Contexto inmobiliario:
{context}

Pregunta del cliente: {question}
Respuesta del asistente inmobiliario:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "current_date", "question"],
    template=template
).partial(current_date=current_date)

# === FAISS и метаданные ===
index = FAISS.load_local("apolo_faiss", embeddings, allow_dangerous_deserialization=True)
with open("apolo_metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

# === Streamlit session memory ===
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.as_retriever(search_kwargs={"k": 10}),
    memory=st.session_state["memory"],
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# === Streamlit UI ===
st.set_page_config(page_title="IA Asistente de inmobiliaria", page_icon="🏖️")
st.sidebar.markdown("# 🏖️ Apolo IA")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding-bottom: 60px;
}
[data-testid="stSidebar"] {
    background-color: transparent !important;
}
div.stTextInput {
    position: fixed !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 60% !important;
    background-color: #262730 !important;
    padding: 10px !important;
    border-radius: 10px !important;
    z-index: 1000;
}
[data-testid="stVerticalBlock"] {
    flex-grow: 1;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

content_container = st.container()
query = st.chat_input("Qué quieres saber?")

if query:
    result = qa({"question": query})
    respuesta = result["answer"]

    with content_container:
        st.subheader("🏡  Respuesta de la IA:")
        st.write(respuesta)

        # 🔍 Попытка найти JSON вручную в тексте ответа
        json_match = re.search(r'\{.*?"lead_detected"\s*:\s*true.*?\}', respuesta, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if parsed.get("lead_detected"):
                    mensaje_agente = f"📞 *Nuevo cliente interesado*:\n\n"
                    mensaje_agente += f"🧑 Nombre: {parsed.get('nombre')}\n"
                    mensaje_agente += f"📱 Teléfono: {parsed.get('telefono')}\n"
                    mensaje_agente += f"📧 Email: {parsed.get('email')}\n"
                    mensaje_agente += f"💬 Mensaje: {parsed.get('mensaje')}\n"
                    mensaje_agente += f"🕑 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    enviar_whatsapp_agentes(mensaje_agente)
                    st.success("📢 ¡Datos del cliente enviados al agente por WhatsApp!")
            except json.JSONDecodeError:
                st.error("❌ Ошибка при разборе JSON-ответа")

        # === Показ фото (если объект упоминается) ===
        coincidencia = next((item for item in metadata if item.get("url") in respuesta), None)
        if coincidencia and "photos" in coincidencia:
            st.subheader("📷 Fotos del inmueble:")
            for foto in coincidencia["photos"][:5]:
                st.image(foto, use_container_width=True)

        # === История диалога ===
        with st.expander("💬 Historia del diálogo"):
            for message in st.session_state["memory"].chat_memory.messages:
                role = "Tú" if message.type == "human" else "AI"
                st.markdown(f"**{role}:** {message.content}")
