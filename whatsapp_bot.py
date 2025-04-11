from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
import json
import re
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# === Flask App ===
app = Flask(__name__)

# === Twilio config ===
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")  # e.g. "whatsapp:+14155238886"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# === Load FAISS and metadata ===
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    deployment="text-embedding-ada-002",
    api_version="2023-05-15"
)

index = FAISS.load_local("apolo_faiss", embeddings, allow_dangerous_deserialization=True)

# === Prompt ===
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
- Evita comenzar cada mensaje con “Hola [nombre]” si la conversación ya ha comenzado.
- No incluyas firmas como “[Nombre del Asistente]” al final de los mensajes.

⚠️ Si el cliente demuestra un interés claro en una propiedad (por ejemplo, expresa "me interesa", "quiero agendar", o comparte su nombre, teléfono o email),
PERO no ha proporcionado nombre, teléfono o email, ENTONCES solicita esos datos explícitamente.
SOLO cuando el cliente haya mostrado un interés claro Y haya proporcionado al menos un dato de contacto (nombre, teléfono o email), responde con el siguiente JSON, sin ningún texto adicional fuera del bloque:

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
current_date = datetime.now().strftime("%Y-%m-%d")
PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "current_date", "question"],
    template=template
).partial(current_date=current_date)

# === Chatbot setup ===
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4",
    api_version="2024-02-15-preview"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)


# === Ruta para recibir mensajes de WhatsApp ===
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

def enviar_whatsapp_agentes(mensaje):
    client = Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )

    numeros_agentes = [
        "whatsapp:+79110057195",
        "whatsapp:+5219982363432"
    ]

    for numero in numeros_agentes:
        client.messages.create(
            from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
            to=numero,
            body=mensaje
        )


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")
    
    # логируем
    print(f"[CLIENT] {sender} => {incoming_msg}")
    
    # обрабатываем через LangChain
    result = qa({"question": incoming_msg})
    respuesta = result["answer"]
    print("=== LLM Response ===")
    print(respuesta)

    
    # создаём ответ
    twilio_response = MessagingResponse()
    twilio_response.message(respuesta)

    # если это лид — шлём агентам
    match = re.search(r'\{.*?"lead_detected"\s*:\s*true.*?\}', respuesta, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            mensaje_agente = f"📞 *Nuevo cliente interesado*:\n\n"
            mensaje_agente += f"🧑 Nombre: {parsed.get('nombre')}\n"
            mensaje_agente += f"📱 Teléfono: {parsed.get('telefono')}\n"
            mensaje_agente += f"📧 Email: {parsed.get('email')}\n"
            mensaje_agente += f"💬 Mensaje: {parsed.get('mensaje')}\n"
            mensaje_agente += f"🕑 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            enviar_whatsapp_agentes(mensaje_agente)
        except Exception as e:
            print("❌ Error al procesar lead:", e)

    return str(twilio_response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
