import json
import time
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# === Загрузка .env ===
load_dotenv()

# === Конфигурация Azure OpenAI ===
embeddings_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    deployment="text-embedding-ada-002",
    api_version="2023-05-15"
)

# === Загрузка и фильтрация карточек ===
with open("apolo_all_listings_parsed.json", "r", encoding="utf-8") as f:
    listings = json.load(f)

filtered_listings = [
    l for l in listings
    if l.get("description") and l.get("address")
]

# === Индексация FAISS ===
faiss_index = None
metadata = []

for idx, listing in enumerate(filtered_listings, start=1):
    print(f"📌 Генерация embedding {idx}/{len(filtered_listings)}")

    combined_text = (
        f"Título: {listing.get('title', '')}. "
        f"Precio: {listing.get('price', '')}. "
        f"Dirección: {listing.get('address', '')}. "
        f"Descripción: {listing.get('description', '')}. "
        f"Características: {listing.get('features', '')}. "
        f"Agente: {listing.get('agent_name', '')}, "
        f"Email: {listing.get('agent_email', '')}, "
        f"Teléfono: {listing.get('agent_phone', '')}. "
        f"Número de fotos: {len(listing.get('photos', []))}. "
        f"URL: {listing.get('url', '')}"
    )

    embedding = FAISS.from_texts([combined_text], embeddings_model)

    if faiss_index is None:
        faiss_index = embedding
    else:
        faiss_index.merge_from(embedding)

    metadata.append({
        "url": listing.get("url"),
        "title": listing.get("title"),
        "price": listing.get("price"),
        "address": listing.get("address"),
        "agent_name": listing.get("agent_name"),
        "agent_email": listing.get("agent_email"),
        "agent_phone": listing.get("agent_phone"),
        "photos": listing.get("photos", [])
    })

    time.sleep(0.5)

# === Сохраняем результат ===
faiss_index.save_local("apolo_faiss")
with open("apolo_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Embeddings и метаданные успешно сохранены.")