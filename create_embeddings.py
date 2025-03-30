import json
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import time
import os
from dotenv import load_dotenv
load_dotenv()

# === Конфигурация Azure OpenAI ===
embeddings_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    deployment="text-embedding-ada-002",
    api_version="2023-05-15"
)

# === Загрузка и фильтрация данных ===
with open("cancun_listings.json", "r", encoding="utf-8") as f:
    listings = json.load(f)

# Фильтрация пустых объектов
filtered_listings = [
    listing for listing in listings 
    if listing.get('description') != 'нет данных' and listing.get('address') != 'нет данных'
]

faiss_index = None
metadata = []

# === Генерация embeddings по одному с задержкой ===
for idx, listing in enumerate(filtered_listings, start=1):
    print(f"🔍 Генерируем embedding {idx}/{len(filtered_listings)}")
    
    combined_text = f"{listing['name']}. {listing['price']}. {listing['id']}. {listing['address']}. {listing['description']}. {' '.join(listing['features'])}"
    
    embedding_current = FAISS.from_texts([combined_text], embeddings_model)

    if faiss_index is None:
        faiss_index = embedding_current
    else:
        faiss_index.merge_from(embedding_current)
    
    metadata.append({
        "url": listing["url"],
        "price": listing["price"],
        "address": listing["address"]
    })
    
    time.sleep(0.5)  # задержка полсекунды между запросами

# === Сохраняем FAISS-индекс ===
faiss_index.save_local("cancun_faiss")

# === Сохраняем метаданные ===
with open("cancun_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("🚀 FAISS-индекс и метаданные успешно сохранены!")
