from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

# ✅ Настройки браузера
options = Options()
options.add_argument("user-data-dir=C:/Users/jolypab/AppData/Local/Google/Chrome/User Data")
options.add_argument("profile-directory=Default")
options.add_argument("--start-maximized")
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")

browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 🗂️ Все категории
categories = [
    "https://century21apolo.com/casas-en-venta/",
    "https://century21apolo.com/departamentos-en-venta/",
    "https://century21apolo.com/terrenos-en-venta/",
    "https://century21apolo.com/venta-de-inmuebles-commerciales/",
    "https://century21apolo.com/edificios-en-venta/",
    "https://century21apolo.com/casas-en-renta/",
    "https://century21apolo.com/departamentos-en-renta/",
    "https://century21apolo.com/terrenos-en-renta/",
    "https://century21apolo.com/renta-de-inmuebles-commerciales/",
    "https://century21apolo.com/edificios-en-renta/"
]

all_links = []

# 🔁 Обход всех категорий
for cat_url in categories:
    print(f"\n=== 📂 Категория: {cat_url} ===")
    page = 1
    while True:
        url = cat_url if page == 1 else f"{cat_url}page/{page}/"
        print(f"🔄 Загружаем: {url}")
        browser.get(url)
        time.sleep(3)

        cards = browser.find_elements(By.XPATH, "//a[contains(@href, '/inmueble/')]")
        links = [
    a.get_attribute("href") 
    for a in cards 
    if a.get_attribute("href") 
    and a.get_attribute("href").startswith("https://century21apolo.com/inmueble/")
]

        if not links:
            print("⛔ Нет объявлений, переходим к следующей категории")
            break

        print(f"✅ Найдено: {len(links)}")
        all_links.extend(links)
        page += 1

# 🧹 Удаление дубликатов
unique_links = list(set(all_links))
print(f"\n📊 Всего уникальных ссылок: {len(unique_links)}")

# 💾 Сохраняем
with open("apolo_all_listings_scraped.json", "w", encoding="utf-8") as f:
    json.dump(unique_links, f, ensure_ascii=False, indent=2)

print("🚀 Ссылки сохранены в apolo_all_listings_scraped.json")
browser.quit()
