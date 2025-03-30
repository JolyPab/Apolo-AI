from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
from collections import Counter

options = Options()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--headless")

browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(browser, 20)

url = "https://century21mexico.com/v/resultados/oficina_501-apolo_local"

all_links = []

def close_cookie_banner():
    try:
        cookie_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Aceptar')]")))
        cookie_button.click()
        print("✅ Куки-баннер закрыт")
    except:
        print("ℹ️ Куки-баннер не найден")

page = 1
while True:
    full_url = url if page == 1 else f"https://century21mexico.com/v/resultados/pagina_{page}/oficina_501-apolo_local"
    print(f"🔄 Загружаем страницу {page}: {full_url}")
    browser.get(full_url)
    time.sleep(5)
    close_cookie_banner()

    items = browser.find_elements(By.XPATH, "//a[contains(@href, '/propiedad/')]")
    if not items:
        print(f"⛔ Объявления закончились на {full_url}")
        break

    for item in items:
        href = item.get_attribute("href")
        if href:
            all_links.append(href)

    print(f"✅ Найдено {len(items)} объявлений на стр. {page}")
    page += 1

# Анализ результатов
print(f"\n📊 Всего найдено ссылок: {len(all_links)}")

duplicates = [link for link, count in Counter(all_links).items() if count > 1]
print(f"🔁 Дубликатов: {len(duplicates)}")

if duplicates:
    print("\n🔁 Примеры повторяющихся ссылок:")
    for i, link in enumerate(duplicates[:10], 1):
        print(f"{i}. {link}")


unique_links = list(set(all_links))
print(f"✅ Уникальных ссылок: {len(unique_links)}")

with open("cancun_listings_scraped.json", "w", encoding="utf-8") as f:
    json.dump(all_links, f, ensure_ascii=False, indent=2)

print(f"✅ Сохранено {len(all_links)} уникальных ссылок в cancun_listings_scraped.json")

browser.quit()
