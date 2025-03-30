from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time
import os

# Настройки браузера
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")


def parse_listing(url, index=None):
    browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        browser.get(url)
        time.sleep(4)

        name = browser.find_element(By.CSS_SELECTOR, "h5.card-title").text.strip()
        id_block = browser.find_element(By.CSS_SELECTOR, "div.col-sm-12.h5").text.strip()
        listing_id = id_block.split("ID:")[-1].strip()

        price = browser.find_element(By.CSS_SELECTOR, "h6.text-muted.fw-bold").text.strip()
        address = browser.find_element(By.CSS_SELECTOR, "h6.small.mb-3.text-muted").text.strip()

        try:
            description = browser.find_element(By.CSS_SELECTOR, "p.text-muted[style*='white-space']").text.strip()
        except:
            description = "нет данных"

        features = [
            f.text.strip() for f in browser.find_elements(By.CSS_SELECTOR, "div.col-sm-12.col-md-6.col-lg-4.my-2")
        ]

        images = [img.get_attribute("src") for img in browser.find_elements(By.CSS_SELECTOR, "div.row.gx-1.gy-1 img")]

        return {
            "url": url,
            "name": name,
            "id": listing_id,
            "price": price,
            "address": address,
            "description": description,
            "features": features,
            "images": images
        }

    except Exception as e:
        # Сохраняем HTML при ошибке
        if index is not None:
            with open(f"error_page_{index}.html", "w", encoding="utf-8") as f:
                f.write(browser.page_source)
        print(f"❌ Ошибка на {url}: {e}")
        return None
    finally:
        browser.quit()

def main():
    with open("cancun_listings_scraped.json", "r", encoding="utf-8") as f:
        links = json.load(f)

    listings = []
    max_workers = 8  # не ставь много, иначе браузеры съедят всю память

    print(f"🚀 Начинаем парсинг {len(links)} объектов...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(parse_listing, url, idx): url for idx, url in enumerate(links)}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            result = future.result()
            if result:
                listings.append(result)

            # Каждые 10 объектов — автосейв
            if i % 10 == 0 and listings:
                with open("cancun_listings.json", "w", encoding="utf-8") as f:
                    json.dump(listings, f, ensure_ascii=False, indent=2)

    with open("cancun_listings_parsed.json", "w", encoding="utf-8") as f:
        json.dump(listings, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Готово! Спарсено: {len(listings)} объектов.")

if __name__ == "__main__":
    main()
