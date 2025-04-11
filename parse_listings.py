import json
import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Настройка Selenium с профилем, где включен Hola VPN
options = Options()
options.add_argument("user-data-dir=C:/Users/jolypab/AppData/Local/Google/Chrome/User Data")
options.add_argument("profile-directory=Default")
options.add_argument("--start-maximized")

browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(browser, 15)

# Загрузка URL
with open("apolo_all_listings_scraped.json", "r", encoding="utf-8") as f:
    urls = json.load(f)

parsed = []

# Запросить подтверждение 1 раз
input("⏸ Включи Hola VPN (если нужно) и нажми Enter для начала...\n")

for idx, url in enumerate(tqdm(urls, desc="Парсинг карточек"), start=1):
    try:
        print(f"\n[{idx}/{len(urls)}] 🔗 Загружаем: {url}")
        browser.get(url)
        time.sleep(1)

        title = browser.find_element(By.CSS_SELECTOR, "h1.entry-title.entry-prop").text.strip()

        address = ""
        try:
            address_block = browser.find_element(By.CSS_SELECTOR, "div.property_categs")
            address = address_block.text.strip()
        except:
            pass

        price = ""
        try:
            price = browser.find_element(By.CSS_SELECTOR, ".price_area").text.strip()
        except:
            pass

        # Детали
        features = ""
        try:
                details_block = browser.find_element(By.CSS_SELECTOR, "#accordion_property_details_collapse .panel-body")
                features = details_block.text.strip()
        except:
            if description:
                features = description.strip()

        # Дополняем features текстом из "Otras características"
        try:
            extra_block = browser.find_element(By.CSS_SELECTOR, "#accordion_features_details_collapse .panel-body")
            extra_text = extra_block.text.strip()
            if extra_text:
                features += "\n\n" + extra_text
        except:
            pass


         # Описание
        try:
            desc_block = browser.find_element(By.CSS_SELECTOR, "#wpestate_property_description_section")
            description = desc_block.text.strip()
        except:
            description = ""

        photos = []
        # Из background-image
        for div in browser.find_elements(By.CSS_SELECTOR, "div.item"):
            style = div.get_attribute("style")
            if style and "background-image" in style:
                try:
                    url_part = style.split("url(")[1].split(")")[0].strip('"\'')

                    if url_part and url_part not in photos:
                        photos.append(url_part)

                except IndexError:
                    continue





        # Агент
        agent_name = ""
        agent_phone = ""
        agent_email = ""

        try:
            agent_block = browser.find_element(By.CLASS_NAME, "agent_unit_widget_sidebar_details_wrapper")
            agent_name = agent_block.find_element(By.TAG_NAME, "h4").text.strip()
            agent_url = agent_block.find_element(By.TAG_NAME, "a").get_attribute("href")

            if agent_url:
                browser.get(agent_url)
                time.sleep(2)

                try:
                    # Имя
                    try:
                        name_el = browser.find_element(By.CSS_SELECTOR, "h3 a")
                        agent_name = name_el.text.strip()
                    except:
                        pass

                    # 📱 Мобильный телефон
                    try:
                        phone_mobile = browser.find_element(By.CSS_SELECTOR, ".agent_mobile_class a[href^='tel']")
                        agent_phone = phone_mobile.get_attribute("href").replace("tel:", "").strip()
                    except:
                        # ☎️ Если мобилки нет — пробуем офис
                        try:
                            phone_office = browser.find_element(By.CSS_SELECTOR, ".agent_phone_class a[href^='tel']")
                            agent_phone = phone_office.get_attribute("href").replace("tel:", "").strip()
                        except:
                            pass

                    # ✉️ Email
                    try:
                        email_el = browser.find_element(By.CSS_SELECTOR, ".agent_email_class a[href^='mailto']")
                        agent_email = email_el.get_attribute("href").replace("mailto:", "").strip()
                    except:
                        pass

                except Exception as e:
                    print("⚠️ Ошибка при парсинге агента:", e)

                browser.back()

        except:
            pass



        parsed.append({
            "url": url,
            "title": title,
            "price": price,
            "address": address,
            "features": features,
            "description": description,
            "agent_name": agent_name,
            "agent_phone": agent_phone,
            "agent_email": agent_email,
            "photos": photos
        })

    except Exception as e:
        print(f"⚠️ Ошибка при парсинге {url}: {e}")

browser.quit()

# Сохраняем
with open("apolo_all_listings_parsed.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, ensure_ascii=False, indent=2)

print("\n✅ Парсинг завершён. Сохранено:", len(parsed), "объектов")
