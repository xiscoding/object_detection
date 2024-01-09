from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import requests
from PIL import Image
from io import BytesIO
import os

chromedriver_path ='/home/xdoestech/Desktop/object_detection/chromedriver-linux64/chromedriver'
# Example for Chrome; replace with your browser and driver path if different
driver = webdriver.Chrome()

search_query = "red triangle yield sign"
driver.get("https://www.google.com/imghp?hl=en")
search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys(search_query)
search_box.send_keys(Keys.RETURN)

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)  # Wait to load page
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

try:
    # Use find_elements (plural) to ensure a list is returned
    images = driver.find_elements(By.TAG_NAME, 'img')
except Exception as e:
    print(f"Error finding images: {e}")
    images = []
image_urls = [image.get_attribute('src') for image in images]

for i, url in enumerate(image_urls):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.save(os.path.join('/home/xdoestech/Desktop/object_detection/scraped_images/yield_sign_3', f'image_{i}.jpg'))
    except Exception as e:
        print(f"Error - Could not download image {i}: {e}")

driver.close()
