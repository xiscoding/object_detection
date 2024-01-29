from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import requests
from PIL import Image
from io import BytesIO
import os

# Path to your chromedriver
chromedriver_path = '/home/xdoestech/Desktop/object_detection/chromedriver-linux64/chromedriver'
driver = webdriver.Chrome()

# Search query
search_query = "us traffic stop sign"
driver.get("https://yandex.com/images/")

# Locate and interact with the search box
search_box = driver.find_element(By.NAME, 'text')
search_box.send_keys(search_query)
search_box.send_keys(Keys.RETURN)

# Wait for images to load (adjust time as needed)
time.sleep(5)

# Initialize a list to store image URLs
image_urls = []

try:
    # Use find_elements (plural) to ensure a list is returned
    images = driver.find_elements(By.TAG_NAME, 'img')
except Exception as e:
    print(f"Error finding images: {e}")
    images = []
image_urls = [image.get_attribute('src') for image in images]

# Directory to save images
save_dir = '/home/xdoestech/Desktop/object_detection/scraped_images/stop_sign_2'

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Download and save images
for i, url in enumerate(image_urls):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.save(os.path.join(save_dir, f'image_{i}.jpg'))
    except Exception as e:
        print(f"Error - Could not download image {i}: {e}")

# Close the browser
driver.close()
