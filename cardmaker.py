import re
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib  
options = Options()
options.headless = True
import json    
def stats_taking(player):
    global data  # Ensure we're modifying the global data variable
    data = {}  # Reset data for a single player
    data[player] = {}  # Initialize player dictionary
    driver = webdriver.Chrome(options=options)
    options.add_argument('--headless')
    driver.get('https://stats.espncricinfo.com/ci/engine/stats/index.html')
    driver.maximize_window()

    search_box = driver.find_element(By.NAME, "search")
    search_box.send_keys(player.strip())
    search_box.send_keys(Keys.RETURN)

    link = driver.find_element(By.XPATH, "//a[starts-with(text(), 'Players and Officials')]")
    link.click()

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "gurusearch_player")))
    table = driver.find_element(By.ID, "gurusearch_player")

    max_matches = 0
    link_to_click = None
    rows = table.find_elements(By.XPATH, ".//table/tbody/tr[@valign='top']")
    
    for row in rows:
        try:
            match_links = row.find_elements(By.XPATH, ".//td[3]/a[contains(text(), 'Twenty20 matches player')]")
            
            for link in match_links:
                parent_text = link.find_element(By.XPATH, "./..").text
                match = re.search(r"(\d+) matches", parent_text)

                if match:
                    matches_count = int(match.group(1))
                    if matches_count > max_matches:
                        max_matches = matches_count
                        data[player]['matches'] = matches_count 
                        link_to_click = link
        except Exception:
            continue

    try:
        link_to_click.click()
    except Exception as e:
        print("Player does not exist! Please try another player.")
        driver.quit()

    menu_url = driver.current_url
    try:
        player_name = driver.find_element(By.CSS_SELECTOR, "div.ciPhotoContainer p b").text
        data[player]['name'] = player_name
    except Exception as e:
        print("Error:", e)
    # Bowling stats
    radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='bowling']")
    radio_button.click()
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
    submit_button.click()

    row = driver.find_element(By.XPATH, "//tr[@class='data1']")
    cells = row.find_elements(By.TAG_NAME, "td")

    try:
        data[player]['wickets'] = int(cells[7].text)
    except Exception:
        data[player]['wickets'] = None
    try:
        data[player]['bowling_average'] = float(cells[9].text)
    except Exception:
        data[player]['bowling_average'] = None
    try:
        data[player]['economy_rate'] = float(cells[10].text)
    except Exception:
        data[player]['economy_rate'] = None

    driver.get(menu_url)

    # Batting stats
    radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='batting']")
    radio_button.click()
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
    submit_button.click()

    table_row = driver.find_element(By.CLASS_NAME, "data1")
    cells = table_row.find_elements(By.TAG_NAME, "td")

    try:
        data[player]['runs_made'] = int(cells[5].text)
    except Exception:
        data[player]['runs_made'] = None
    try:
        data[player]['batting_average'] = float(cells[7].text)
    except Exception:
        data[player]['batting_average'] = None
    try:
        data[player]['strike_rate'] = float(cells[9].text)
    except Exception:
        data[player]['strike_rate'] = None

    driver.quit()

    with open(f"/Users/jatin/Documents/python/cricket attax/dummy.json", "w") as f:
        json.dump(data, f, indent=4)

player = input("Enter player name: ")
stats_taking(player=player)


scaler = joblib.load("scaler.pkl")  # Ensure this file exists

model = keras.models.load_model("player_performance_model.keras", compile=False)

#[matches, runs_made, strike_rate, batting_avg, bowling_avg, wickets, economy_rate]
with open("/Users/jatin/Documents/python/cricket attax/dummy.json") as f:
    data = json.load(f)

new_player = np.array([[data[player]['matches'],data[player]['runs_made'], data[player]['strike_rate'],data[player]['batting_average'], data[player]['bowling_average'],data[player]['wickets'], data[player]['economy_rate']]])
print(new_player)

# Scale the input
new_player_scaled = scaler.transform(new_player)

# Predict batting, runs, and bowling
predicted_values = model.predict(new_player_scaled)
predicted_values[0] = np.array([int(x.item()) for x in predicted_values[0]])
if predicted_values[0][0] > 101:
    predicted_values[0][0] = 101
if predicted_values[0][1] > 101:
    predicted_values[0][1] = 101
print(f"Batting: {predicted_values[0][0]}, Runs: {predicted_values[0][1]}, Bowling: {predicted_values[0][2]}")

