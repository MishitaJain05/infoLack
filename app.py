from flask import Flask, render_template, request, redirect, url_for,jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import nltk
from nltk import WordNetLemmatizer
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import xgboost
from xgboost import XGBClassifier
from joblib import dump, load
import joblib
import csv
import pandas as pd
import numpy as np
import re
import requests
import time
from selenium.common.exceptions import NoSuchElementException
import random
from flask_pymongo import PyMongo
from bson import Binary
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer





app = Flask(__name__)
app.config["MONGO_URI"]="mongodb://localhost:27017/myDatabase"
db= PyMongo(app).db






def process_link(link):
    options = Options()
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(link)
    return driver

def is_url_scrapable(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        if response.status_code == 200:
            return True
        else:
            return False

    except requests.exceptions.RequestException as e:
        return False


def review_scrap(driver):
    rev_all=""
    i=0
    nxt= driver.find_element("xpath","/html/body/div[4]/div/section[1]/section[2]/section/div[5]/div[2]/nav/button[2]")
    while i<3:
      reviews= driver.find_element("xpath","/html/body/div[4]/div/section[1]/section[2]/section/table/tbody")
      rev_all+=reviews.text
      time.sleep(3)
      if not nxt.is_enabled():
          break
      nxt.click()
      i+=1
    return rev_all

def remove_text_occurrences(text, to_remove):
    return text.replace(to_remove, '')

def remove_lines_with_star(input_text):
    pattern = re.compile('.*[*$#].*')
    lines = input_text.split('\n')
    filtered_lines = [line for line in lines if not pattern.match(line)]
    result_text = '\n'.join(filtered_lines)
    return result_text

def remove_specific_lines(input_text, phrases_to_remove):
    lines = input_text.split('\n')
    filtered_lines = [line for line in lines if not any(phrase in line for phrase in phrases_to_remove)]
    result_text = '\n'.join(filtered_lines)
    return result_text

def remove_lines_and_next(input_text):
    lines = input_text.split('\n')
    filtered_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if "Reply from" in line:
            # Skip the current line and the next line
            i += 2
        else:
            filtered_lines.append(line)
            i += 1

    result_text = '\n'.join(filtered_lines)
    return result_text



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

final_rev=""
product_name = ""
product_catalog=""
link=""
count=0
f=0
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    global final_rev,product_name,product_catalog,link,count,f
    if request.method=='POST':
      

    # scraping
    
      link= request.form['link']
      query ={"url":link}
      count= int(db.inventory.count_documents(query))



    
      driver= process_link(link)
      driver.get(link)
      time.sleep(3)
      
      final_rev=""
      product_name = ""
      
      product_catalog=""

      try:
        product_name = driver.find_element("xpath", "/html/body/div[2]/main/div[1]/div[1]/div[3]/div/div/div[2]/div/div/div[1]/h1/span").text
      except NoSuchElementException:
        product_name = ""

      print(product_name)


      try:
          product_catalog = driver.find_element("xpath", "/html/body").text
      except NoSuchElementException:
          product_catalog =  ""


    
      try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.find_element("xpath","/html/body/div[2]/main/div[1]/div[1]/div[4]/div[2]/div/div[1]/div/div[3]/div[2]/div/div/div[2]/div[1]/div/div/a").click()
        reviews= review_scrap(driver)
        time.sleep(3)

        card_item= driver.find_element("xpath","/html/body/div[4]/div/section[1]/section[2]/section/table/tbody/tr[2]/td[1]/div/div/div[2]/div[2]")      
        final_rev= remove_text_occurrences(reviews,card_item.text)

      except NoSuchElementException:   
        pass

      
      

    # scraping end
      

    #   -----------------------------------------------------------------------------------------------------#
    try:
        output_text = remove_lines_with_star(final_rev)
        phrases_to_remove = ["Verified purchase", "Past year", "More than a year ago", "Past 6 months", "Past month"]

        cleaned_output_text = remove_specific_lines(output_text, phrases_to_remove)
        clean = remove_lines_and_next(cleaned_output_text)
        clean_list = clean.split("\n")
        filtered_list = [item for item in clean_list if item]
        sia = SentimentIntensityAnalyzer()
        score = 0
        for sentence in filtered_list:
            score += sia.polarity_scores(sentence)['compound']

        final_score = ((score/len(filtered_list)) * 50) + 50
    
    except:
        final_score=0

  


      

    # #   model start
    tfidf_vectorizer = joblib.load("final_vectorizer.joblib")
    xgb = joblib.load("final_category.joblib")
    txt = product_name

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def remove_punct(text):
        if isinstance(text, str):
            return "".join([ch for ch in text if ch not in string.punctuation])
        else:
            return str(text)
    txt = remove_punct(txt)


    # Tokenization and lowercase
    tokenized = word_tokenize(txt)
    lowercase = [word.lower() for word in tokenized]

    # Remove stopwords
    stop = set(stopwords.words('english'))
    stopwords_removed = [word for word in lowercase if word not in stop]

    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(stopwords_removed)

    # Get WordNet POS tags
    wordnet_pos = [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos_tags]

    # Lemmatization
    wnl = WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(word, tag) for word, tag in wordnet_pos]
    lemmatized = [word for word in lemmatized if word not in stop]

    # Convert lemmatized list to string
    lemma_str = ' '.join(lemmatized)


    X = tfidf_vectorizer.transform([lemma_str])

    predicted_class = xgb.predict(X)[0]

    Class_Mapping = {0: 'Beauty & Personal Care', 1: 'Electronics', 2: 'Fashion', 3: 'household'}

    info = product_catalog.split(" ")
    for i in range(0,  len(info)):
        info[i] = info[i].lower()
    

    new1 = []
    for s in info:
        new1.append(s.replace(" ", "_"))

    required = ["warranty", "guide", "expiration", "color", "modified", "material", "size", "effect"]


    def syno_for_word(word):
        synonyms = set()

        for syn in wordnet.synsets(word):
            for i in syn.lemmas():
                synonyms.add(i.name())
        return list(synonyms)

    syno_of_words=[]
    for word in required :
        syno_of_words.append(syno_for_word(word))

    warranty = -1
    brand = -1
    manual_guide = -1
    og = -1
    expiration_date = -1
    effect = -1
    modified_item = -1
    material = -1
    model = -1
    color = -1
    size_type=-1
    condition=-1


    for i in range(0, len(new1)):
        if ("brand" == new1[i]):
            brand = 1
        if ("model" == new1[i]):
            model = 1
        if ("unbranded" == new1[i]):
            brand = -1
        if ("condition" == new1[i]):
            condition = 1

        if predicted_class == 1:
            if new1[i] in syno_of_words[0] or new1[i] == "manufacturer_warranty":
                warranty = 1
            if new1[i] == "manual_guide" or new1[i] in syno_of_words[1]:
                manual_guide = 1
            if new1[i] == "with_original_box/packaging" or new1[i] == "with_papers":
                og = 1
            
        

        if predicted_class == 3:
            if info[i] == "expiration_date" or info[i] in syno_of_words[2]:
                expiration_date = 1

        if predicted_class == 0:
            if new1[i] == "effect" or info[i] in syno_of_words[7]:
                effect = 1
            if (new1[i] == ("color" or "shade")) or info[i] in syno_of_words[3]:
                color = 1
            if new1[i] == "modified_item" or info[i] in syno_of_words[4]:
                modified_item = 1

        if predicted_class == 2:
            if new1[i] == "material" or info[i] in syno_of_words[5]:
                material = 1
            if new1[i] == "size_type" or info[i] in syno_of_words[6]:
                size_type = 1
            if new1[i] in syno_of_words[0] or new1[i] == "manufacturer_warranty" or new1[i] == "warranty":
                warranty = 1



    if predicted_class == 1:

        condition_weight = 0.2
        warranty_weight = 0.15
        brand_weight = 0.15
        manual_guide_weight = 0.15
        original_boxing_weight = 0.15
        model_number_weight = 0.2

        overall_score = (
            condition * condition_weight +
            warranty * warranty_weight +
            brand * brand_weight +
            manual_guide * manual_guide_weight +
            og * original_boxing_weight +
            model * model_number_weight
        )
    
    if predicted_class == 0:
        
        effect_weight = 0.4
        color_weight = 0.3
        modified_weight = 0.3
        overall_score = (
        effect * effect_weight +
        color * color_weight +
        modified_item * modified_weight
        )


    if predicted_class == 2:
        condition_weight = 0.3
        warranty_weight = 0.15
        brand_weight = 0.2
        material_weight = 0.15
        size_type_weight = 0.2

        if(warranty==-1):
            warranty=0
        overall_score = (
        material * material_weight +
        size_type * size_type_weight
        +brand*brand_weight+
        material*material_weight
        +size_type*size_type_weight
        )
        

    
    if predicted_class == 3:
        warranty_weight = 0.6
        expiry_date_weight = 0.4

        overall_score = (
        warranty * warranty_weight +
        expiration_date * expiry_date_weight
        )

    if not final_score:
        answer= (0.7* overall_score*50)+50+(0.3*final_score)
    else:
        answer= overall_score*50+50

    if(count>10 and count<50):
        f= 0.2*count
    if(count>50 and count<100):
        f=0.3*count
    else:
        f=0.4*count

    answer= answer-f
    if(answer<0):
        answer=10
        

    #   model end
        
    schema = {
    'link': link,
    'warranty': warranty,
    'brand': brand,
    'manual_guide': manual_guide,
    'og': og,
    'expiration_date': expiration_date,
    'effect': effect,
    'modified_item': modified_item,
    'material': material,
    'model': model,
    'color': color,
    'size_type': size_type,
    'condition': condition
    }       
    
    db.seller_db.insert_one(schema)


        

    
      

      


    # answer= random.randint(0, 100)

    return render_template("analysis.html",answer=answer)


    return render_template("analysis.html")


@app.route('/report', methods=['POST'])
def report():

    if request.method=='POST':
        name= request.form['fname']
        email= request.form['mail']
        url= request.form['url']
        desc= request.form['dark_pattern_noticed']
        file = request.files["image"]

        file_content = file.read()

        image_data = {"image": Binary(file_content)}


        data = {
        "name": name,
        "email": email,
        "url": url,
        "description": desc,
        "attached_file": image_data
        }

        query ={"url":url}
        count= db.inventory.count_documents(query)
        db.inventory.insert_one(data)

        
       

    return render_template("index.html",count=count)

    


    

if __name__ == '__main__':
    app.run(debug=True)
