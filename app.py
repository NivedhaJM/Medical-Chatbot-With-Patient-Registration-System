# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import os
from googletrans import Translator
import sqlite3
import re

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model2.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

lst = []
app = Flask(__name__)

@app.route("/")
def signup():
    return render_template("signup.html")

@app.route("/signup1")
def signup1():
    return render_template("signup.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/patientregister")
def patientregister():
    return render_template("patientregister.html")

@app.route("/index")
def index():
    print(patient_details_list)
    return render_template("chatbot.html")

database='chatbot1.db'

def createtable():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("create table if not exists register(id integer primary key autoincrement, name text, email text, password text)")
    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patient_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        gender TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        height REAL NOT NULL,
                        weight REAL NOT NULL,
                        address TEXT NOT NULL,
                        city TEXT NOT NULL,
                        state TEXT NOT NULL,
                        pincode INTEGER NOT NULL,
                        phone TEXT NOT NULL,
                        email TEXT NOT NULL,
                        work_type TEXT,
                        blood_group TEXT NOT NULL,
                        medications TEXT NOT NULL,
                        medication_name TEXT,
                        medication_frequency TEXT,
                        dosage TEXT,
                        remarks TEXT
                    )
                    ''')
    cursor.execute("create table if not exists feedback(id integer primary key autoincrement, name text, email text, feedback text)")
    conn.commit()
    conn.close()
createtable()

@app.route('/feedback' , methods=['GET','POST'])
def feedback():
    if request.method == 'POST':
        name = patient_details_list[1]
        email = patient_details_list[12]
        feedback = request.form['feedback']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback(name, email, feedback) values(?,?,?)",(name, email, feedback))
        conn.commit()
        return render_template('login.html')
    return render_template('home.html')



@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email =  request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM register WHERE email=?",(email,))
        registered=cursor.fetchall()
        if registered:
            return render_template('signup.html', error="Your email is already registered.")
        else:
            cursor.execute("INSERT INTO register(name, email, password) values(?,?,?)",(name, email, password))
            conn.commit()
            return render_template('login.html', error="Your email Id registered.")

    return render_template()

# Declare a global list outside the function
patient_details_list = []

@app.route("/logindetails", methods=['GET', 'POST'])
def logindetails():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM register WHERE email=? AND password=?", (email, password))
        data = cursor.fetchone()

        if data is None:
            return render_template('login.html',error="Id Not registered")
        else:
            # Fetch patient details based on the email
            cursor.execute("SELECT * FROM patient_details WHERE email=?", (email,))
            patient_data = cursor.fetchone()
            
            # Update the global patient_details_list
            if patient_data:
                global patient_details_list  # Declare the list as global to modify it
                patient_details_list = list(patient_data)  # Convert tuple to list

                # Print patient details to the backend
                print("Patient Details:", patient_details_list)
            else:
                print("No patient details found for this user.")
            
            conn.commit()
            return render_template('home.html')

    return render_template('login.html')

@app.route('/submit', methods=['POST'])
def submit():
    global patient_details_list  # Use the same global list

    # Create a new list with submitted form data
    submitted_data = [
        request.form['first_name'],
        request.form['last_name'],
        request.form['gender'],
        request.form['age'],
        request.form['height'],
        request.form['weight'],
        request.form['address'],
        request.form['city'],
        request.form['state'],
        request.form['pincode'],
        request.form['phone'],
        request.form['email'],
        request.form.get('work_type', ''),
        request.form['blood_group'],
        request.form['medications'],
        request.form.get('medication_name', ''),
        request.form.get('medication_frequency', ''),
        request.form.get('dosage', ''),
        request.form.get('remarks', '')
    ]

    # Print the submitted data to verify
    print("Submitted Data:", submitted_data)

    # Append the submitted data to the patient_details_list
    patient_details_list.extend(submitted_data)

    # Print the updated patient_details_list
    print("Updated Patient Details List:", patient_details_list)

    # Connect to the database and insert data using the submitted data
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO patient_details (first_name, last_name, gender, age, height, weight, address, city, state, pincode, phone, email, work_type, blood_group, medications, medication_name, medication_frequency, dosage, remarks)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', submitted_data)

    # Commit and close the connection
    conn.commit()
    conn.close()

    # Render the index.html template
    return render_template('home.html')

name_messages = ['my name', 'you know my name']
height_messages = ['my height', 'updated height', 'height', 'hgt']
weight_messages = ['my weight', 'updated weight', 'weight', 'wgt']
medications = ['medications', 'medicine name', 'my medicines']
medications_fry = ['medication frequency', 'medicine frequency', 'my medicines frequency']
medications_dos = ['medication dosage', 'medicine dosages', 'my medicine dosages', 'medicine dose']

@app.route("/get", methods=["POST"])
def get_bot_response():
    msg = request.form["msg"]
    print(msg)
    if msg in name_messages:
        response = patient_details_list[1]
    elif msg in height_messages:
        response = patient_details_list[5]
    elif msg in weight_messages:
        response = patient_details_list[6]
    elif msg in medications:
        response = patient_details_list[15]
    elif msg in medications_fry:
        response = patient_details_list[16]
    elif msg in medications_dos:
        response = patient_details_list[17]
    else:
        response = chatbot_response(msg)

    if isinstance(response, float):
        response = str(response)
    return response


def chatbot_response(msg):
    messg = msg.lower()
    print('messg',messg)
    lst.insert(0, messg)    
    msg = check_symptoms_in_database(messg)
    print('msg',msg)
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

import re

stopwords = {'i', 'have', 'the', 'a', 'and', 'is', 'of', 'to', 'with', 'for', 'in', 'on', 'at', 'by', 'it', 'from'}

def check_symptoms_in_database(symptom):

    symptom_keywords = symptom.lower().split()
    print('symptom_keywords:', symptom_keywords)

    filtered_keywords = [word for word in symptom_keywords if word not in stopwords]
    print('filtered_keywords:', filtered_keywords)

    if 'symptoms' in filtered_keywords:
        target_diseases = ['diabetes', 'hypertension', 'heart attack']

        filtered_diseases = [disease for disease in patient_details_list if isinstance(disease, str) and disease.lower() in target_diseases]

        print("Filtered Diseases:", filtered_diseases)
        if filtered_diseases:
                    value = f" {', '.join(filtered_diseases)} disease and {symptom}"
                    print('check',value)
                    return value
        
        return symptom    
    return symptom

#tokenise the input, normalize the word, lemmatize each word, preprocess the list of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    result = "I'm sorry, I didn't understand that."
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if __name__ == "__main__":
    app.run(port=400)