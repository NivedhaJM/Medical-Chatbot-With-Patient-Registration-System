# ===== IMPORTS =====
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, redirect, session
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from deep_translator import GoogleTranslator
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# ===== INITIAL SETUP =====
app = Flask(__name__)
app.secret_key = "healmate_secret_key"

database = 'chatbot1.db'

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

model = load_model("chatbot_model2.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

last_intent_tag = None
last_response_index = -1

# ===== DATABASE TABLES =====
def createtable():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS register(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            gender TEXT,
            age INTEGER,
            height REAL,
            weight REAL,
            address TEXT,
            city TEXT,
            state TEXT,
            pincode INTEGER,
            phone TEXT,
            email TEXT UNIQUE,
            work_type TEXT,
            blood_group TEXT,
            medications TEXT,
            medication_name TEXT,
            medication_frequency TEXT,
            dosage TEXT,
            remarks TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            feedback TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

createtable()

# ===== ROUTES =====
@app.route("/")
def signup():
    return render_template("signup.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/patientregister")
def patientregister():
    return render_template("patientregister.html")

@app.route("/index")
def index():
    print(patient_details_list)
    return render_template("chatbot.html")

@app.route("/home")
def home():
    if "email" not in session:
        return redirect("/login")

    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM patient_details WHERE email=?",
        (session["email"],)
    )
    patient_data = cursor.fetchone()

    conn.close()

    return render_template("home.html", patient=patient_data)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        cursor.execute("SELECT email FROM register WHERE email=?", (email,))
        if cursor.fetchone():
            conn.close()
            return render_template('signup.html',
                                   error="Email already registered.")

        cursor.execute(
            "INSERT INTO register(name,email,password) VALUES (?,?,?)",
            (name, email, hashed_password)
        )

        conn.commit()
        conn.close()

        return redirect('/login')

    return render_template('signup.html')

@app.route("/logindetails", methods=['GET', 'POST'])
def logindetails():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        cursor.execute("SELECT password FROM register WHERE email=?", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[0], password):

            session["email"] = email

            # Fetch patient details (optional)
            cursor.execute(
                "SELECT * FROM patient_details WHERE email=?",
                (email,)
            )
            patient_data = cursor.fetchone()

            conn.close()

            if patient_data:
                session["patient_exists"] = True
            else:
                session["patient_exists"] = False

            return redirect("/home")

        else:
            conn.close()
            return render_template('login.html',
                                   error="Invalid Email or Password")

    return render_template('login.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():

    if "email" not in session:
        return redirect("/login")

    if request.method == 'POST':
        feedback_text = request.form['feedback']
        email = session["email"]

        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # Fetch patient first name
        cursor.execute(
            "SELECT first_name FROM patient_details WHERE email=?",
            (email,)
        )
        patient = cursor.fetchone()

        name = patient[0] if patient else "User"

        # Insert feedback
        cursor.execute(
            "INSERT INTO feedback(name, email, feedback) VALUES (?,?,?)",
            (name, email, feedback_text)
        )

        conn.commit()
        conn.close()

        return redirect("/home")

    return render_template('feedback.html')

@app.route('/submit', methods=['POST'])
def submit():
    global patient_details_list

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

    print("Submitted Data:", submitted_data)

    patient_details_list.extend(submitted_data)
    print("Updated Patient Details List:", patient_details_list)

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO patient_details (
            first_name, last_name, gender, age, height, weight,
            address, city, state, pincode, phone, email,
            work_type, blood_group, medications,
            medication_name, medication_frequency, dosage, remarks
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', submitted_data)

    conn.commit()
    conn.close()

    return render_template('home.html')

# ===== NLP FUNCTIONS =====
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence)
    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        return []

    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints):
    global last_intent_tag, last_response_index

    if not ints:
        return "I'm not confident about that. Can you rephrase?"

    tag = ints[0]["intent"]

    for intent in intents["intents"]:
        if intent["tag"] == tag:

            responses = intent["responses"]

            if tag == last_intent_tag:
                new_index = last_response_index
                while new_index == last_response_index and len(responses) > 1:
                    new_index = random.randrange(len(responses))
            else:
                new_index = random.randrange(len(responses))

            last_intent_tag = tag
            last_response_index = new_index

            return responses[new_index]

    return "I'm not sure about that."

def chatbot_response(msg):

    global last_intent_tag, last_response_index

    messg = msg.lower()

    if messg in ["no", "no no", "nope", "nah", "not this", "another", "i dont want this", 
                 "different", "any other", "other", "anything else", "something different",
                 "something else", "else", "something", "any other", "other" ]:

        if last_intent_tag:
            for i in intents["intents"]:
                if i["tag"] == last_intent_tag:
                    responses = i["responses"]
                    last_response_index = (last_response_index + 1) % len(responses)
                    return responses[last_response_index]

        return "Please ask something first."

    msg_checked = check_symptoms_in_database(messg)
    ints = predict_class(msg_checked)

    if ints:
        last_intent_tag = ints[0]["intent"]
        last_response_index = -1

    return getResponse(ints)

# ===== CHAT ROUTE =====
name_messages = ['my name', 'you know my name']
height_messages = ['my height', 'updated height', 'height', 'hgt']
weight_messages = ['my weight', 'updated weight', 'weight', 'wgt']
medications = ['medications', 'medicine name', 'my medicines']
medications_fry = ['medication frequency', 'medicine frequency',
                   'my medicines frequency']
medications_dos = ['medication dosage', 'medicine dosages',
                   'my medicine dosages', 'medicine dose']

@app.route("/get", methods=["POST"])
def get_bot_response():
    global last_intent_tag

    if "email" not in session:
        return "Please login first."

    msg = request.form["msg"]
    if not msg.strip():
        return "Please type something."
    
    lang = request.form.get("lang", "en")
    print("Selected language:", lang)
    print("Original message:", msg)
    
    try:
        if lang != "en":
            msg_en = GoogleTranslator(source='auto', target='en').translate(msg)
        else:
            msg_en = msg
    except Exception as e:
        print("Translation error:", e)
        msg_en = msg

    if msg_en in name_messages:
        response_en = patient_details_list[1]
    elif msg_en in height_messages:
        response_en = patient_details_list[5]
    elif msg_en in weight_messages:
        response_en = patient_details_list[6]
    elif msg_en in medications:
        response_en = patient_details_list[15]
    elif msg_en in medications_fry:
        response_en = patient_details_list[16]
    elif msg_en in medications_dos:
        response_en = patient_details_list[17]
    else:
        response_en = chatbot_response(msg_en)
 
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO chat_history (email,user_message,bot_response)
        VALUES (?,?,?)
    """, (session["email"], msg_en, response_en))
    conn.commit()
    conn.close()

    try:
        if lang != "en":
            response = GoogleTranslator(source='en', target=lang).translate(response_en)
        else:
            response = response_en
    except Exception as e:
        print("Response translation error:", e)
        response = response_en 

    return response

stopwords = {
    'i', 'have', 'the', 'a', 'and', 'is', 'of', 'to', 'with', 'for',
    'in', 'on', 'at', 'by', 'it', 'from'
}

def check_symptoms_in_database(symptom):
    symptom_keywords = symptom.lower().split()
    print('symptom_keywords:', symptom_keywords)

    filtered_keywords = [
        word for word in symptom_keywords if word not in stopwords
    ]
    print('filtered_keywords:', filtered_keywords)

    if 'symptoms' in filtered_keywords:
        target_diseases = ['diabetes', 'hypertension', 'heart attack']
        filtered_diseases = [
            disease for disease in patient_details_list
            if isinstance(disease, str)
            and disease.lower() in target_diseases
        ]
        print("Filtered Diseases:", filtered_diseases)
        if filtered_diseases:
            value = f" {', '.join(filtered_diseases)} disease and {symptom}"
            print('check', value)
            return value

    return symptom


@app.route("/history")
def view_history():
    if "email" not in session:
        return redirect("/login")

    if patient_details_list:
        user_email = patient_details_list[12]
    else:
        user_email = "guest"
        
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_message, bot_response, timestamp
        FROM chat_history
        WHERE email=?
        ORDER BY id DESC
    """, (session["email"],))

    chats = cursor.fetchall()
    conn.close()

    return render_template("history.html", chats=chats)


@app.route("/delete_history", methods=["POST"])
def delete_history():
    if "email" not in session:
        return "error"

    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chat_history WHERE email=?",
                   (session["email"],))

    conn.commit()
    conn.close()

    return "success"

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

if __name__ == "__main__":
    app.run(port=4000)
