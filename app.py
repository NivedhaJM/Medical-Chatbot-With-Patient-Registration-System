# ===== IMPORTS =====
import re
import random
import numpy as np
import pickle
import json
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, redirect, session
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from deep_translator import GoogleTranslator
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask import flash, redirect, url_for

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ===== INITIAL SETUP =====
app = Flask(__name__)
app.secret_key = "healmate_secret_key"
patient_details_list = []

database = 'chatbot1.db'

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

model = load_model("chatbot_model2.h5")
intents = json.loads(open("intents.json", encoding="utf-8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# ===== DRUG DATASET LOAD =====
with open("drugs_dataset.json", "r", encoding="utf-8") as f:
    drug_data = json.load(f)
drugs = drug_data["drugs"]
symptom_to_drug = drug_data["symptom_to_drug"]
context_keywords = drug_data["context_keywords"]

last_intent_tag = None
last_response_index = -1

# Tracks last symptom-based drug lookup for "any other medicine?" feature
last_symptom_lookup = None   # e.g. "allergy"
last_drug_index = 0          # index of last shown drug in symptom_to_drug list

# ===== ALLERGY FLOW =====
ALLERGY_TRIGGERS = [
    "allergy", "allergic", "i have allergy", "i have an allergy",
    "i am allergic", "allergies", "i have allergies"
]

ALLERGY_TYPE_MAP = {
    "allergy_skin": [
        "skin", "rash", "hives", "urticaria", "eczema", "itchy skin",
        "skin itching", "skin allergy", "contact dermatitis", "skin irritation",
        "bumps on skin", "red patches", "skin reaction", "skin burning"
    ],
    "allergy_eye": [
        "eye", "eyes", "itchy eyes", "watery eyes", "red eyes",
        "eye allergy", "eye irritation", "conjunctivitis",
        "eyes are itching", "eyes are red", "burning eyes", "swollen eyes"
    ],
    "allergy_nasal": [
        "nose", "nasal", "sneezing", "runny nose", "blocked nose",
        "stuffy nose", "dust", "pollen", "nasal allergy",
        "allergic rhinitis", "nose is blocked", "continuous sneezing"
    ],
    "allergy_food": [
        "food", "eating", "after eating", "seafood", "nut", "milk",
        "lactose", "gluten", "wheat", "egg", "food allergy", "food reaction"
    ],
    "allergy_insect": [
        "insect", "bee", "wasp", "mosquito", "bite", "sting",
        "ant bite", "bug bite", "i was bitten", "i was stung"
    ],
    "allergy_drug": [
        "medicine", "drug", "tablet", "antibiotic", "penicillin",
        "medicine allergy", "drug allergy", "allergic to medicine",
        "rash after medicine", "medicine reaction"
    ]
}

NUTRITION_TRIGGERS = [
    "diet", "nutrition", "what should i eat", "healthy food", "healthy eating",
    "diet advice", "nutrition tips", "balanced diet", "diet plan",
    "what to eat", "food advice", "healthy diet", "foods to avoid"
]

NUTRITION_TYPE_MAP = {
    "nutrition_carbs":   ["carbs", "carbohydrates", "carb", "carbohydrate"],
    "nutrition_protein": ["protein", "proteins", "protein intake"],
    "nutrition_fats":    ["fat", "fats", "healthy fat", "dietary fat", "saturated", "unsaturated"],
    "nutrition_fibre":   ["fibre", "fiber", "dietary fibre", "roughage"],
    "nutrition_vitamins":["vitamin", "vitamins", "minerals", "micronutrients", "supplements"],
    "nutrition_water":   ["water", "hydration", "hydrate", "how much water", "daily water"]
}

def is_nutrition_trigger(msg):
    msg_lower = msg.lower().strip()
    return any(trigger in msg_lower for trigger in NUTRITION_TRIGGERS)

def detect_nutrition_type(msg):
    msg_lower = msg.lower().strip()
    for tag, keywords in NUTRITION_TYPE_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            return tag
    return None

def get_nutrition_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            r = intent["responses"][0]
            return r if isinstance(r, str) else str(r)
    return "I couldn't find info on that. Try asking about carbs, protein, fats, fibre, vitamins, or water!"

def get_diet_overview():
    for intent in intents["intents"]:
        if intent["tag"] == "diet_nutrition":
            r = intent["responses"][0]
            return r if isinstance(r, str) else str(r)
    return "A balanced diet is key to good health!"

def is_allergy_trigger(msg):
    msg_lower = msg.lower().strip()
    return any(trigger in msg_lower for trigger in ALLERGY_TRIGGERS)

def detect_allergy_type(msg):
    msg_lower = msg.lower().strip()
    for tag, keywords in ALLERGY_TYPE_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            return tag
    return None

def get_allergy_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent["responses"][0]
    return "I'm not sure about that allergy type. Could you describe your symptoms more?"

# ===== HEADACHE FLOW =====
HEADACHE_TRIGGERS = [
    "headache", "head ache", "my head hurts", "head pain", "head is paining",
    "head throbbing", "pain in head", "splitting headache", "severe headache",
    "i have headache", "i have a headache", "migraine", "i have migraine",
    "i have a migraine", "cluster headache", "tension headache"
]

HEADACHE_TYPE_MAP = {
    "headache_tension": [
        "tension", "both sides", "forehead", "tight", "pressure", "constant",
        "dull", "stress", "heavy head", "squeezing", "band around"
    ],
    "headache_migraine": [
        "migraine", "one side", "throbbing", "pulsing", "light", "sound",
        "nausea", "vomiting", "aura", "half", "visual", "sensitive"
    ],
    "headache_cluster": [
        "cluster", "eye", "behind eye", "around eye", "eye socket",
        "one eye", "watery eye", "severe", "stabbing"
    ]
}

def is_headache_trigger(msg):
    msg_lower = msg.lower().strip()
    return any(trigger in msg_lower for trigger in HEADACHE_TRIGGERS)

def detect_headache_type(msg):
    msg_lower = msg.lower().strip()
    for tag, keywords in HEADACHE_TYPE_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            return tag
    return None

def get_headache_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent["responses"][0]
    return "I'm not sure about the headache type. Could you describe it more — throbbing, dull, or sharp? And where — forehead, temples, or one side?"

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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            log_type TEXT,
            value1 REAL,
            value2 REAL,
            note TEXT,
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

    email = session.get('email')   # assuming login system

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_details WHERE email = ?", (email,))
    data = cursor.fetchone()
    conn.close()

    return render_template('patientregister.html', data=data)

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

        try:
            conn = sqlite3.connect(database, timeout=10)
            cursor = conn.cursor()

            cursor.execute("SELECT email FROM register WHERE email=?", (email,))
            if cursor.fetchone():
                return render_template('signup.html',
                                       error="Email already registered.")

            cursor.execute(
                "INSERT INTO register(name,email,password) VALUES (?,?,?)",
                (name, email, hashed_password)
            )

            conn.commit()

        finally:
            conn.close()

        return redirect(url_for('logindetails'))

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

@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():

    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            return render_template("forgot.html",
                                   message="Passwords do not match",
                                   category="error")

        hashed_pw = generate_password_hash(new_password)

        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM register WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            cursor.execute("UPDATE register SET password = ? WHERE email = ?", (hashed_pw, email))
            conn.commit()
            conn.close()
            flash("Password updated successfully. Please login.")
            return redirect(url_for('login'))
        else:
            conn.close()
            return render_template("forgot.html",
                                   message="Email not found",
                                   category="error")

    return render_template('forgot.html')
    



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

    submitted_data = [
        request.form.get('first_name', ''),
        request.form.get('last_name', ''),
        request.form.get('gender', ''),
        int(request.form.get('age', 0)),
        float(request.form.get('height', 0)),
        float(request.form.get('weight', 0)),
        request.form.get('address', ''),
        request.form.get('city', ''),
        request.form.get('state', ''),
        request.form.get('pincode', ''),
        request.form.get('phone', ''),
        request.form.get('email', ''),
        request.form.get('work_type', ''),
        request.form.get('blood_group', ''),
        request.form.get('medications', ''),
        request.form.get('medication_name', ''),
        request.form.get('medication_frequency', ''),
        request.form.get('dosage', ''),
        request.form.get('remarks', '')
    ]

    print("Submitted Data:", submitted_data)

    print("Updated Patient Details List:", patient_details_list)

    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patient_details WHERE email = ?", (submitted_data[11],))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE patient_details SET
            first_name=?, last_name=?, gender=?, age=?, height=?, weight=?,
            address=?, city=?, state=?, pincode=?, phone=?,
            work_type=?, blood_group=?, medications=?,
            medication_name=?, medication_frequency=?, dosage=?, remarks=?
            WHERE email=?
        """, submitted_data[:11] + submitted_data[12:] + [submitted_data[11]])
    else:
        cursor.execute("""
            INSERT INTO patient_details (
                first_name, last_name, gender, age, height, weight,
                address, city, state, pincode, phone, email,
                work_type, blood_group, medications,
                medication_name, medication_frequency, dosage, remarks
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, submitted_data)

    conn.commit()
    conn.close()

    flash("Data submitted successfully")
    return redirect(url_for('patientregister'))

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
        return "I didn't quite understand that 🤔 Could you rephrase? You can ask me about symptoms, medicines, or your health data!"

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

            if tag == "fever":
                from flask import session as flask_session
                flask_session["awaiting_temp"] = True

            return responses[new_index]

    return "I'm not sure about that."

# 🔽 ADD YOUR FUNCTION HERE

# ===== DRUG LOOKUP FUNCTIONS =====
def is_drug_query(user_input: str) -> bool:
    """
    Only triggers when user explicitly mentions:
    - A drug/medicine keyword (medicine, drug, tablet, etc.)
    - A specific drug name (paracetamol, ibuprofen, etc.)
    General symptom messages like 'i have fever' go to the ML model as usual.
    """
    keywords = [
        "medicine", "drug", "tablet", "capsule", "dosage",
        "dose", "medication", "pill", "syrup", "injection",
        "remedies", "remedy", "treatment", "cure", "what to take",
        "what should i take", "what can i take", "suggest"
    ]
    drug_names = [d["drug_name"].lower() for d in drugs] + \
                 [d["generic_name"].lower() for d in drugs]
    query = user_input.lower()
    return (
        any(kw in query for kw in keywords) or
        any(name in query for name in drug_names)
    )

def format_drug(match, field=None, detailed=False) -> str:
    """Formats a drug entry. If field is specified, returns only that field."""
    if field == "frequency":
        return f"🕐 {match['drug_name']} should be taken: {match['frequency']}"
    elif field == "dosage":
        return f"📏 {match['drug_name']} dosage: {match['dosage']}"
    elif field == "usage":
        return f"🩺 {match['drug_name']} is used for: {match['usage']}"
    elif field == "warnings":
        return f"⚠️ {match['drug_name']} warnings: {match['warnings']}"
    elif detailed:
        return (
            f"💊 {match['drug_name']} ({match['generic_name']})\n"
            f"📂 Category  : {match['category']}\n"
            f"📏 Dosage    : {match['dosage']}\n"
            f"🩺 Usage     : {match['usage']}\n"
            f"🕐 Frequency : {match['frequency']}\n"
            f"⚠️  Warnings  : {match['warnings']}"
        )
    else:
        return (
            f"💊 {match['drug_name']} ({match['generic_name']})\n"
            f"📂 Category  : {match['category']}\n"
            f"📏 Dosage    : {match['dosage']}\n"
        )

def detect_context(query: str):
    """Detects what specific info the user wants (dosage, frequency, usage, warnings)."""
    # Check JSON-defined context keywords first
    for phrase, field in context_keywords.items():
        if phrase in query:
            return field

    # Extra natural language patterns
    dosage_patterns = [
        "its dosage", "what is its dosage", "how much should i take",
        "how much to take", "how many mg", "what dose", "what dosage"
    ]
    frequency_patterns = [
        "how many times", "how often", "how frequently",
        "when should i take", "when to take", "how many times a day",
        "frequency", "how frequent"
    ]
    usage_patterns = [
        "what is it for", "what is it used for", "what does it treat",
        "what is this for", "what is its use", "used for", "use of"
    ]
    warning_patterns = [
        "side effects", "what are its side effects", "any side effects",
        "side effect", "warnings", "is it safe", "precautions",
        "adverse effects", "what are the warnings", "any warnings"
    ]

    for p in dosage_patterns:
        if p in query:
            return "dosage"
    for p in frequency_patterns:
        if p in query:
            return "frequency"
    for p in usage_patterns:
        if p in query:
            return "usage"
    for p in warning_patterns:
        if p in query:
            return "warnings"

    return None

def lookup_drug(user_input: str) -> str:
    """
    Smart lookup:
    1. If drug name found → return its info (with context if detected)
    2. If symptom found → suggest matching drugs
    3. Otherwise → not found message
    """
    query = user_input.lower().strip()
    field = detect_context(query)

    # 1. Try to match by drug name or generic name → full details
    match = next(
        (d for d in drugs
         if d["drug_name"].lower() in query or d["generic_name"].lower() in query),
        None
    )
    if match:
        return format_drug(match, field, detailed=True)

    # 2. Try symptom-based lookup → short format, track for follow-ups
    matched_symptom = next(
        (symptom for symptom in symptom_to_drug if symptom in query),
        None
    )
    if matched_symptom:
        global last_symptom_lookup, last_drug_index
        last_symptom_lookup = matched_symptom
        last_drug_index = 0
        drug_name = symptom_to_drug[matched_symptom][0]
        drug = next((d for d in drugs if d["drug_name"] == drug_name), None)
        drug_list = symptom_to_drug[matched_symptom]
        has_more = len(drug_list) > 1
        response = f"For {matched_symptom}, here is a commonly used medicine:\n\n" + format_drug(drug, field, detailed=False)
        if has_more:
            response += "\n\nAsk 'any other medicine?' to see alternatives."
        return response

    # 3. Not found
    return (
        "Sorry, I couldn't find information about that. "
        "Try mentioning a drug name (e.g. 'paracetamol') or a symptom (e.g. 'fever')."
    )



# ===== CONTEXT FROM CHAT HISTORY =====
def get_recent_context(email, limit=3):
    """Fetches last `limit` messages from chat_history for the user."""
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_message, bot_response FROM chat_history
        WHERE email = ?
        ORDER BY id DESC
        LIMIT ?
    """, (email, limit))
    rows = cursor.fetchall()
    conn.close()
    return list(reversed(rows))  # oldest first


def get_context_aware_response(msg, email):
    """
    Uses recent chat history to handle follow-up questions.
    E.g. 'what about side effects?' after asking about paracetamol.
    Skips context if current message already contains a specific drug name.
    """
    recent = get_recent_context(email, limit=3)
    msg_lower = msg.lower().strip()

    # If user explicitly mentions a drug name in current message,
    # skip context and let lookup_drug handle it directly
    for drug in drugs:
        if drug["drug_name"].lower() in msg_lower or drug["generic_name"].lower() in msg_lower:
            return None

    followup_keywords = [
        # general follow-ups
        "what about", "tell me more", "more about", "and also", "same medicine", "that medicine", "this medicine",
        # dosage related
        "what is its dosage", "its dosage", "dosage", "dose", "how much", "quantity",
        "how much should i take", "how much to take", "how many mg",
        # frequency related
        "how many times", "how often", "when to take", "frequency",
        "how many times a day", "how frequently", "when should i take",
        # usage related
        "what is it for", "what is it used for", "used for", "what does it treat",
        "what is this for", "what is its use",
        # side effects / warnings
        "side effects", "what are its side effects", "side effect",
        "any side effects", "warnings", "is it safe", "precautions",
        "what are the warnings", "any warnings", "adverse effects",
        # general pronoun-based follow-ups
        "can i take it", "can i take this", "can i use it",
        "how to take it", "how do i take it", "how to use it"
    ]

    is_followup = any(kw in msg_lower for kw in followup_keywords)

    if is_followup and recent:
        # Search most recent messages first (newest to oldest)
        # Check bot responses first as they contain the drug that was last recommended
        mentioned_drug = None

        for user_msg, bot_msg in reversed(recent):
            # Check bot response first — most reliable source of last recommended drug
            for drug in drugs:
                if drug["drug_name"].lower() in bot_msg.lower() or drug["generic_name"].lower() in bot_msg.lower():
                    mentioned_drug = drug
                    break
            if mentioned_drug:
                break
            # Then check user message
            for drug in drugs:
                if drug["drug_name"].lower() in user_msg.lower() or drug["generic_name"].lower() in user_msg.lower():
                    mentioned_drug = drug
                    break
            if mentioned_drug:
                break

        if mentioned_drug:
            field = detect_context(msg_lower)
            return format_drug(mentioned_drug, field)

    return None  # No context match, fall through to normal response



def handle_alternative_drug():
    """Returns the next drug in the list when user asks for an alternative."""
    global last_symptom_lookup, last_drug_index

    if not last_symptom_lookup:
        return None

    drug_list = symptom_to_drug.get(last_symptom_lookup, [])
    last_drug_index += 1

    if last_drug_index >= len(drug_list):
        last_drug_index = len(drug_list) - 1
        return f"Sorry, there are no more alternatives for {last_symptom_lookup}. Those were all the common medicines!"

    drug_name = drug_list[last_drug_index]
    drug = next((d for d in drugs if d["drug_name"] == drug_name), None)
    if drug:
        has_more = last_drug_index < len(drug_list) - 1
        response = f"Another medicine for {last_symptom_lookup}:\n\n" + format_drug(drug)
        if has_more:
            response += "\n\nAsk 'any other medicine?' for more alternatives."
        return response

    return None


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

    # Normal prediction
    msg_checked = check_symptoms_in_database(messg)
    ints = predict_class(msg_checked)

    if ints:
        last_intent_tag = ints[0]["intent"]
        last_response_index = -1

    return getResponse(ints)



# ===== CHAT ROUTE =====
# chatbot message shortcuts
name_messages = [
    'my name', 'you know my name', 'what is my name',
    'tell my name', 'say my name', 'who am i', "what's my name", "whats my name"
]

height_messages = [
    'my height', 'updated height', 'height', 'hgt',
    'what is my height', 'tell my height',
    'how tall am i', 'current height',  "what's my height", "whats my height"
]

weight_messages = [
    'my weight', 'updated weight', 'weight', 'wgt',
    'what is my weight', 'tell my weight',
    'how much do i weigh', 'current weight',  "what's my weight", "whats my weight"
]

medications = [
    'medications', 'medicine name', 'my medicines',
    'what medicines am i taking', 'my medication',
    'current medicines', 'medicine details', "what are all the medications"
]

medications_fry = [
    'medication frequency', 'medicine frequency',
    'my medicines frequency', 'how often do i take medicine',
    'medicine schedule', 'how many times a day'
]

medications_dos = [
    'medication dosage', 'medicine dosages',
    'my medicine dosages', 'medicine dose',
    'how much medicine do i take',
    'dosage details', 'dose amount'
]

drug_info_triggers = [
    'drug info', 'medicine info', 'drug information',
    'medicine information', 'tell me about medicine',
    'what is this drug', 'drug details', 'medicine details'
]


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

    email = session.get('email')

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_details WHERE email = ?", (email,))
    patient = cursor.fetchone()
    conn.close()

    if not patient:
        response_en = "No patient data found. Please register first."
    else:
        response_en = "I'm not sure about that. Could you rephrase?"  # safe default
        # ===== 0. ALLERGY MULTI-TURN FLOW =====
        if session.get("awaiting_allergy_type"):
            # User is responding to "what type of allergy?"
            allergy_tag = detect_allergy_type(msg_en)
            if allergy_tag:
                session.pop("awaiting_allergy_type", None)
                response_en = get_allergy_response(allergy_tag)
            else:
                response_en = (
                    "I didn't quite catch that! Could you tell me the type — "
                    "for example: skin, eyes, nose, food, insect bite, or medicine?"
                )
        elif is_allergy_trigger(msg_en) and not detect_allergy_type(msg_en):
            # User said "I have allergy" without specifying type — ask them
            session["awaiting_allergy_type"] = True
            response_en = random.choice([
                "I'd like to help with your allergy! Could you tell me what type — skin, eyes, nose, food, insect bite, or medicine?",
                "Allergies can vary a lot! Is it a skin rash, eye irritation, nasal/sneezing, food reaction, insect bite, or medicine reaction?",
                "Sure, I can help! Just let me know where or what kind — skin, eyes, nose, food, insect, or a drug allergy?"
            ])
        elif is_allergy_trigger(msg_en) and detect_allergy_type(msg_en):
            # User mentioned allergy type directly (e.g. "I have skin allergy")
            session.pop("awaiting_allergy_type", None)
            allergy_tag = detect_allergy_type(msg_en)
            response_en = get_allergy_response(allergy_tag)

        # ===== MIGRAINE PREVENTION FOLLOW-UP =====
        elif session.get("awaiting_migraine_prevention"):
            msg_lower = msg_en.lower().strip()
            if any(w in msg_lower for w in ["yes", "yeah", "sure", "please", "tell me", "go ahead", "ok", "okay", "yep"]):
                session.pop("awaiting_migraine_prevention", None)
                response_en = next(
                    (i["responses"][0] for i in intents["intents"] if i["tag"] == "migraine_prevention"),
                    "Stay consistent with sleep, hydration, and stress management to reduce migraines long-term!"
                )
            else:
                session.pop("awaiting_migraine_prevention", None)
                response_en = "No problem! Feel free to ask me anything else. 😊"

        # ===== HEADACHE MULTI-TURN FLOW =====
        elif session.get("awaiting_headache_type"):
            headache_tag = detect_headache_type(msg_en)
            if headache_tag:
                session.pop("awaiting_headache_type", None)
                if headache_tag == "headache_migraine":
                    session["awaiting_migraine_prevention"] = True
                response_en = get_headache_response(headache_tag)
            elif any(w in msg_en.lower() for w in ["not sure", "don't know", "i don't know", "unsure", "no idea"]):
                session.pop("awaiting_headache_type", None)
                response_en = (
                    "No worries! Here are general tips for any headache:\n\n"
                    "💧 Drink water first — dehydration is the most common cause.\n"
                    "💊 Paracetamol (500–1000mg) or Ibuprofen (400mg) for pain relief.\n"
                    "🌑 Rest in a quiet, dark room and avoid screens.\n"
                    "🌡️ Try a cold or warm compress on your forehead.\n"
                    "🩺 If headaches are frequent or very severe, please see a doctor."
                )
            else:
                response_en = "Could you describe it more? Is it throbbing on one side (migraine), a tight band around the head (tension), or severe pain near the eye (cluster)?"

        elif is_headache_trigger(msg_en) and not detect_headache_type(msg_en):
            session["awaiting_headache_type"] = True
            response_en = random.choice([
                "Sorry to hear that! 😔 Could you tell me more — is it a throbbing pain on one side, a tight band around the head, or a sharp pain near the eye?",
                "Headaches can have different causes! Is it dull and constant, throbbing, or sharp? And where — forehead, temples, one side, or near the eye?",
                "I can help! Is it a tension headache (tight/pressure), migraine (throbbing, one side), or cluster (severe, near the eye)? Or just describe how it feels!"
            ])

        elif is_headache_trigger(msg_en) and detect_headache_type(msg_en):
            session.pop("awaiting_headache_type", None)
            headache_tag = detect_headache_type(msg_en)
            if headache_tag == "headache_migraine":
                session["awaiting_migraine_prevention"] = True
            response_en = get_headache_response(headache_tag)

        # ===== NUTRITION MULTI-TURN FLOW =====
        elif session.get("awaiting_nutrition_type"):
            nutr_tag = detect_nutrition_type(msg_en)
            if nutr_tag:
                response_en = get_nutrition_response(nutr_tag)
                # keep session alive so user can keep asking follow-ups
            elif any(w in msg_en.lower() for w in ["yes", "yeah", "sure", "more", "tell me more", "go on", "okay", "ok", "yep"]):
                response_en = "What would you like to know more about? You can ask about carbs, protein, fats, fibre, vitamins, or water! 😊"
            elif any(w in msg_en.lower() for w in ["no", "nope", "that's all", "done", "enough", "nothing", "no thanks"]):
                session.pop("awaiting_nutrition_type", None)
                response_en = "Got it! Feel free to ask anytime if you have more health questions. 😊"
            else:
                session.pop("awaiting_nutrition_type", None)
                response_en = chatbot_response(msg_en)

        elif is_nutrition_trigger(msg_en):
            nutr_tag = detect_nutrition_type(msg_en)
            if nutr_tag:
                # User asked directly e.g. "how much protein should I eat"
                session["awaiting_nutrition_type"] = True
                response_en = get_nutrition_response(nutr_tag)
            else:
                # General diet question — give overview and prompt follow-up
                session["awaiting_nutrition_type"] = True
                response_en = get_diet_overview()

        # ===== DIRECT KEYWORD MATCHING (bypass ML for common inputs) =====
        elif re.match(r'^(hi+|he+y+|hello+|howdy|sup|yo|greetings|good (morning|evening|afternoon|day)|how are (you|u|ya)|what\'?s? ?up|is anyone there)$', msg_en.lower().strip()):
            response_en = random.choice([
                "Hey! 👋 I'm HealMate, your personal health assistant. What's on your mind today?",
                "Hi there! How are you feeling? I'm here to help with any health questions.",
                "Hello! Great to see you. Tell me your symptoms or ask about any medicine — I'm here!",
                "Hey! Feel free to ask me anything health-related. 😊"
            ])

        elif re.match(r'^(bye+|goodbye|good ?night|see (you|ya)|take care|cya|ttyl|later|i (am |got to )?go+)$', msg_en.lower().strip()):
            response_en = random.choice([
                "Goodbye! Take care and stay healthy! 🌿",
                "See you later! Don't hesitate to return if you have more questions.",
                "Take care! Remember, your health is your wealth. Goodbye! 😊"
            ])

        elif re.match(r'^(thanks?|thank (you|u)|thx|ty|much appreciated|that (was )?helpful|great help|appreciate it)$', msg_en.lower().strip()):
            response_en = random.choice([
                "You're welcome! Take care of yourself 😊",
                "Happy to help! Don't hesitate to come back if you have more questions.",
                "Glad I could be of help! Wishing you a speedy recovery. 🌿"
            ])

        elif re.match(r'^(ok+a?y?|alright|got it|sure|cool|noted|fine|i see|great|perfect|good)$', msg_en.lower().strip()):
            response_en = random.choice([
                "Got it! Let me know if there's anything else I can help with. 😊",
                "Alright! Feel free to ask if you have more questions.",
                "Sure thing! Stay healthy and take care."
            ])

        # ===== TEMPERATURE FLOW =====
        elif session.get("awaiting_temp"):
            temp_match = re.search(r'\d+\.?\d*', msg_en)
            if temp_match:
                temp = float(temp_match.group())
                session.pop("awaiting_temp", None)
                if temp < 99:
                    response_en = (
                        "Your temperature is normal! 🌡️ No medication needed.\n\n"
                        "💧 Rest well and drink plenty of fluids.\n"
                        "🍵 If you feel tired, try warm fluids like herbal tea or soup.\n"
                        "😴 Get adequate sleep to keep your immune system strong."
                    )
                elif temp <= 100.4:
                    response_en = (
                        "You have a mild fever. 🤒 Here's what to do:\n\n"
                        "💊 Paracetamol (Dolo 650mg) can help reduce fever. "
                        "Buy here: https://blinkit.com/prn/dolo-650mg-strip-of-15-tablets/prid/639498\n"
                        "💧 Drink plenty of fluids — water, ORS, or coconut water.\n"
                        "🧊 Apply a cool damp cloth on your forehead.\n"
                        "😴 Rest as much as possible to help your body recover.\n"
                        "🩺 Visit a doctor if fever persists beyond 2 days or worsens."
                    )
                else:
                    response_en = (
                        "You have a high fever! 🚨 Please take action immediately.\n\n"
                        "💊 Take Paracetamol (Dolo 650mg) right away to reduce fever. "
                        "Buy here: https://blinkit.com/prn/dolo-650mg-strip-of-15-tablets/prid/639498\n"
                        "💧 Stay well hydrated — drink ORS or fluids every hour.\n"
                        "🧊 Use a cool damp cloth on forehead, armpits, and neck.\n"
                        "🌡️ Monitor temperature every 2 hours.\n"
                        "🚫 Avoid heavy clothing or blankets — let heat escape.\n"
                        "🩺 Consult a doctor as soon as possible — high fever can indicate serious infection."
                    )
            else:
                response_en = "Please enter your temperature as a number in °F (e.g. 101.5) 🌡️"

        # ===== 1. ALTERNATIVE DRUG — "any other medicine?" =====
        else:
            alt_keywords = [
                "any other medicine", "another medicine", "other medicine",
                "any alternative", "alternative medicine", "other drug",
                "any other drug", "another drug", "other option",
                "any other option", "something else for"
            ]
            if any(kw in msg_en.lower() for kw in alt_keywords):
                alt = handle_alternative_drug()
                if alt:
                    response_en = alt
                else:
                    response_en = chatbot_response(msg_en)

            # ===== 2. CONTEXT-AWARE — follow-up questions =====
            elif get_context_aware_response(msg_en, email) is not None:
                context_response = get_context_aware_response(msg_en, email)
                response_en = context_response

            # ===== 3. DRUG LOOKUP — explicit drug/medicine queries =====
            elif is_drug_query(msg_en.lower()):
                response_en = lookup_drug(msg_en.lower())

            elif msg_en in name_messages:
                response_en = patient[1]   # first_name

            elif msg_en in height_messages:
                response_en = f"Your height is {patient[5]} cm."

            elif msg_en in weight_messages:
                response_en = f"Your weight is {patient[6]} kg."

            elif msg_en in medications:
                response_en = patient[15]

            elif msg_en in medications_fry:
                response_en = patient[17]

            elif msg_en in medications_dos:
                response_en = patient[18]

            # ===== 4. ML MODEL — everything else =====
            else:
                response_en = chatbot_response(msg_en)

    print("Fetched patient data:", patient)


 
    try:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO chat_history (email,user_message,bot_response,timestamp)
            VALUES (?,?,?,?)
        """, (session["email"], msg_en, response_en, now_ist()))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Chat history insert error:", e)

    try:
        if lang != "en":
            response = GoogleTranslator(source='en', target=lang).translate(response_en)
        else:
            response = response_en
    except Exception as e:
        print("Response translation error:", e)
        response = response_en

    return response if response else "Sorry, I couldn't process that. Please try again."

# ======== model helper functions ========

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


# ===== HEALTH LOG ROUTES =====
@app.route("/health_log/add", methods=["POST"])
def add_health_log():
    if "email" not in session:
        return "unauthorized", 401
    email = session["email"]
    log_type = request.form.get("log_type")
    value1 = request.form.get("value1")
    value2 = request.form.get("value2", None)
    note = request.form.get("note", "")
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO health_log (email, log_type, value1, value2, note, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (email, log_type, value1, value2, note, now_ist()))
    conn.commit()
    conn.close()
    return "success"


@app.route("/health_log/data")
def get_health_log():
    if "email" not in session:
        return "unauthorized", 401
    import json
    email = session["email"]
    log_type = request.args.get("log_type", "blood_sugar")
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT value1, value2, note, timestamp FROM health_log
        WHERE email = ? AND log_type = ?
        ORDER BY timestamp ASC LIMIT 30
    """, (email, log_type))
    rows = cursor.fetchall()
    conn.close()
    data = [{"value1": r[0], "value2": r[1], "note": r[2], "timestamp": r[3]} for r in rows]
    return json.dumps(data)


if __name__ == "__main__":
    app.run()
