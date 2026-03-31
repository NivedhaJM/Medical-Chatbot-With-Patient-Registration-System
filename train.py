# libraries
import tensorflow
import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('punkt_tab')

words = []
classes = []
documents = []
ignore_words = ["?", "!"]

intents = json.loads(open("intents.json", encoding="utf-8").read())

# words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatize and clean
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique lemmatized words")

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print("Training data created —", len(train_x), "samples")

# ===== IMPROVED MODEL =====
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

sgd = SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(
    train_x, train_y,
    epochs=200,
    batch_size=4,
    verbose=1
)

model.save("chatbot_model2.h5")
print("Model saved successfully!")
print(f"Final training accuracy: {hist.history['accuracy'][-1]:.4f}")
