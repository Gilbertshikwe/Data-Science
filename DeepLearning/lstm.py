import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers #type:ignore
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D #type:ignore
from sklearn.model_selection import train_test_split
import zipfile
import requests
import io

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# Download the dataset
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the specific file
z.extractall()

# Read the specific file from the extracted contents
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

# Prepare the data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Tokenize the text
max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)

# Define target variable
Y = df['label'].values

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
print(f'Test Accuracy: {accuracy}')

# Save the model in the native Keras format
model.save("sms_spam_lstm.keras")

# Load the model
loaded_model = keras.models.load_model("sms_spam_lstm.keras")

# Print a summary of the model architecture
loaded_model.summary()

