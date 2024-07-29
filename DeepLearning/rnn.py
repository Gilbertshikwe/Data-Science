import tensorflow as tf
from tensorflow.keras.datasets import imdb #type:ignore
from tensorflow.keras.preprocessing import sequence #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense #type:ignore 

# Load a smaller subset of the IMDB dataset
max_features = 10000  # Number of words to consider as features
max_len = 200  # Cut texts after this number of words (among top max_features most common words)
batch_size = 32
num_samples = 5000  # Number of samples to use for training and testing

# Load the IMDB dataset
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

# Use a smaller subset of the data
input_train = input_train[:num_samples]
y_train = y_train[:num_samples]
input_test = input_test[:num_samples]
y_test = y_test[:num_samples]

# Pad sequences (ensure equal length)
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)

# Build the RNN model
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(input_test, y_test)
print(f'Test accuracy: {test_acc}') 

