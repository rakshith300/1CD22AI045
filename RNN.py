# -*- coding: utf-8 -*-
"""
Improved RNN Text Generation
- Better preprocessing
- Dropout regularization
- Stable training
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Input Text
# -----------------------------
text = "The beautiful girl whom I met last time is very intelligent also"

# Character-level processing
chars = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

VOCAB_SIZE = len(chars)
SEQ_LENGTH = 6

# -----------------------------
# Dataset Creation
# -----------------------------
X, y = [], []

for i in range(len(text) - SEQ_LENGTH):
    X.append([char_to_index[c] for c in text[i:i+SEQ_LENGTH]])
    y.append(char_to_index[text[i+SEQ_LENGTH]])

X = np.array(X)
y = np.array(y)

X = tf.one_hot(X, VOCAB_SIZE)
y = tf.one_hot(y, VOCAB_SIZE)

# -----------------------------
# RNN Model
# -----------------------------
model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(SEQ_LENGTH, VOCAB_SIZE)),
    Dropout(0.2),
    Dense(VOCAB_SIZE, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Training
# -----------------------------
model.fit(X, y, epochs=60, verbose=1)

# -----------------------------
# Text Generation
# -----------------------------
start_seq = "The beautiful girl "
generated_text = start_seq

for _ in range(50):
    input_seq = [char_to_index[c] for c in generated_text[-SEQ_LENGTH:]]
    input_seq = tf.one_hot([input_seq], VOCAB_SIZE)
    prediction = model.predict(input_seq, verbose=0)
    next_char = index_to_char[np.argmax(prediction)]
    generated_text += next_char

print("\nGenerated Text:")
print(generated_text)
