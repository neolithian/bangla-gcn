import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("bangla_comments_drama.csv", encoding='utf-8')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
word_index = tokenizer.word_index
max_sequence_length = 100
data = pad_sequences(sequences, maxlen=max_sequence_length)

labels = df['sentiment']
labels = np.asarray(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

input_layer = Input(shape=(max_sequence_length,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1, 128, input_length=max_sequence_length)(input_layer)

# First GCN layer
conv1_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding_layer)
max_pool1_layer = MaxPooling1D(pool_size=2)(conv1_layer)
gcn1_layer = GlobalMaxPooling1D()(max_pool1_layer)

# Second GCN layer
conv2_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
max_pool2_layer = MaxPooling1D(pool_size=2)(conv2_layer)
gcn2_layer = GlobalMaxPooling1D()(max_pool2_layer)

# Third GCN layer
conv3_layer = Conv1D(filters=128, kernel_size=7, activation='relu')(embedding_layer)
max_pool3_layer = MaxPooling1D(pool_size=2)(conv3_layer)
gcn3_layer = GlobalMaxPooling1D()(max_pool3_layer)

# Concatenate the outputs from the three GCN layers
concat_layer = concatenate([gcn1_layer, gcn2_layer, gcn3_layer])

# Dropout layer
dropout_layer = Dropout(0.5)(concat_layer)

# Output layer
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

text = ["সে খুব ভাল মানুষ"]
text_sequences = tokenizer.texts_to_sequences(text)
text_data = pad_sequences(text_sequences, maxlen=max_sequence_length)
prediction = model.predict(text_data)
print(prediction)



