import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
max_features = 10000
maxlen = 200
(train_data, train_labels), (test_data, test_labels) =
imdb.load_data(num_words=max_features)
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)
def build_train_model(model_type, max_features, maxlen, train_data, train_labels):
model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=maxlen))
if model_type == 'LSTM':
model.add(layers.LSTM(64, return_sequences=False))
elif model_type == 'GRU':
model.add(layers.GRU(64, return_sequences=False))
elif model_type == 'SimpleRNN':
model.add(layers.SimpleRNN(64, return_sequences=False))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=10, batch_size=128,
validation_split=0.2)
return model, history
lstm_model, lstm_history = build_train_model('LSTM', max_features, maxlen,
train_data, train_labels)
gru_model, gru_history = build_train_model('GRU', max_features, maxlen,
train_data, train_labels)
rnn_model, rnn_history = build_train_model('SimpleRNN', max_features, maxlen,
train_data, train_labels)
lstm_test_loss, lstm_test_acc = lstm_model.evaluate(test_data, test_labels)
gru_test_loss, gru_test_acc = gru_model.evaluate(test_data, test_labels)
rnn_test_loss, rnn_test_acc = rnn_model.evaluate(test_data, test_labels)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['accuracy'], label='LSTM Train')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Val')
plt.plot(gru_history.history['accuracy'], label='GRU Train')
plt.plot(gru_history.history['val_accuracy'], label='GRU Val')
plt.plot(rnn_history.history['accuracy'], label='RNN Train')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='LSTM Train')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val')
plt.plot(gru_history.history['loss'], label='GRU Train')
plt.plot(gru_history.history['val_loss'], label='GRU Val')
plt.plot(rnn_history.history['loss'], label='RNN Train')
plt.plot(rnn_history.history['val_loss'], label='RNN Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
def plot_confusion_matrix(model, test_data, test_labels, model_name):
predictions = (model.predict(test_data) > 0.5).astype("int32")
cm = confusion_matrix(test_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'{model_name} Confusion Matrix')
plt.show()
plot_confusion_matrix(lstm_model, test_data, test_labels, 'LSTM')
plot_confusion_matrix(gru_model, test_data, test_labels, 'GRU')
plot_confusion_matrix(rnn_model, test_data, test_labels, 'SimpleRNN')