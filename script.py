import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Parâmetros ajustados
vocab_size = 10000
max_length = 150
embedding_dim = 64  # Alterado de 128 para 256
lstm_units = 16     # Alterado de 64 para 128
dropout_rate = 0.5
learning_rate = 0.00005
batch_size = 256
num_epochs = 15

# Carregar dados do IMDb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train_padded = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test_padded = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# Implementação da camada de atenção
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        et = K.tanh(K.dot(x, self.W) + self.b)
        at = K.softmax(et, axis=1)
        ot = x * at
        return K.sum(ot, axis=1)

# Definição do modelo com hiperparâmetros ajustados
inputs = Input(shape=(max_length,))
embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
lstm = LSTM(lstm_units, return_sequences=True)(embedding)
attention = AttentionLayer()(lstm)
dense1 = Dense(64, activation='relu')(attention)
dropout = Dropout(dropout_rate)(dense1)
outputs = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=inputs, outputs=outputs)

# Compilação do modelo com taxa de aprendizado ajustada
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Treinamento do modelo
history = model.fit(x_train_padded, np.array(y_train), epochs=num_epochs, batch_size=batch_size, validation_data=(x_test_padded, np.array(y_test)))

# Avaliação do modelo
loss, accuracy = model.evaluate(x_test_padded, y_test)
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')