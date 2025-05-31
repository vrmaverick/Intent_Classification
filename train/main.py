from data import prepare_data
from vectorize import initiate_vectroizer,initiate_embedder,encode_labels
from evaluate import eval
import tensorflow as tf
import tensorflow.keras.layers as layers

X_train,X_valid,X_test,y_train,y_valid,y_test = prepare_data()
text_Vectorizer = initiate_vectroizer()
embedding = initiate_embedder()
y_train_encoded,y_val_encoded = encode_labels(y_train,y_valid)
text_Vectorizer.adapt(X_train)

# Code to train bi directional RNN

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_Vectorizer(inputs)
x = embedding(x)
x = layers.Bidirectional(layer= layers.GRU(128, return_sequences= True))(x)
x = layers.Bidirectional(layer= layers.LSTM(64))(x)
x = layers.Dense(64, activation = 'relu')(x)
outputs = layers.Dense(27, activation = "softmax")(x)

model = tf.keras.Model(inputs, outputs, name="model_BIRNN")
model.compile(optimizer= 'Adam',loss= tf.keras.losses.sparse_categorical_crossentropy, metrics= ['Accuracy'])
print(model.summary())
history = model.fit(X_train,y_train_encoded,epochs = 10, validation_data=(X_valid,y_val_encoded))

model.save("BI_RNN.keras")

model_results_on_validation = eval(model,X_valid,y_val_encoded)
print(model_results_on_validation) 