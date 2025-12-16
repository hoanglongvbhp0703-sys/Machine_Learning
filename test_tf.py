import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as pl
import  numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
import logging
centers = [[-5,2],[-2,-2],[1,2],[5,-2]]
X_train,y_train = make_blobs(n_samples=2000,centers=centers,cluster_std=1.0,random_state=30)
model = Sequential(
    [
        Dense(25,activation = 'relu'),
        Dense(15,activation = 'relu'),
        Dense(4,activation = 'softmax')
    ]
)
model_tmp=tf.keras.models.clone_model(model)
model_tmp.build((None,2))
W0,b0=model_tmp.layers[0].get_weights()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.fit(
    X_train,y_train,
    epochs=20
)
x_model=model.predict(X_train)
model.summary()

for layer in model.layers:
    W,b = layer.get_weights()
    print(layer.name)
    print("W shape", W.shape)
    print("b shape", b.shape)
    print()

W1,b1=model_tmp.layers[0].get_weights()
print(W0)
print(b0)
print(W1)
print(b1)
print(np.allclose(W0,W1))