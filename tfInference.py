import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input, decode_predictions


def load_model():
   model = EfficientNetV2L(weights='imagenet')
   return model


def preprocess_predict(model, X, batchSize):
   X = np.asarray(X) # convert X to numpy array
   X = preprocess_input(X)
   Y = model.predict(X, batch_size=batchSize)

   return Y
