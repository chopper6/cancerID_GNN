# feedforward neural network

import tensorflow as tf
from tensorflow.keras import layers


def build_ffnn(hidden_units, dropout_rate, num_classes=5):
	fnn_layers = []

	for units in hidden_units:
		#fnn_layers += [layers.BatchNormalization()]
		#fnn_layers += [layers.Dropout(dropout_rate)]
		fnn_layers += [layers.Dense(units, activation=tf.nn.relu)]
		fnn_layers += [layers.Dropout(dropout_rate)]
	fnn_layers += [layers.Dense(num_classes)]

	return tf.keras.Sequential(fnn_layers)

