# feedforward neural network


def train_model(model,Xtrain,Ytrain,verbose=0):
	# Compile the model.
	model.compile(
		optimizer=keras.optimizers.Adam(PARAMS['learning_rate']),
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
	)
	# Create an early stopping callback.
	early_stopping = keras.callbacks.EarlyStopping(
		monitor="val_acc", patience=50, restore_best_weights=True
	)
	# Fit the model.
	history = model.fit(
		x=Xtrain,
		y=Ytrain,
		epochs=PARAMS['num_epochs'],
		batch_size=PARAMS['batch_size'],
		validation_split=PARAMS['validation_split'],
		callbacks=[early_stopping],
		verbose=verbose
	)

	return history


def ffn(hidden_units, dropout_rate, name=None):
	fnn_layers = []

	for units in hidden_units:
		fnn_layers.append(layers.BatchNormalization())
		fnn_layers.append(layers.Dropout(dropout_rate))
		fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

	return keras.Sequential(fnn_layers, name=name)

def build_baseline_model(num_nodes, num_classes=2):
	hidden_units, dropout_rate = PARAMS['hidden_units'], PARAMS['dropout_rate']
	inputs = layers.Input(shape=(num_nodes,), name="input_features")
	x = ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
	for block_idx in range(2): # could make # laers a param
		# Create an FFN block.
		x1 = ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
		# Add skip connection.
		x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
	# Compute logits.
	logits = layers.Dense(num_classes, name="logits")(x)
	# Create the model.
	return keras.Model(inputs=inputs, outputs=logits, name="baseline")




def run_model(model, Xtrain, Xtest, Ytrain, Ytest):
	history = train_model(model, Xtrain, Ytrain, verbose=0)
	display_learning_curves(history)
	_, test_accuracy = model.evaluate(x=Xtest, y=Ytest, verbose=0)
	print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")	
	return history, test_accuracy