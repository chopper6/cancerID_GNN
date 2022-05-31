# adapted from: https://keras.io/examples/graph/gnn_citations/

# check that can do proj here instead of colabs, def prefer

# data: 
# a graph file that gives the adjacency matrix
# ssRNA data that gives normalized RNA values (in 0,1) for each node in the network
#	with corresponding tags for cancer/non-cancer (or w.e classification)

# TODO:
# split into multiple files jp
# have the toy problem use a feature of the topo
# display learning curves will mix FF vs GNN later

# TODO: check if this axis arg in call() is correct 


import os, sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
tf.debugging.enable_check_numerics() # recommended while debugging

PARAMS = {
	'hidden_units' : [int(2**5), int(2**5)],
	'learning_rate': 0.001,
	'dropout_rate' : 0.1,
	'num_epochs' : 20,
	'batch_size' : int(2**7),
	'validation_split': 0.2
}

def run_practice():
	A, ssRNA, labels, num_nodes = get_toy_data()
	Xtrain, Xval, Xtest, Ytrain, Yval, Ytest= split_data(ssRNA, labels)

	print('\n')
	pnet = PracticeNet(A)
	print('\n')


	pnet.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
	in_shape= (None,len(Xtrain[0]))
	print('inshape=',in_shape) #Xtrain[:64].shape)
	pnet.build(in_shape)
	pnet.summary()

	history = pnet.fit(
		Xtrain,
		Ytrain,
		batch_size=64,
		epochs=20,
		validation_data=(Xval, Yval),
	)

	display_learning_curves(history)
	test_err = pnet.evaluate(x=Xtest, y=Ytest, verbose=0)
	print(f"Test error: {round(test_err * 100, 2)}%")


def run_toy_model():
	A, ssRNA, labels, num_nodes = get_toy_data()
	Xtrain, Xtest, Ytrain, Ytest= split_data(ssRNA, labels)

	if 0:
		print("\n\n~~~~~~~~ Running Baseline ~~~~~~~~\n\n")
		baseline_model = build_baseline_model(num_nodes)
		baseline_model.summary()
		hist_baseline, acc_baseline = run_model(gnn_model, Xtrain, Xtest, Ytrain, Ytest)

	print("\n\n~~~~~~~~ Running GNN ~~~~~~~~\n\n")
	gnn_model = build_GNN_model(A, num_nodes, Xtrain[0])
	gnn_model.summary()
	hist_gnn, acc_gnn = run_model(gnn_model, Xtrain, Xtest, Ytrain, Ytest)


def run_model(model, Xtrain, Xtest, Ytrain, Ytest):
	history = train_model(model, Xtrain, Ytrain, verbose=0)
	display_learning_curves(history)
	_, test_accuracy = model.evaluate(x=Xtest, y=Ytest, verbose=0)
	print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")	
	return history, test_accuracy

#################################### Practice Custom Model ####################################

class PracticeLayer(keras.layers.Layer):
	def __init__(self, adj):
		super(PracticeLayer, self).__init__()
		self.adj =  tf.sparse.from_dense(adj)  # adjacency matrix

	def build(self, input_shape):
		assert(isinstance(self.adj,tf.sparse.SparseTensor))
		num_edges = len(self.adj.indices)
		#print('number of weights should be', input_shape[-1]+ input_shape[-1]*num_edges)
		print('\n\nbuild:',input_shape[0],input_shape[1], num_edges,'\n\n')

		self.w = self.add_weight(
			shape=(input_shape[-1], num_edges),  
			initializer="random_normal",
			trainable=True,
		)
		self.b = self.add_weight(
			shape=(input_shape[-1]), initializer="random_normal", trainable=True
		) # doesn't like shape=(input_shape[-1]) or any variant, so just adding an extra dim and ignoring during call()

		self.sources = self.adj.indices[:,0] # indices of source nodes for each edge
		#edge_indices = np.where(self.adj!=0)
		#self.w_indices = tnp.array([edge_indices[0],edge_indices[1]]) # could also use a constant from tf
		# want to add an array of the non-0 indices, then use to form sq matrix each time
		#	except, should anticipate huge ass network ie use sparse
	

	def call(self, all_inputs):
		# TODO: auto-diffn doesn't work, likely because skipping tf ops...
		# None dim is for dynamic of diff input sizes
		# so need to parallelize with one more dim...but first try w for loop
		x=all_inputs

		if 0:
			x=[]
			print('\ncall: input_shape',all_inputs.shape)
			for i in range(len(all_inputs)):
				inputs = all_inputs[i]

				# inputs are one value per node
				# transmit those node values to their downstream edges and multiply by their weights
				edges_values=inputs[self.sources] #*self.w
				# then map to the adjacency matrix
				# multiplying weights by adjacency matrix in sparse manner by making a new sparse tensor:
				weighted_nghs=tf.sparse.SparseTensor(self.adj.indices,self.adj.values*edges_values,self.adj.dense_shape)
				
				weighted_ngh_sum = tf.sparse.reduce_sum(weighted_nghs, axis=0) 
				# TODO: check if this axis arg is correct
				
				xi = weighted_ngh_sum + self.b[0] # hack soln: self.b has an extra dim due to implementation oddity
			
				x+=[xi]
		
			x=tnp.array(x)

		# DEBUGGING
		#print('\n\nshape0=',edges_values.shape,inputs.shape, self.sources.shape)
		#print('shape1=',weighted_nghs)		
		#print('shape2=',weighted_ngh_sum.shape,self.b.shape[0])
		#print('shape3=',x.shape)

		return x

class PracticeNet(tf.keras.Model):

	def __init__(self, adj, num_classes=3):
		super(PracticeNet, self).__init__()
		self.block_1 = PracticeLayer(adj)
		#self.block_2 = ResNetBlock()
		#self.global_pool = layers.GlobalAveragePooling2D()
		self.classifier = layers.Dense(num_classes)

	def call(self, inputs):
		x = self.block_1(inputs)
		#x = self.block_2(x)
		#x = self.global_pool(x)
		return self.classifier(x)


#################################### Graph Neural Networks ####################################

def build_GNN_model(adjacency_matrix, num_nodes, train_sample, num_classes=2):
	#edge_weights = tf.ones(shape=adjacency_matrix.shape)

	gnn_model = GNNNodeClassifier(
		adjacency_matrix=adjacency_matrix,
		num_classes=num_classes,
		input_shape=[num_nodes],
		hidden_units=PARAMS['hidden_units'],
		dropout_rate=PARAMS['dropout_rate'],
		name="gnn_model",
	)

	train_sample = train_sample[None,:]
	gnn_model(train_sample) # i feel like this extra "None" axis is dumb af

	gnn_model.summary()


class GraphConvLayer(layers.Layer):
	def __init__(
		self,
		hidden_units,
		input_shape,
		dropout_rate=0.2,
		aggregation_type="mean",
		combination_type="concat",
		normalize=False,
		*args,
		**kwargs,
	):
		super(GraphConvLayer, self).__init__(*args, **kwargs)

		self.aggregation_type = aggregation_type
		self.combination_type = combination_type
		self.normalize = normalize

		self.ffn_prepare = ffn(hidden_units, dropout_rate)
		if self.combination_type == "gated":
			self.update_fn = layers.GRU(
				units=hidden_units,
				activation="tanh",
				recurrent_activation="sigmoid",
				dropout=dropout_rate,
				return_state=True,
				recurrent_dropout=dropout_rate,
			)
		else:
			self.update_fn = ffn(input_shape, dropout_rate)

	def prepare(self, node_repesentations, weights=None):
		# node_repesentations shape is [num_edges, embedding_dim].
		messages = self.ffn_prepare(node_repesentations)
		if weights is not None:
			messages = messages * tf.expand_dims(weights, -1)
		return messages

	def aggregate(self, node_data, ngh_data):
		# node_indices shape is [num_edges].
		# neighbour_messages shape: [num_edges, representation_dim].
		num_nodes = len(node_data)
		if self.aggregation_type == "sum":
			ngh_sum = tnp.sum(ngh_data,axis=1) #TODO: check axis

			aggregated_message = node_data + ngh_sum 
			
		else:
			assert(0) #others not yet implemented
		'''
		elif self.aggregation_type == "mean":
			aggregated_message = tf.math.unsorted_segment_mean(
				neighbour_messages, node_indices, num_segments=num_nodes
			)
		elif self.aggregation_type == "max":
			aggregated_message = tf.math.unsorted_segment_max(
				neighbour_messages, node_indices, num_segments=num_nodes
			)
		else:
			raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
		'''
		return aggregated_message

	def update(self, node_repesentations, aggregated_messages):
		# node_repesentations shape is [num_nodes, representation_dim].
		# aggregated_messages shape is [num_nodes, representation_dim].
		if self.combination_type == "gru":
			# Create a sequence of two elements for the GRU layer.
			h = tf.stack([node_repesentations, aggregated_messages], axis=0)
		elif self.combination_type == "concat":
			# Concatenate the node_repesentations and aggregated_messages.
			h = tf.concat([node_repesentations, aggregated_messages], axis=0)
		elif self.combination_type == "add":
			# Add node_repesentations and aggregated_messages.
			h = node_repesentations + aggregated_messages
		else:
			raise ValueError(f"Invalid combination type: {self.combination_type}.")

		# Apply the processing function.
		h=h[None,:]
		node_embeddings = self.update_fn(h)
		if self.combination_type == "gru":
			node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

		if self.normalize:
			node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
		return node_embeddings

	def call(self, inputs):
		"""Process the inputs to produce the node_embeddings.

		inputs: a tuple: (adjacency matrix, input_data, edge_weights)
			where the input data is one cell of RNA seq (i.e. expression level for each node in network)
		Returns: node_embeddings of shape (num_nodes).
		"""

		input_data, adj, edge_weights = inputs
		input_data = tnp.array(input_data)[0]
		neighbour_data = input_data[adj]

		# Prepare the messages of the neighbours.
		#neighbour_messages = self.prepare(neighbour_data, edge_weights)

		# Aggregate the neighbour messages.
		aggregated_messages = self.aggregate(input_data, neighbour_data)
		# Update the node embedding with the neighbour messages.
		return self.update(input_data, aggregated_messages)

	def build(self, input_shape):
		# TODO: add this
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="random_normal",
			trainable=True,
		)
		self.b = self.add_weight(
			shape=(self.units,), initializer="random_normal", trainable=True
		)

class GNNNodeClassifier(tf.keras.Model):
	def __init__(
		self,
		adjacency_matrix,
		num_classes,
		hidden_units,
		input_shape,
		aggregation_type="sum",
		combination_type="concat",
		dropout_rate=0.2,
		normalize=True,
		*args,
		**kwargs,
	):
		super(GNNNodeClassifier, self).__init__(*args, **kwargs)

		# Unpack graph_info to three elements: node_features, edges, and edge_weight.
		
		self.adj = adjacency_matrix.toarray() # change to dense representation
		self.edge_weights = None # for now at least
		if self.edge_weights is None:
			self.edge_weights = tf.ones(shape=adjacency_matrix.shape[1])
		# Scale edge_weights to sum to 1.
		self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

		# Create a process layer.
		self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name='input')
		self.preprocess = ffn(input_shape, dropout_rate, name="preprocess")
		# Create the first GraphConv layer.
		self.conv1 = GraphConvLayer(
			hidden_units,
			input_shape,
			dropout_rate,
			aggregation_type,
			combination_type,
			normalize,
			name="graph_conv1",
		)
		# Create the second GraphConv layer.
		self.conv2 = GraphConvLayer(
			hidden_units,
			input_shape,
			dropout_rate,
			aggregation_type,
			combination_type,
			normalize,
			name="graph_conv2",
		)
		# Create a postprocess layer.
		self.postprocess = ffn(hidden_units, dropout_rate, name="postprocess")
		# Create a compute logits layer.
		self.compute_logits = layers.Dense(units=num_classes, name="logits")

	def call(self, x):
		x = self.preprocess(x)
		# Apply the first graph conv layer.
		x1 = self.conv1((x, self.adj, self.edge_weights))
		# Skip connection.
		x = x1 + x
		# Apply the second graph conv layer.
		x2 = self.conv2((x, self.adj, self.edge_weights))
		# Skip connection.
		x = x2 + x
		# Postprocess node embedding.
		x = self.postprocess(x)
		return self.compute_logits(x)

#################################### Baseline Model ####################################

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

def display_learning_curves(history):
	#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.legend(["train", "validation"], loc="upper right")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.show()
	plt.clf()

	if 0: # skip accuracy plots
		plt.plot(history.history["acc"])
		plt.plot(history.history["val_acc"])
		plt.legend(["train", "test"], loc="upper right")
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
	
		plt.show()
		plt.clf()

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


#################################### Data Handling ####################################

def get_toy_data(soln='sum',use_dense=True):
	num_nodes, num_cells, edgepr = 100, 10000, .01
	#G = nx.erdos_renyi_graph(num_nodes, edgepr) 
	G = nx.barabasi_albert_graph(num_nodes, int(num_nodes*edgepr))
	A = nx.adjacency_matrix(G)
	if use_dense:
		A= np.array(A.todense()) 
		# scipy be dumb, A[0]=A, for example A.shape=(100,100), A[0].shape=(1,100) \:
		# soln is to explicitly cast to numpy

	ssRNA = np.random.normal(size=(num_cells, num_nodes))

	labels = np.random.choice([0,1,2],size=num_cells,p=[.3,.1,.6]) # where 0 refers to normal and 1 to cancerous or something

	# TODO: harder toy problems, incld nonlinear	
	if soln=='sum':
		summ = np.sum(ssRNA,axis=1)
		labels_1 = (summ>.1)
		labels_2 = (summ>.8)
		labels = np.zeros(labels.shape)+labels_1+labels_2
	elif soln=='graph':
		topo_sum = np.sum(A,axis=1)
		assert(0) # TODO: how to make toy data that depends on graph? couldn't a regular NN learn the same weights?
		labels = (topo_sum>np.mean(topo_sum))
		print(labels.shape,len(ssRNA))
		#print('balance=',sum(labels)/len(labels))
	else:
		assert(0)

	labels = convert_to_1hot(labels)

	return A, ssRNA, labels, num_nodes


def convert_to_1hot(labels):
	onehot=[]
	for l in labels:
		a=[0,0,0]
		a[int(l)]=1
		onehot+=[a]
	return np.array(onehot)

def split_data(ssRNA, labels, train_split=.1,shuffle=False,validation=True, test_split=.3):
	# splits into train and test sets
	# validation is split within the neural network functions
	assert(shuffle==False) # have not implemented, but need to if data is not shuffled 
	assert(len(ssRNA)==len(labels)) 

	Xtrain, Xtest = ssRNA[:int(train_split*len(ssRNA))],ssRNA[int(train_split*len(ssRNA)):]

	Ytrain, Ytest = labels[:int(train_split*len(labels))], labels[int(train_split*len(labels)):]

	if validation:
		Xtest, Xval = Xtest[:int(test_split*len(Xtest))], Xtest[int(test_split*len(Xtest)):,]
		Ytest, Yval =  Ytest[:int(test_split*len(Ytest))], Ytest[int(test_split*len(Ytest)):,]
		print('number of samples:',len(Ytrain),'training',len(Yval),'validation',len(Ytest),'test.')
		return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest

	print('number of samples:',len(Ytrain),'training',len(Ytest),'test.')
	return Xtrain,  Xtest, Ytrain, Ytest

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 GNNtest.py ... haven't decided yet")

	run_practice()
	#run_toy_model()
