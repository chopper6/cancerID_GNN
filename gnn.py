# basically implemented attention: https://arxiv.org/pdf/1710.10903.pdf

# data: 
# a graph file that gives the adjacency matrix
# ssRNA data that gives normalized RNA values (in 0,1) for each node in the network
#	with corresponding tags for cancer/non-cancer (or w.e classification)

# TODO:
# more cleaning
# train w dense layer non-trainable
# implmt ffwdNN, likey logistic reg too
# !!check if this axis arg in call() is correct!! 

# MaybeDO:
# normz w.r.t ngh size
# nonlinear op, likely sigmoid
# dropout
# add self loops
# 1-3 GNN layer
# simple hyperparam loop/trainer
# have the toy problem use a feature of the topo?

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()  # numpy style indexing and such
#tf.debugging.enable_check_numerics() # recommended while debugging


#################################### Graph Neural Network ####################################

class GNNLayer(tf.keras.layers.Layer):
	def __init__(self, adj, input_shape, activation_fn):
		super(GNNLayer, self).__init__()
		self.adj =  tf.sparse.from_dense(adj)  # adjacency matrix

		assert(isinstance(self.adj,tf.sparse.SparseTensor))
		num_edges = len(self.adj.indices)
		self.batch_size = input_shape[0]

		self.activation_fn = activation_fn

		self.w = self.add_weight(
			shape=(num_edges,),  
			initializer="random_normal",
			trainable=True,
			name='GNN conv weights'
		)
		self.b = self.add_weight(
			shape=(input_shape[-1],), initializer="random_normal", trainable=True, name='GNN node bias'
		) # doesn't like shape=(input_shape[-1]) or any variant, so just adding an extra dim and ignoring during call()

		self.sources = self.adj.indices[:,0] # indices of source nodes for each edge
		self.adj3d_shape = tf.constant([input_shape[0],int(self.adj.dense_shape[0]),int(self.adj.dense_shape[1])],dtype=tf.int64)
		# for some reason a later sparse reduce_sum() requires both shape and indices to be int64 lol

		# lazy but only have to run this once
		# the larger the batch size, the slower the loop
		adj3d_indices = []
		for i in range(input_shape[0]):
			adj3d_indices += [[i,int(ind[0]),int(ind[1])] for ind in self.adj.indices]
		self.adj3d_indices = tf.constant(adj3d_indices,dtype=tf.int64)

	
	def call(self, inputs):
		num_nodes, num_edges = len(inputs[0]), len(self.adj.indices)
		edges_values=inputs[:,self.sources] 
		weighted_edges = (self.adj.values*edges_values*self.w)
		weighted_edges = weighted_edges.reshape(num_edges*self.batch_size) # basically flatten it
		# TODO: checked that weighted edges are flattened in such a way that they match self.adj3d_indices
		
		weighted_nghs=tf.sparse.SparseTensor(self.adj3d_indices, weighted_edges, self.adj3d_shape)
		weighted_ngh_sum = tf.sparse.reduce_sum(weighted_nghs, axis=1) 
		# TODO: check if this axis arg is correct (either 1 or 2, not 0 since that is for the instances)

		x = weighted_ngh_sum + self.b
		# TODO: is bias really nec?

		if self.activation_fn == 'sigmoid':
			x = tf.math.sigmoid(x)
		elif self.activation_fn == 'tanh':
			x = tf.keras.activations.tanh(x) 
		elif self.activation_fn == 'relu':
			x = tf.nn.relu(x) 
		else:
			assert(self.activation_fn == 'linear') #otherwise activation function is not recognized

		# TODO: nonlinear transf of some kind

		# DEBUG PRINTS:		
		#print('\n\n#instances, #nodes, #edges=',self.batch_size, num_nodes, num_edges)
		#print('\ninput shape within call:',inputs.shape)
		#print('\n\nshapez:',self.adj.dense_shape, edges_values.shape, inputs.shape)
		#print('\n\nweighted edge sshape=',weighted_edges.shape, 'vs flat:',num_edges*num_instances,'\n\n')
		#print('\nmoar shapez',weighted_edges.shape, self.adj.values.shape, edges_values.shape)
		#print('3 sections:\n',self.adj3d_indices,'\n',weighted_edges,'\n',self.adj3d_shape)
		#print('w ngh sum shape=',weighted_ngh_sum.shape)
			
		return x

class GNN(tf.keras.Model):

	def __init__(self,PARAMS, adj, input_shape,  num_classes=3):
		super(GNN, self).__init__()
		self.num_graph_layers = PARAMS['number_graph_layers']
		self.graph_layers, self.dropouts = [], []
		for i in range(self.num_graph_layers):
			self.graph_layers += [GNNLayer(adj, input_shape, PARAMS['graph_activation_function'])]
			self.dropouts += [layers.Dropout(PARAMS['dropout_rate'])]
			
		if PARAMS['trainable dense']:
			self.classifier = layers.Dense(num_classes) 
		else:
			self.classifier = layers.Dense(num_classes, trainable=False)

	def call(self, inputs):
		x = inputs 
		for i in range(self.num_graph_layers):
			x = self.graph_layers[i](x)
			x = self.dropouts[i](x)
		x = self.classifier(x)
		return x


