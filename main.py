# call this function to execute a model. Usage:
# 	python3 GNNtest.py [GNN | FFNN | LOGREG]

# modify the PARAMS below to fine-tune it

import sys
import util, gnn, ffnn
import tensorflow as tf

PARAMS = {
	'GNN' : {
		'number_graph_layers' : 2,
		'learning_rate': 0.001,
		'dropout_rate' : 0.05,
		'graph_activation_function':'sigmoid',  # options: 'linear', 'relu', 'sigmoid', 'tanh'
		'num_epochs' : 3,
		'batch_size' : int(2**4),
		'validation_split': 0.2,
		'trainable dense': True, # can turn off to make sure model learns by only training graph convolution weights 
	},
	'FFNN' : {
		'hidden_units' : [int(2**5),int(2**5)],
		'learning_rate': 0.001,
		'dropout_rate' : 0.1,
		'num_epochs' : 5,
		'batch_size' : int(2**5),
		'validation_split': 0.2,
	},
	'LOGREG' : {
		'regularization': 1
	}
}


def run_gnn():
	params = PARAMS['GNN']
	A, ssRNA, labels, num_nodes = util.get_toy_data(params)
	Xtrain, Xval, Xtest, Ytrain, Yval, Ytest= util.split_data(params, ssRNA, labels)
	assert(len(Xtrain)>0) # if batch_size > amount of data this might be false
	assert(len(Xtrain)%params['batch_size']==0) # data must be evenly divisible by batch_size

	print('\n')
	in_shape= (params['batch_size'],len(Xtrain[0]))
	gnn_model = gnn.GNN(params, A, in_shape)
	print('\n')

	gnn_model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
	#print('inshape=',in_shape) #Xtrain[:64].shape)
	#pnet.build(in_shape)

	untrained_err = gnn_model.evaluate(x=Xtest, y=Ytest, verbose=1, batch_size=params['batch_size'])
	gnn_model.summary()
	#print('\nws before',pnet.trainable_variables) 
	history = gnn_model.fit(
		Xtrain,
		Ytrain,
		batch_size=params['batch_size'],
		epochs=params['num_epochs'],
		validation_data=(Xval, Yval), # instead of validation_split, since need to evenly fit for batch_size
	)
	#print('\nws after:',pnet.trainable_variables) 
	util.display_learning_curves(history)
	test_err = gnn_model.evaluate(x=Xtest, y=Ytest, verbose=1, batch_size=params['batch_size'])
	print(f"Test error/untrained error: {round(test_err/untrained_err, 5)}")



if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 GNNtest.py [GNN | FFNN | LOGREG]")

	if sys.argv[1].upper()=='GNN':
		run_gnn()
	elif sys.argv[1].upper()=='FFNN':
		run_ffnn()
	elif sys.argv[1].upper()=='LOGREG':
		run_logreg()
