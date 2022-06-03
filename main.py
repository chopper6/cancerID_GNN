# call this function to execute a model. Usage:
# 	python3 main.py [GNN | FFNN]

# modify the PARAMS below to fine-tune it

import sys
import util, gnn, ffnn
import tensorflow as tf
from copy import deepcopy

PARAMS = {
	'GNN' : {
		'number_graph_layers' : 1,
		'learning_rate': 0.001,
		'dropout_rate' : 0.1, # 0.05,
		'graph_activation_function':'linear',  # options: 'linear', 'relu', 'sigmoid', 'tanh'
		'num_epochs' : 20,
		'batch_size' : int(2**4),
		'trainable dense': False, # True, # can turn off to make sure model learns by only training graph convolution weights
		'cutoff':1 
	},
	'FFNN' : {
		'hidden_units' : [int(2**5),int(2**5)],
		'learning_rate': 0.001,
		'dropout_rate' : 0.1,
		'num_epochs' : 5,
		'batch_size' : int(2**5),
	}
}

def hyperparam_search_gnn():

	log_file = './hyperparam_log.txt'
	with open(log_file,'w') as f:
		f.write("GNN Hyperparameter Search")
	hypers = {'number_graph_layers':[1,2,3],'dropout_rate':[0,.1],'graph_activation_function':['linear','sigmoid','tanh'] }
	params_orig = PARAMS['GNN']
	print('...Initializing...')
	A, X, Y = util.get_real_data(params_orig)
	Xtrain, Xval, Xtest, Ytrain, Yval, Ytest= util.split_data(params_orig, X, Y, train_split=.7, test_split=.3) # .7, .3 by default

	min_err = 100000
	best_params=None
	print('...Starting Hyperparam Search...')
	for k in hypers.keys():
		for v in hypers[k]:
			params=deepcopy(params_orig)
			params[k] = v

			in_shape= (params['batch_size'],len(Xtrain[0]))
			gnn_model = gnn.GNN(params, A, in_shape)
			
			val_err = run_model(gnn_model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, full=False)

			s="Val err with " + str(k) + '=' + str(v) + ':' + str(val_err)
			with open(log_file,'a') as f:
				f.write(s)
			print(s)
			if val_err < min_err: 
				min_err = val_err
				best_params = (k,v)

	print("\nBest param change:", best_params)


def run(model_type):
	params = PARAMS[model_type]
	#A, X, Y = util.get_toy_data(params)
	print('...Getting data...')
	A, X, Y = util.get_real_data(params)
	Xtrain, Xval, Xtest, Ytrain, Yval, Ytest= util.split_data(params, X, Y, train_split=.7, test_split=.3, cutoff=params['cutoff']) # .7, .3 by default

	print('\n')
	print('...Initializing model...')
	if model_type=='GNN':
		in_shape= (params['batch_size'],len(Xtrain[0]))
		model = gnn.GNN(params, A, in_shape)
	else:
		model = ffnn.build_ffnn(hidden_units, dropout_rate, num_classes=5)
	print('\n')

	train_model(params, model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, full=True)


def train_model(params, model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, full=True):
	model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

	if full:
		print("...Checking Untrained error...")
		untrained_err = model.evaluate(x=Xtest, y=Ytest, verbose=1, batch_size=params['batch_size'])
		model.summary()

	print("...Training Model...")
	history = model.fit(
		Xtrain,
		Ytrain,
		batch_size=params['batch_size'],
		epochs=params['num_epochs'],
		validation_data=(Xval, Yval), # instead of validation_split, since need to evenly fit for batch_size
		callbacks= tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
	)

	if full:
		util.display_learning_curves(history)
		test_err = model.evaluate(x=Xtest, y=Ytest, verbose=1, batch_size=params['batch_size'])
		print(f"Test error/untrained error: {round(test_err/untrained_err, 5)}")
	else:
		return history.history["val_loss"][-1]


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 GNNtest.py [GNN | FFNN]")

	if sys.argv[1].upper()=='GNN':
		run('GNN')
	elif sys.argv[1].upper()=='FFNN':
		run('FFNN')
