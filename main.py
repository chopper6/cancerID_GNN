# call this function to execute a model. Usage:
# 	python3 main.py [GNN | FFNN]

# modify the PARAMS below to fine-tune it

import sys
import util, gnn, ffnn, logreg
import tensorflow as tf
from copy import deepcopy

PARAMS = {
	'global' : {
		'batch_size' : int(2**4), #**5
		'cutoff':1,
		'toy_data':False 
	},
	'GNN' : {
		'number_graph_layers' : 1,  
		'learning_rate': 0.00001, #0.000001
		'dropout_rate' : .3, #only applied to final dense layer
		'graph_activation_function':'linear',  # options: 'linear', 'relu', 'sigmoid', 'tanh'
		'num_epochs' : 200,
		'patience' : 40,
		'trainable dense': True, # True, # can turn off to make sure model learns by only training graph convolution weights
	},
	'FFNN' : {
		'hidden_units' : [int(2**4)], #,int(2**5)],
		'learning_rate': 0.0001, #0.00001
		'dropout_rate' : .5,
		'num_epochs' : 20,
		'patience' : 4,
	},
	'LOGREG' : {
		'max_iter' : 1000
	}
}


def run(model_type):
	params = PARAMS['global']
	params = {**params, **PARAMS[model_type]}
	print('...Getting data...')
	if params['toy_data']:
		A, X, Y = util.get_toy_data(params)
	else:
		A, X, Y = util.get_real_data(params)
	Xtrain, Xval, Xtest, Ytrain, Yval, Ytest= util.split_data(params, X, Y, train_split=.7, test_split=.3, cutoff=params['cutoff']) # .7, .3 by default

	print('\n')
	print('...Initializing model...')
	if model_type=='GNN':
		in_shape= (params['batch_size'],len(Xtrain[0]))
		model = gnn.GNN(params, A, in_shape)
	elif model_type=='FFNN':
		model = ffnn.build_ffnn(params['hidden_units'], params['dropout_rate'], num_classes=5)
	elif model_type =='LOGREG':
		logreg.run_logreg(params['max_iter'],Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)

	print('\n')
	if model_type != 'LOGREG':
		train_model(params, model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, test=True)


def train_model(params, model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, test=True):
	model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(params['learning_rate']), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

	if test:
		test_CE_err, test_acc = model.evaluate(x=Xtest, y=Ytest, verbose=1, batch_size=params['batch_size'])
		print("before training cross entropy error =", test_CE_err, ", fraction of correct labels =,", test_acc)
		model.summary()

	print("...Training Model...")
	history = model.fit(
		Xtrain,
		Ytrain,
		batch_size=params['batch_size'],
		epochs=params['num_epochs'],
		verbose=2,
		validation_data=(Xval, Yval), # instead of validation_split, since need to evenly fit for batch_size
		callbacks= tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=params['patience'])
	)

	if test:
		util.display_learning_curves(history)
		test_CE_err, test_acc = model.evaluate(x=Xtest, y=Ytest, verbose=1, batch_size=params['batch_size'])

		print("Test cross entropy error =", test_CE_err, ", fraction of correct labels =,", test_acc)
	else:
		return history.history["val_loss"][-1], history.history['val_acc'][-1]


def hyperparam_search_gnn():
	# not used much, mainly just searched manually

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
			
			val_err = run_model(gnn_model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, test=False)

			s="Val err with " + str(k) + '=' + str(v) + ':' + str(val_err)
			with open(log_file,'a') as f:
				f.write(s)
			print(s)
			if val_err < min_err: 
				min_err = val_err
				best_params = (k,v)

	print("\nBest param change:", best_params)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 GNNtest.py [GNN | FFNN]")

	if sys.argv[1].upper()=='GNN':
		run('GNN')
	elif sys.argv[1].upper()=='FFNN':
		run('FFNN')
	elif sys.argv[1].upper()=='LOGREG':
		run('LOGREG')
	else:
		sys.exit("Unrecognized argument. Usage: python3 GNNtest.py [GNN | FFNN]")
