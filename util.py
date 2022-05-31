# miscellaneous functions such as plotting, toy data, and data manipulations

import numpy as np
import networkx as nx
import math
from matplotlib import pyplot as plt

def display_learning_curves(history):
	#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

	plt.plot(history.history["loss"][1:], color='#009999',linewidth=3,alpha=.8)   #00cc00
	plt.plot(history.history["val_loss"][:-1], color='#cc0066',linewidth=3,alpha=.8)   ##cc0099
	# since validation is measured at the end of epoch, whereas training is during the epoch, offset each by 1
	plt.legend(["training", "validation"], loc="upper right")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	ax = plt.gca()
	ax.set_yscale('log')
	plt.show()
	plt.clf()


def get_toy_data(params, soln='sum',use_dense=True):
	# possible additions: 
	# toy data that depends on graph, some nonlinearity
	num_nodes, num_cells, edgepr = 100, 10000, .01
	num_cells = int(num_cells/params['batch_size'])*params['batch_size']
	assert(num_cells%params['batch_size']==0)
	#G = nx.erdos_renyi_graph(num_nodes, edgepr) 
	G = nx.barabasi_albert_graph(num_nodes, int(num_nodes*edgepr))
	A = nx.adjacency_matrix(G)
	if use_dense:
		A= np.array(A.todense()) 
		# scipy be dumb, A[0]=A, for example A.shape=(100,100), A[0].shape=(1,100) \:
		# soln is to explicitly cast to numpy

	np.random.seed(323344645)
	ssRNA = np.random.normal(size=(num_cells, num_nodes))

	labels = np.random.choice([0,1,2],size=num_cells,p=[.3,.1,.6]) # where 0 refers to normal and 1 to cancerous or something
	
	if soln=='sum':
		summ = np.sum(ssRNA,axis=1)
		labels_1 = (summ>.1)
		labels_2 = (summ>.8)
		labels = np.zeros(labels.shape)+labels_1+labels_2
	elif soln=='graph':
		topo_sum = np.sum(A,axis=1)
		assert(0) 
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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_data(params, ssRNA, labels, train_split=.7,shuffle=True,validation=True, test_split=.3):
	# splits into train and test sets
	# validation is split within the neural network functions
	if shuffle:
		ssRNA, labels = unison_shuffled_copies(ssRNA, labels) 

	# number of samples has to be evenly divisible by batch size
	num_train = int(math.floor(int(train_split*len(ssRNA))/params['batch_size'])*params['batch_size'])

	Xtrain, Xtest = ssRNA[:num_train],ssRNA[num_train:]

	Ytrain, Ytest = labels[:num_train], labels[num_train:]

	if validation:
		# number of samples has to be evenly divisible by batch size
		num_test = int(math.floor(int(test_split*len(Xtest))/params['batch_size'])*params['batch_size'])

		Xtest, Xval = Xtest[:num_test], Xtest[num_test:,]
		Ytest, Yval =  Ytest[:num_test], Ytest[num_test:,]
		print('number of samples:',len(Ytrain),'training',len(Yval),'validation',len(Ytest),'test.')
		return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest

	print('number of samples:',len(Ytrain),'training',len(Ytest),'test.')
	return Xtrain,  Xtest, Ytrain, Ytest