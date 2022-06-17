# miscellaneous functions such as plotting, toy data, and data manipulations

import numpy as np
import networkx as nx
import math, pickle
from matplotlib import pyplot as plt

def display_learning_curves(history,logscale=False):
	#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

	plt.plot(history.history["loss"][1:], color='#009999',linewidth=3,alpha=.8)   #00cc00
	plt.plot(history.history["val_loss"][:-1], color='#cc0066',linewidth=3,alpha=.8)   ##cc0099
	# since validation is measured at the end of epoch, whereas training is during the epoch, offset each by 1
	plt.legend(["training", "validation"], loc="upper right")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	if logscale:
		ax = plt.gca()
		ax.set_yscale('log')
	plt.savefig('training_history_long_ffnn2.png')
	plt.clf()




def get_real_data(params, use_dense=True):
	net_file = './GRN'
	data_file = './scRNA.pickle'
	G = nx.read_edgelist(net_file)

	with open(data_file,'rb') as f:
		data = pickle.load(f)
		X, Y, gene_order = data['scRNA'], data['labels'], data['genes']
		#print('\n\ngot from pickle:',X.shape, len(Y), len(gene_order))
		#print("for ex", X[8,8], Y[8], gene_order[8])

	A = nx.adjacency_matrix(G,nodelist=gene_order)
	if use_dense:
		A= np.array(A.todense()) 
		# scipy be dumb, A[0]=A, for example A.shape=(100,100), A[0].shape=(1,100) \:
		# soln is to explicitly cast to numpy

	X = X.astype(float)

	unique, counts = np.unique(Y,axis=0,return_counts=True)
	print("\tFrequency of each label:",counts) #,'\nunique=',unique)


	#print("from original G, #nodes, #edges, #gene order=",len(G.nodes()),len(G.edges()), len(gene_order))
	#print("vs X shape, Y shape:", X.shape, Y.shape)
	assert(len(G.nodes()) == len(X[0]))
	assert(len(X) == len(Y))

	return A, X, Y


def scRNA_from_raw():
	net_file = './GRN'
	raw_file = './scRNA_raw.csv'
	data_file = './scRNA.pickle'
	net_file = './GRN'
	G = nx.read_edgelist(net_file)
	genes_in_network = list(G.nodes())

	with open(raw_file, 'r') as f:
		i=0
		gene_order = []
		data = []
		for line in f:
			line = line.rstrip().split(',')
			line = [l.replace("'",'').replace('"','') for l in line]
			if i%1000==0:
				print('at line',i)

			if i==0:
				cell_types = line[1:]
			else:
				if line[0] in genes_in_network:
					gene_order += [line[0]]
					data += [line[1:]]
			i+=1

			#if i==1000:
			#	break
		data = np.array(data).T # flip so each row is an instance
		print('filtered data shape:',data.shape)

	to_rm = []
	for g in genes_in_network:
		if g not in gene_order:
			to_rm +=[g]
	for g in to_rm:
		G.remove_node(g)
		
	for e in G.edges():
		assert(e[0] in G.nodes())
		assert(e[1] in G.nodes())
	nx.write_edgelist(G, net_file)
	assert(len(G.nodes())==len(data[0]))

	cell_types, unq_labels = convert_to_1hot(cell_types)
	with open(data_file, 'wb') as f:
		pickle.dump({'scRNA':data,'genes':gene_order,'labels':cell_types}, f)


def GRN_from_raw():
	raw_file = './GRNraw.csv'
	net_file = './GRN'
	#edges = []
	#all_nodes = []
	zscore_thresh, corr_thresh = 10,.1 #20,.2 (.5/50 for less data)
	i=0
	G=nx.empty_graph(create_using=nx.DiGraph())
	with open(raw_file, 'r') as f:
		for line in f:
			if i!=0:
				line = line.rstrip().split(',')
				if float(line[2]) > zscore_thresh and float(line[3]) > corr_thresh:
					#edges += [line[:2]]
					G.add_edge(line[0],line[1])
				#all_nodes += [line[0]]
				#all_nodes += [line[1]]
			i+=1
	#genes = set(all_nodes)
	#print('orig #genes =',len(genes))
	print('# edges=',len(G.edges()), '# nodes=',len(G.nodes()))
	nx.write_edgelist(G, net_file)
	


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
	scRNA = np.random.normal(size=(num_cells, num_nodes))

	labels = np.random.choice([0,1,2],size=num_cells,p=[.3,.1,.6]) # where 0 refers to normal and 1 to cancerous or something

	if soln=='sum':
		summ = np.sum(scRNA,axis=1)
		labels_1 = (summ>-.1)
		labels_2 = (summ>-.05)
		labels_3 = (summ>0)
		labels_4 = (summ>.05)
		#labels_5 = (summ>.1)
		labels = np.zeros(labels.shape)+labels_1+labels_2+labels_3+labels_4 #+labels_5
	elif soln=='graph':
		topo_sum = np.sum(A,axis=1)
		assert(0) 
		labels = (topo_sum>np.mean(topo_sum))
		print(labels.shape,len(scRNA))
		#print('balance=',sum(labels)/len(labels))
	else:
		assert(0)

	labels, unq_labels = convert_to_1hot(labels)

	return A, scRNA, labels


def convert_to_1hot(labels):
	onehot=[]
	unq_labels = list(set(labels))
	for l in labels:
		a=[0 for _ in range(len(unq_labels))]
		a[unq_labels.index(l)]=1
		onehot+=[a]
	return np.array(onehot), unq_labels

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def trim_by_modulo(arr,m):
	rem = len(arr)%m 
	if rem==0:
		return arr
	return arr[:-rem]

def split_data(params, scRNA, labels, train_split=.7,shuffle=True,validation=True, test_split=.3, cutoff=None):
	# splits into train and test sets
	# validation is split within the neural network functions

	if shuffle:
		scRNA, labels = unison_shuffled_copies(scRNA, labels) 

	if cutoff is not None:
		scRNA = scRNA[:int(cutoff*len(scRNA))]
		labels = labels[:int(cutoff*len(labels))]

	# number of samples has to be evenly divisible by batch size
	num_train = int(math.floor(int(train_split*len(scRNA))/params['batch_size'])*params['batch_size'])

	Xtrain, Xtest = scRNA[:num_train],scRNA[num_train:]

	Ytrain, Ytest = labels[:num_train], labels[num_train:]

	if validation:
		# number of samples has to be evenly divisible by batch size
		num_test = int(math.floor(int(test_split*len(Xtest))/params['batch_size'])*params['batch_size'])

		Xtest, Xval = Xtest[:num_test], Xtest[num_test:,]
		Ytest, Yval =  Ytest[:num_test], Ytest[num_test:,]
		Xval = trim_by_modulo(Xval,params['batch_size'])
		Yval = trim_by_modulo(Yval,params['batch_size'])
		print('number of samples:',len(Ytrain),'training',len(Yval),'validation',len(Ytest),'test.')
		assert(len(Xtrain)>0) # if batch_size > amount of data this might be false
		assert(len(Xtest)>0) # if batch_size > amount of data this might be false
		assert(len(Xval)>0) # if batch_size > amount of data this might be false
		assert(len(Xtrain)%params['batch_size']==0) # data must be evenly divisible by batch_size
		assert(len(Xval)%params['batch_size']==0) # data must be evenly divisible by batch_size
		assert(len(Xtest)%params['batch_size']==0) # data must be evenly divisible by batch_size
		return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest

	print('number of samples:',len(Ytrain),'training',len(Ytest),'test.')
	return Xtrain,  Xtest, Ytrain, Ytest



if __name__ == "__main__":
	# generally don't call this file directly, just for preprocessing
	print("Calling util.py directly should be reserved for preprocessing.")
	GRN_from_raw()
	scRNA_from_raw()
	#real_data({})