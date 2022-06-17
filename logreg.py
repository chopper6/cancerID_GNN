from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as CE
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt


def run_logreg(max_iter, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest):

	Ytrain, Yval, Ytest = oneHot2dense(Ytrain), oneHot2dense(Yval), oneHot2dense(Ytest)
	Cs = [0] + [10**i for i in range(-5,3)]
	best_val, best_C = 1000, None
	train_errs, val_errs = [],[]
	for C in Cs:
		print('at C=',C)
		penalty = 'l2'
		if C==0:
			penalty = 'none'
		model = LogisticRegression(C=C,penalty=penalty,max_iter=max_iter) #,solver='saga')
		fitted_model, train_err, val_err = train(model, Xtrain, Xval, Ytrain, Yval)
		train_errs += [train_err]
		val_errs += [val_err]
		if val_err < best_val:
			best_val = val_err
			best_C = C

	plot_bars(Cs,train_errs,val_errs,'C')

	print('optimal C=',best_C)
	model = LogisticRegression(C=C,penalty='l2',max_iter=max_iter)
	fitted_model, train_err, val_err = train(model, Xtrain, Xval, Ytrain, Yval)
	test_err, test_acc = check_test_error(fitted_model, Xtest, Ytest)
	print("\nFinal test cross ent error=", test_err/5, 'and accuracy=', test_acc)
	# note that to be comparable to tensorflow cross entropy, need to divide by number of classes (5)


def train(model, Xtrain, Xval, Ytrain, Yval):
	fitted_model = model.fit(Xtrain, Ytrain)
	train_err = CE(Ytrain,fitted_model.predict_proba(Xtrain)) #/len(X[train_i])
	val_err = CE(Yval,fitted_model.predict_proba(Xval)) #/len(X[val_i])
	return fitted_model, train_err, val_err
	
def check_test_error(fitted_model, x_test, y_test):
	y_pred =  fitted_model.predict_proba(x_test)
	ce_err = CE(y_test,y_pred) #/len(x_test)
	for i in range(len(y_pred)):

		max_ind = np.argmax(y_pred[i])
		y=np.zeros(len(y_pred[i]))
		y[max_ind] = 1
		y_pred[i] = y
	y_pred = oneHot2dense(y_pred)
	acc = accuracy_score(y_test, y_pred)
	return ce_err, acc

def oneHot2dense(Y):
	newY=[]
	for y in Y:
		newY+=[list(y).index(1)]
	return newY

def plot_bars(hyperparam,train,val,xlabel, logy=False):
	# compare validation and training data over some hyperparameter range
	x = np.arange(len(hyperparam))
	width = 0.2 
	fig, ax = plt.subplots()
	ax.bar(x - width/2, train, width,label='train',color='#009999')
	ax.bar(x + width/2, val, width, label='validation',color='#cc0066')

	ax.set_ylabel('error',fontsize=16)
	ax.set_xlabel(xlabel,fontsize=16)
	ax.set_xticks(x)
	ax.set_xticklabels(hyperparam,fontsize=10)
	if logy:
		ax.set_yscale('log')
	ax.legend()
	ax.legend(prop={'size': 18})

	fig.tight_layout()
	plt.savefig("LogReg Parameter Search")
