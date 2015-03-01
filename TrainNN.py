import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
import pandas as pd


def GetTitle(names):
	titles = []
	for n in names:
		i=0
		while n[i]!=',':
			i+=1
		titles.append(str(n[i+2:i+5]))
	return np.array(titles)

def MakeFeatures(Raw):
	#features = []
	# Get titles, make dummy variables.
	N = len(Raw)
	Raw['Title'] = GetTitle(Raw['Name'])
	title_df=pd.get_dummies(Raw['Title'])
	# Keep only useful titles
	title_df = title_df[['Mr.', 'Mrs', 'Mas', 'Mis', 'Dr.']]
	# Add Class
	Pclass_df = pd.get_dummies(Raw['Pclass'], prefix = "Pclass")
	# Add Fares
	features = pd.concat([title_df,Pclass_df],axis=1)
	# Get Fares, normalize them and add them to features
	fares = np.array(Raw['Fare'])
	norm = (np.divide(np.ones([N,]),1+np.exp(-0.01*fares))-0.5)*2	
	features['Fare'] = norm
	# get cabin info
	#cabin = np.array(map(str,np.array(Raw['Cabin'])))
	#iscabin = np.array(map(int,cabin!='nan'))
	#features['IsCabin'] = iscabin
	#print features


	return features

def sigmoid(A): return np.divide(np.ones(np.shape(A)),1+np.exp(-A))

def NNFwdProp(data, weights, hidden = [10,10]):

	s = np.shape(data)[1]
	N = len(data)
	h1 = hidden[0]
	h2 = hidden[1]

	w1 = np.reshape(weights[0:((s+1)*h1)], [s+1, h1])
	w2 = np.reshape(weights[((s+1)*h1):((s+1)*h1)+(h1+1)*h2], [h1+1, h2])
	w3 = np.reshape(weights[-(h2+1):], [h2+1,1])
	
	# Add bias to first layer
	X = np.c_[np.ones(N), data]
	#Compute output of first hidden layer
	X1 = sigmoid(np.dot(X,w1))
	# Add bias to second layer
	X = np.c_[np.ones(N), X1]
	#Compute output of second hidden layer
	X2 = sigmoid(np.dot(X,w2))
	# Add bias to final layer
	X = np.c_[np.ones(N), X2]
	#Compute final output
	y = sigmoid(np.dot(X,w3))

	return np.ravel(y)

def err(weights, train, response, hidden):

	ypred = NNFwdProp(train, weights, hidden)
	mse = np.sum(np.square(ypred - response))
	return mse


def TrainNN(train,response,hidden = [10,10]):
	
	S = np.shape(train)[1]
	h1 = hidden[0]
	h2 = hidden[1]
	nw1 = (S+1)*h1
	nw2 = (h1+1)*h2
	nw3 = (h2+1)
	np.random.seed(42)
	init_weights = np.random.randn(nw1+nw2+nw3)
	res = minimize(err, init_weights, args = (train, response, hidden), options = {'maxiter':1000})
	return res.x


def main():
	print 'Reading train data...'
	train = pd.read_csv('train.csv')
	print 'Extracting features...'
	features = MakeFeatures(train)
	response = np.array(train['Survived'])

	tr, cv, y_tr, y_cv = train_test_split(features,response, test_size=0.20, random_state=123)

	print 'Training the Neural Network...'
	NNshape = [20,20]
	best_weights = TrainNN(np.array(tr),y_tr,hidden = NNshape)
	'''
	print '\n\nEstimated best weights are:'
	print best_weights
	'''
	
	
	for thresh in 0.01*np.array(range(30,85,5)):
		tr_preds = NNFwdProp(tr, best_weights, NNshape)
		cv_preds = NNFwdProp(cv, best_weights, NNshape)
		tr_preds[tr_preds>=thresh]=1
		tr_preds[tr_preds<thresh]=0
		cv_preds[cv_preds>=thresh]=1
		cv_preds[cv_preds<thresh]=0
		print "Threshold = "+ str(thresh)
		print "Train Accuracy = "+ str(100*(1-np.sum(np.abs(tr_preds-y_tr))/len(y_tr)))
		print "CV Accuracy = "+ str(100*(1-np.sum(np.abs(cv_preds-y_cv))/len(y_cv)))
	

	print 'Reading test data...'
	test = pd.read_csv('test.csv')
	print 'Extracting features...'
	x_test = MakeFeatures(test)
	print 'Making predictions for test data...'
	preds_test = NNFwdProp(x_test, best_weights, NNshape)
	preds_test[preds_test>=0.6]=1
	preds_test[preds_test<0.6]=0
	preds_test[np.isnan(preds_test)]=0
	preds_test = map(int, preds_test)
	
	print 'Writing to CSV file..'
	template = pd.read_csv('genderclassmodel.csv')
	template['Survived'] = preds_test
	template.to_csv('Test_predictions.csv', index = False)
	print 'Done!!'

	return

if __name__ == '__main__':
	main()
