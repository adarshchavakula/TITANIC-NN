import numpy as np
from scipy.optimize import minimize
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
	
	init_weights = np.random.randn(nw1+nw2+nw3)
	res = minimize(err, init_weights, args = (train, response, hidden), options = {'maxiter':500})
	return res.x


def main():
	print 'Reading train data...'
	train = pd.read_csv('train.csv')
	print 'Extracting features...'
	features = MakeFeatures(train)
	response = np.array(train['Survived'])

	print 'Training the Neural Network...'
	NNshape = [10,10]
	best_weights = TrainNN(np.array(features),response,hidden = NNshape)
	'''
	print '\n\nEstimated best weights are:'
	print best_weights
	'''
	preds = NNFwdProp(features, best_weights, NNshape)
	#print '\n\nTrain Predictions:'
	#print map(int,np.round(preds))

	print "Train Accuracy = "+ str(100*(1-np.sum(np.abs(preds-response))/len(features)))


	return

if __name__ == '__main__':
	main()
