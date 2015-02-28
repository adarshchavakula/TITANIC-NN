import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
import matplotlib.pyplot as plt
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
	return features

def main():
	train = pd.read_csv('train.csv')
	counts = train['Survived'].value_counts()
	counts.plot(kind='bar')
	#plt.show()
	names = train['Name']
	titles = GetTitle(names)
	train['Title'] = titles
	print train['Title'].value_counts()
	return

if __name__ == '__main__':
	main()
