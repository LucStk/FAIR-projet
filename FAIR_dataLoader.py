import pandas
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer

BATCH_SIZE  = 100

def load_Adult():
	data = pandas.read_csv("adult.csv")

	"""
	Les paramètres :
	'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
	'marital-status', 'occupation', 'relationship', 'race', 'gender',
	'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
	'income'

	On prédit le genre

	Sont continues:
	    'age', 'fnlwgt', 'educational-num',
	    'capital-gain', 'capital-loss', 'hours-per-week'

	On encode one-hot : 
	    workclass, relationship, race, native-country, occupation, marital-status

	On binarise :
	    income (<= 50k ou >50k) et gender (H ou F)

	On n'utilise pas "education" mais sa version en continue avec educational-num
	"""

	y = torch.Tensor(LabelBinarizer().fit_transform(data.income)).squeeze()

	data_continues = data[['age', 'fnlwgt', 'educational-num',
	                       'capital-gain', 'capital-loss', 'hours-per-week']].to_numpy()

	#Normalisation data_continues
	data_continues = (data_continues - data_continues.min(axis=0))/data_continues.max(axis=0)

	data_one_hot = pandas.get_dummies(data[['workclass', 'relationship', 'native-country', 'occupation', 'marital-status', 'race']]).to_numpy()
	data_binary  = LabelBinarizer().fit_transform(data.gender)
	x            = np.concatenate((data_continues, data_one_hot, data_binary), axis = 1)		

	dataset_train    = torch.utils.data.TensorDataset( torch.Tensor(x[10000:]), torch.Tensor(y[10000:]))
	dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)
	dataset_test     = (torch.Tensor(x[:10000]), torch.Tensor(y[:10000]))

	k = 90 #sensitive feature Gender
	return dataloader_train, dataset_test, k, x.shape[1], len(data)