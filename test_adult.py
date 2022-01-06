import torch
import torch.nn as nn
import torchvision
import pandas
import time
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

"""
    Télécharger le dataset csv "Adult income dataset" sur kaggle 
    https://www.kaggle.com/wenruliu/adult-income-dataset

    Description de la base : http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html
"""

data = pandas.read_csv("adult.csv")
writer = SummaryWriter("logs/"+str(time.time()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
data_continues = (data_continues - data_continues.mean(0))/data_continues.std(0)

data_one_hot = pandas.get_dummies(data[['workclass', 'relationship', 'race', 'native-country', 'occupation', 'marital-status']]).to_numpy()
data_binary  = LabelBinarizer().fit_transform(data.gender)
x            = np.concatenate((data_continues, data_binary, data_one_hot), axis = 1)

INPUT_SIZE  = x.shape[1]
OUTPUT_SIZE = 2
BATCH_SIZE  = 30

dataset_train    = torch.utils.data.TensorDataset( torch.Tensor(x[10000:]
), torch.Tensor(y[10000:]))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)

dataset_test    = ( torch.Tensor(x[:10000]
), torch.Tensor(y[:10000]))

"""
    Création du Prédicteur
"""
H1_SIZE = 50
H2_SIZE = 30
Predicteur = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1_SIZE), nn.ReLU(),
    nn.Linear(H1_SIZE, H2_SIZE)   , nn.ReLU(),
    nn.Linear(H2_SIZE, OUTPUT_SIZE)
).to(device)

"""
Optimisation
"""
opti_predicteur = torch.optim.Adam(Predicteur.parameters(), 1e-4)
loss = nn.CrossEntropyLoss()
NB_MAX_ITERATION = 30
cpt = 0
for i in range(NB_MAX_ITERATION):
    for x, y in dataloader_train:
        cpt += 1
        # On apprend le prédicteur
        opti_predicteur.zero_grad()

        y_hat = Predicteur(x.to(device))
        l_predict = loss(y_hat,y.long().to(device))*(BATCH_SIZE/len(data))
        l_predict.backward()

        opti_predicteur.step()
        #On calcule le nombre de bon résultats
        acc = (torch.max(y_hat.cpu(), dim = 1)[1] == y).float().mean()

        #writer.add_scalar('train/Loss_selecteur' , l_select, cpt)
        writer.add_scalar('train/Loss_predicteur', l_predict.cpu()  , cpt)
        writer.add_scalar('train/Accuracy', acc  , cpt)

    x, y = dataset_test
    with torch.no_grad():
        y_hat = Predicteur(x.to(device))
        l_predict = loss(y_hat,y.long().to(device))*(BATCH_SIZE/len(data))
        acc = (torch.max(y_hat.cpu(), dim = 1)[1] == y).float().mean()
    
    writer.add_scalar('test/Loss_predicteur', l_predict.cpu(), i)
    writer.add_scalar('test/Accuracy', acc, i)

    
