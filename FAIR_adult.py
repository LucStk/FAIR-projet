import torch
import torch.nn as nn
import torchvision
import pandas
import time
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

"""
    Télécharger le dataset csv "Adult income dataset" sur kaggle 
    https://www.kaggle.com/wenruliu/adult-income-dataset

    Description de la base : http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html
"""

data = pandas.read_csv("adult.csv")
name = input('Nom enregistrement :')
writer = SummaryWriter("logs/"+name+'-'+str(time.time()))
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
data_continues = (data_continues - data_continues.min(axis=0))/data_continues.max(axis=0)

data_one_hot = pandas.get_dummies(data[['workclass', 'relationship', 'native-country', 'occupation', 'marital-status', 'race']]).to_numpy()
data_binary  = LabelBinarizer().fit_transform(data.gender)
x            = np.concatenate((data_continues, data_one_hot, data_binary), axis = 1)

INPUT_SIZE  = x.shape[1]
OUTPUT_SIZE = 2
BATCH_SIZE  = 20

dataset_train    = torch.utils.data.TensorDataset( torch.Tensor(x[10000:]
), torch.Tensor(y[10000:]))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)

dataset_test    = ( torch.Tensor(x[:10000]
), torch.Tensor(y[:10000]))
"""
    Création du Séléctionneur, 
    on met qu'une seule couche pour l'instant et on sort en sigmoid pour avoir des proba
"""
H1_SIZE = 100

Selecteur = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1_SIZE)   , nn.ReLU(),
    nn.Linear(H1_SIZE   , INPUT_SIZE), nn.Sigmoid()
).to(device)

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
loss = nn.CrossEntropyLoss(reduction = 'sum')
BCE  = nn.BCELoss(reduction = "sum").to(device)
opti_selecteur  = torch.optim.Adam(Selecteur.parameters(), 1e-4)
opti_predicteur = torch.optim.Adam(Predicteur.parameters(), 1e-4)

#Création du filtre à appliquer sur le selecteur
k = 90 #sensitive feature Gender
NB_MAX_ITERATION = 30
cpt = 0

for i in range(NB_MAX_ITERATION):
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.long().to(device)

        filtre = torch.ones(x.shape).to(device)
        filtre[:,k] = 0

        cpt += 1
        g  = Selecteur(x)
        g_ = g*filtre

        select = (torch.rand(x.shape).to(device) < g_).int()# Sélection without sensitive feature
        select_k = select.clone()
        select_k[:,k] += 1 #Selection with sensitive feature

        # On backward le Selecteur
        y_hat   = Predicteur((x*select)).squeeze() #prediction without sensitive feature
        y_hat_k = Predicteur((x*select_k)).squeeze() #prediction with sensitive feature
        
        l_pred = loss(y_hat, y)
        l_sent = - (F.softmax(y_hat)*F.log_softmax(y_hat_k)).sum()

        ############################
        # Apprentissage Sélécteur  #
        ############################
        opti_selecteur.zero_grad()
        l_select = -((l_sent - l_pred).detach()*BCE(g, select.float()))* (BATCH_SIZE/len(data))
        l_select.backward()
        opti_selecteur.step()

        ############################
        # Apprentissage prédicteur #
        ############################
        opti_predicteur.zero_grad()

        l_predict = (l_pred + l_sent)*(BATCH_SIZE/len(data))
        l_predict.backward()

        opti_predicteur.step()
        #On calcule le nombre de bon résultats
        acc = (torch.max(y_hat.cpu(), dim = 1)[1] == y.cpu()).float().mean()
        writer.add_scalar('train_select/percent_selection' , float(select.sum() / sum(select.shape)), cpt)
        writer.add_scalar('train_select/mean_selection' , g.mean().cpu(), cpt)
        writer.add_scalar('train_select/std_selection'  , g.std().cpu(), cpt)
        writer.add_scalar('train_select/Loss_selecteur' , l_select.cpu(), cpt)

        writer.add_scalar('train_predict/Global_loss', l_predict.cpu()  , cpt)
        writer.add_scalar('train_predict/Loss_pred', l_pred.cpu()  , cpt)
        writer.add_scalar('train_predict/Loss_sent', l_sent.cpu()  , cpt)
        
        writer.add_scalar('train/Accuracy', acc  , cpt)

    x, y = dataset_test
    with torch.no_grad():
        y_hat = Predicteur(x.to(device))
        l_predict = loss(y_hat,y.long().to(device))*(BATCH_SIZE/len(data))
        acc = (torch.max(y_hat.cpu(), dim = 1)[1] == y).float().mean()
    
    writer.add_scalar('test/Loss_predicteur', l_predict.cpu(), i)
    writer.add_scalar('test/Accuracy', acc, i)

    
