import torch
import torch.nn as nn
import torchvision
import pandas
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

"""
    Télécharger le dataset csv "Adult income dataset" sur kaggle 
    https://www.kaggle.com/wenruliu/adult-income-dataset

    Description de la base : http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html
"""

data = pandas.read_csv("adult.csv")
writer = SummaryWriter("logs")
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
    income et gender

On n'utilise pas "education" mais sa version en continue avec educational-num
"""

y = torch.Tensor(LabelBinarizer().fit_transform(data.gender)).squeeze()

data_continues = data[['age', 'fnlwgt', 'educational-num',
                       'capital-gain', 'capital-loss', 'hours-per-week']].to_numpy()

data_one_hot = pandas.get_dummies(data[['workclass', 'relationship', 'race', 'native-country', 'occupation', 'marital-status']]).to_numpy()
data_binary  = LabelBinarizer().fit_transform(data.income)
x            = np.concatenate((data_continues, data_binary, data_one_hot), axis = 1)





INPUT_SIZE  = x.shape[1]
OUTPUT_SIZE = 1
BATCH_SIZE  = 10

dataset      = torch.utils.data.TensorDataset( torch.Tensor(x), torch.Tensor(y) )
dataloader   = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)


"""
    Création du Séléctionneur, 
    on met qu'une seule couche pour l'instant et on sort en sigmoid pour avoir des proba
"""
H1_SIZE = 100

Selecteur = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1_SIZE)   , nn.ReLU(),
    nn.Linear(H1_SIZE   , INPUT_SIZE), nn.Sigmoid()
)

"""
    Création du Prédicteur
"""
H1_SIZE = 50
H2_SIZE = 30
Predicteur = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1_SIZE)    , nn.ReLU(),
    nn.Linear(H1_SIZE   , H2_SIZE)    , nn.ReLU(),
    nn.Linear(H2_SIZE   , OUTPUT_SIZE), nn.Sigmoid()
)

"""
Optimisation
"""

opti_selecteur  = torch.optim.Adam(Selecteur.parameters(), 1e-3)
opti_predicteur = torch.optim.Adam(Predicteur.parameters(), 1e-3)

NB_MAX_ITERATION = 100
cpt = 0
for i in range(NB_MAX_ITERATION):
    for x, y in dataloader:
        cpt += 1
        opti_predicteur.zero_grad()
        opti_selecteur.zero_grad()

        # On selectionne les features
        g = Selecteur(x)
        rand      = torch.rand(x.shape[0], x.shape[1])
        select = (rand < g).int()

        # On calcule les loss
        k = np.random.choice(range(x.shape[1]), x.shape[0]) #Selection des sensitives features pour chaque batch
        selection_k    = select.clone()
        selection      = select.clone()
        selection[range(x.shape[0]),k]   = 0 #Selection sans les sensitives features
        selection_k[range(x.shape[0]),k] = 1 #Selection avec les sensitives features

        #On backward le Selecteur
        pred    = Predicteur(x*selection).squeeze()

        with torch.no_grad():
            l_pred  = - (y*torch.log(pred))
            l_sent  = - (pred*torch.log(Predicteur(x*selection_k).squeeze()))

        pi = (torch.pow(g, select)*torch.pow(1-g, 1-select)).prod(dim=1)

        l_select = - ((l_sent - l_pred)*torch.log(pi)).sum()* (BATCH_SIZE/len(data))
        l_select.backward()
        opti_selecteur.step()

        # On apprend le prédicteur

        pred_k = Predicteur(x*selection_k).squeeze()
        l_pred = - (y * (pred/ pred.detach()) + pred.detach()* (pred_k/pred_k.detach()) + pred*torch.log(pred_k.detach())).sum()*(BATCH_SIZE/len(data))
        l_pred.backward()
        opti_predicteur.step()

        writer.add_scalar('train/Loss_selecteur' , l_select, cpt)
        writer.add_scalar('train/Loss_predicteur', l_pred  , cpt)

    
