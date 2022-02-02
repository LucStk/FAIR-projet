import torch
import torch.nn as nn

H1_Selecteur_SIZE = 100
H1_Predicteur_SIZE = 50
H2_Predicteur_SIZE = 30
LR_SELECTEUR = 1e-4
LR_PREDICTEUR = 1e-4

OUTPUT_SIZE = 2

softmax = torch.nn.Softmax(dim=1)
log_softmax = torch.nn.LogSoftmax(dim=1)

class FairModele(object):
    """Article base model"""

    def __init__(self, device, INPUT_SIZE):
        self.device = device
        #Selecteur
        self.Selecteur = nn.Sequential(
            nn.Linear(INPUT_SIZE, H1_Selecteur_SIZE), nn.ReLU(),
            nn.Linear(H1_Selecteur_SIZE, INPUT_SIZE), nn.Sigmoid()
        ).to(self.device)
        #Predicteur
        self.Predicteur = nn.Sequential(
            nn.Linear(INPUT_SIZE, H1_Predicteur_SIZE), nn.ReLU(),
            nn.Linear(H1_Predicteur_SIZE, H2_Predicteur_SIZE), nn.ReLU(),
            nn.Linear(H2_Predicteur_SIZE, OUTPUT_SIZE)
        ).to(self.device)
        """self.Predicteur = nn.Sequential(
            nn.Linear(INPUT_SIZE, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, OUTPUT_SIZE)
        ).to(self.device)"""
        #Optimisation
        self.loss = nn.CrossEntropyLoss(reduction = 'sum')
        self.BCE = nn.BCELoss(reduction = "sum").to(self.device)
        self.opti_selecteur = torch.optim.Adam(self.Selecteur.parameters(), LR_SELECTEUR)
        self.opti_predicteur = torch.optim.Adam(self.Predicteur.parameters(), LR_PREDICTEUR)
        

    def predict(self, data, k):
        #Selection des features
        self.probaSelect  = self.Selecteur(data)
        self.select_k = (torch.rand(data.shape) < self.probaSelect).int() #Sélection des features
        self.select_nok = self.select_k.clone()
        self.select_k[:,k] = 1 #Avec sensitive feature
        self.select_nok[:,k] = 0 #Sans sensitive feature
        #Prediction
        self.y_hat_nok = self.Predicteur((data*self.select_nok)).squeeze() #without sensitive feature
        self.y_hat_k   = self.Predicteur((data*self.select_k)).squeeze() #with sensitive feature
        return self.y_hat_nok

    def train(self, y, normalizer):
        #Loss
        l_pred = self.loss(self.y_hat_nok, y)
        l_sens = - (softmax(self.y_hat_nok)*log_softmax(self.y_hat_k)).sum()
        #Optimisation Selecteur
        self.opti_selecteur.zero_grad()
        l_select = -((l_sens - l_pred).detach()*self.BCE(self.probaSelect, self.select_nok.float())) * normalizer
        l_select.backward()
        self.opti_selecteur.step()
        #Optimisation Predicteur
        self.opti_predicteur.zero_grad()
        l_predict = (l_pred + l_sens) * normalizer
        l_predict.backward()
        self.opti_predicteur.step()

class Baseline(object):
    """Simple model"""

    def __init__(self, device, INPUT_SIZE):
        self.device = device
        #Predicteur
        self.Predicteur = nn.Sequential(
            nn.Linear(INPUT_SIZE, H1_Predicteur_SIZE), nn.ReLU(),
            nn.Linear(H1_Predicteur_SIZE, H2_Predicteur_SIZE), nn.ReLU(),
            nn.Linear(H2_Predicteur_SIZE, OUTPUT_SIZE)
        ).to(self.device)
        """self.Predicteur = nn.Sequential(
            nn.Linear(INPUT_SIZE, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, OUTPUT_SIZE)
        ).to(self.device)"""
        #Optimisation
        self.loss = nn.CrossEntropyLoss(reduction = 'sum')
        self.opti_predicteur = torch.optim.Adam(self.Predicteur.parameters(), LR_PREDICTEUR)
        

    def predict(self, data, k):
        #Prediction
        self.y_hat = self.Predicteur(data).squeeze() #without sensitive feature
        return self.y_hat

    def train(self, y, normalizer):
        #Loss
        l_pred = self.loss(self.y_hat, y) * normalizer
        #Optimisation Predicteur
        self.opti_predicteur.zero_grad() 
        l_pred.backward()
        self.opti_predicteur.step()