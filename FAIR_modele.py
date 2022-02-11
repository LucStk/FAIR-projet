import torch
import torch.nn as nn
from FAIR_nessMetric import *

H1_Selecteur_SIZE = 100
H1_Predicteur_SIZE = 50
H2_Predicteur_SIZE = 30
LR_SELECTEUR = 1e-4
LR_PREDICTEUR = 1e-3
LR_DISCRIMINATEUR = 1e-3

OUTPUT_SIZE = 2

softmax = torch.nn.Softmax(dim=1)
log_softmax = torch.nn.LogSoftmax(dim=1)
class Baseline(object):
    """Simple model"""

    def __init__(self, device, INPUT_SIZE, writer = None):
        self.writer = writer
        self.device = device
        self.epoch = 0
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

    def test(self, x, y, k):
        self.epoch += 1
        y_hat = self.predict(x, k)
        l_predict = self.loss(y_hat,y.long().to(self.device))
        prediction = torch.argmax(y_hat.cpu(), dim = 1)
        acc = (prediction == y.int()).float().mean()

        if self.writer is not None:
            self.writer.add_scalar('test/Loss_predicteur', l_predict.cpu(), self.epoch)
            self.writer.add_scalar('test/Accuracy', acc, self.epoch)
            self.writer.add_scalar('test/AbsEqOppDiff', AbsEqOppDiff(x[:,k],y,prediction), self.epoch)
            self.writer.add_scalar('test/AbsAvgOddsDiff', AbsAvgOddsDiff(x[:,k],y,prediction), self.epoch)
            self.writer.add_scalar('test/1-DispImpact', DisparateImpact(x[:,k],y,prediction), self.epoch)
            self.writer.flush()
        print("Epoch "+ str(self.epoch)+ " - AccTest : " + str(acc))




class FairModele(Baseline):
    """Article base model"""

    def __init__(self, device, INPUT_SIZE, writer):
        self.device = device
        self.writer = writer
        self.epoch = 0
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
        self.cpt = 0

    def predict(self, data, k):
        #Selection des features
        self.k = k
        self.data = data
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
        self.cpt += 1
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
        
        with torch.no_grad() :
            l_predict = self.loss(self.y_hat_nok,y.long().to(self.device))
            prediction = torch.argmax(self.y_hat_nok.cpu(), dim = 1)
            acc = (prediction == y.int()).float().mean()

            if self.writer is not None:
                self.writer.add_scalar('train/Loss_predicteur', l_predict.cpu(), self.epoch)
                self.writer.add_scalar('train/Accuracy', acc, self.epoch)

                absEqOppDiff    = AbsEqOppDiff(self.data[:,self.k],y,prediction)
                absAvgOddsDiff  = AbsAvgOddsDiff(self.data[:,self.k],y,prediction)
                disparateImpact = DisparateImpact(self.data[:,self.k],y,prediction)

                if torch.isnan(absEqOppDiff) or torch.isnan(absAvgOddsDiff) or torch.isnan(disparateImpact):
                    raise

                self.writer.add_scalar('train/AbsEqOppDiff', absEqOppDiff, self.cpt)
                self.writer.add_scalar('train/AbsAvgOddsDiff', absAvgOddsDiff, self.cpt)
                self.writer.add_scalar('train/1-DispImpact',disparateImpact , self.cpt)
                self.writer.flush()




class FairModele_GAN(object):
    """Article base model"""

    def __init__(self, device, INPUT_SIZE, writer = None):
        self.epoch = 0
        self.device = device
        self.writer = writer
        self.cpt = 0
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

        self.Discriminateur = nn.Sequential(
            nn.Linear(INPUT_SIZE, H1_Predicteur_SIZE), nn.ReLU(),
            nn.Linear(H1_Predicteur_SIZE, H2_Predicteur_SIZE), nn.ReLU(),
            nn.Linear(H2_Predicteur_SIZE, 1), nn.Sigmoid()
        )

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
        self.opti_discriminateur = torch.optim.Adam(self.Discriminateur.parameters(), LR_DISCRIMINATEUR)

    def predict(self, data, k):
        #Selection des features
        self.probaSelect  = self.Selecteur(data)

        self.y_sensitive = data[:,k] #Pour l'apprentissage du discriminateur

        self.select_k = (torch.rand(data.shape) < self.probaSelect).int() #Sélection des features
        self.select_nok = self.select_k.clone()
        self.select_k[:,k] = 1 #Avec sensitive feature
        self.select_nok[:,k] = 0 #Sans sensitive feature
        #Prediction
        self.y_hat_nok = self.Predicteur((data*self.select_nok)).squeeze() #without sensitive feature
        self.y_hat_k   = self.Predicteur((data*self.select_k)).squeeze() #with sensitive feature
        return self.y_hat_nok

    def train(self, y, normalizer):
        self.cpt+=1
        #Loss
        l_pred = self.loss(self.y_hat_nok, y)
        l_sens = - (softmax(self.y_hat_nok)*log_softmax(self.y_hat_k)).sum()
        #Optimisation Selecteur
        self.opti_selecteur.zero_grad()
        l_select = -((l_sens - l_pred).detach()*self.BCE(self.probaSelect, self.select_nok.float())) * normalizer
        l_select.backward()
        self.opti_selecteur.step()

        #Optimisation Discriminateur
        self.opti_discriminateur.zero_grad()
        l_discrim = self.BCE(self.Discriminateur(self.y_hat_nok), self.y_sensitive)
        l_discrim.backward()
        self.opti_discriminateur.step()

        #Optimisation Predicteur
        self.opti_predicteur.zero_grad()
        l_predict = (l_pred + l_sens) * normalizer
        l_predict.backward()
        self.opti_predicteur.step()

        if self.writer is not None:
            self.writer.add_scalar('train_predict/Loss_pred', l_pred.cpu()  , self.cpt)
            self.writer.add_scalar('train_predict/Loss_sent', l_sens.cpu()  , self.cpt)
            self.writer.add_scalar("train_predict/Loss_disc", l_discrim.cpu(), self.cpt)
            self.writer.flush()

    def test(self, x, y, k):
        self.epoch += 1
        y_hat = self.predict(x, k)
        l_predict = self.loss(y_hat,y.long().to(self.device))
        prediction = torch.argmax(y_hat.cpu(), dim = 1)
        acc = (prediction == y.int()).float().mean()

        y_hat_sensitive = self.Discriminateur(self.y_hat_nok)
        acc_sensitive = (torch.argmax(y_hat_sensitive.cpu(), dim = 1) == self.y_sensitive.int().cpu()).float().mean()
  

        if self.writer is not None:
            self.writer.add_scalar('test/Loss_predicteur', l_predict.cpu(), self.epoch)
            self.writer.add_scalar('test/Accuracy', acc, self.epoch)
            self.writer.add_scalar('test/AbsEqOppDiff', AbsEqOppDiff(x[:,k],y,prediction), self.epoch)
            self.writer.add_scalar('test/AbsAvgOddsDiff', AbsAvgOddsDiff(x[:,k],y,prediction), self.epoch)
            self.writer.add_scalar('test/1-DispImpact', DisparateImpact(x[:,k],y,prediction), self.epoch)
            self.writer.add_scalar('test/Accuracy_sensitive', acc_sensitive, self.epoch)
            self.writer.flush()

        print("Epoch "+ str(self.epoch)+ " - AccTest : " + str(acc))

