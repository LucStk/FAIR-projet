import torch
import torch.nn as nn

H1_Selecteur_SIZE = 100
H1_Predicteur_SIZE = 50
H2_Predicteur_SIZE = 30
LR_SELECTEUR = 1e-4
LR_PREDICTEUR = 1e-4

OUTPUT_SIZE = 2

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
        #Optimisation
        self.loss = nn.CrossEntropyLoss(reduction = 'sum')
        self.BCE = nn.BCELoss(reduction = "sum").to(self.device)
        self.opti_selecteur = torch.optim.Adam(self.Selecteur.parameters(), LR_SELECTEUR)
        self.opti_predicteur = torch.optim.Adam(self.Predicteur.parameters(), LR_PREDICTEUR)
        
    def Select(self, data, k):
        filtre = torch.ones(data.shape).to(self.device)
        filtre[:,k] = 0

        probaSelect  = self.Selecteur(data)
        selectInputsFiltr = probaSelect*filtre

        select = (torch.rand(data.shape).to(self.device) < selectInputsFiltr).int()# Sélection without sensitive feature
        select_k = select.clone()
        select_k[:,k] += 1 #Selection with sensitive feature
        return select, select_k, probaSelect


