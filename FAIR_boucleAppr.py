from torch.utils.tensorboard import SummaryWriter
from FAIR_modele import *
from FAIR_dataLoader import *
from FAIR_nessMetric import *
import time

SEED = 10
SHOULDLOG = True
EPOCH = 300
model = "FAIR_"

#Initialisation
torch.manual_seed(SEED)
if(SHOULDLOG):
    name = input('Nom enregistrement :')
    writer = SummaryWriter("logs/"+name+'-'+str(time.time()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
#Recuperation des donnees
dataloader_train, dataset_test, k, INPUT_SIZE, lenData = load_Adult()
#Definition du modele
if(model == "BASELINE"):
    model = Baseline(device, INPUT_SIZE, writer)
elif (model == "FAIR"):
    model = FairModele(device, INPUT_SIZE, writer)
else:
    model = FairModele_GAN(device, INPUT_SIZE, writer)

cpt = 0
for i in range(EPOCH):
    #Train
    for x, y in dataloader_train:
        cpt += 1
        #Recuperation des donnees
        x = x.to(device)
        y = y.long().to(device)
        #Entrainement du model
        model.predict(x, k)
        model.train(y, BATCH_SIZE/lenData)

    #Test
    x, y = dataset_test
    with torch.no_grad():
        model.test(x, y, k)

    