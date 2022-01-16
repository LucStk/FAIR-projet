from torch.utils.tensorboard import SummaryWriter
from FAIR_modele import *
from FAIR_dataLoader import *
from FAIR_nessMetric import *
import time

SEED = 10
SHOULDLOG = True
EPOCH = 30

#Initialisation
torch.manual_seed(SEED)
if(SHOULDLOG):
    name = input('Nom enregistrement :')
    writer = SummaryWriter("logs/"+name+'-'+str(time.time()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
softmax = torch.nn.Softmax(dim=1)
log_softmax = torch.nn.LogSoftmax(dim=1)
#Recuperation des donnees
dataloader_train, dataset_test, k, INPUT_SIZE, lenData = load_Adult()
#Definition du modele
model = FairModele(device, INPUT_SIZE)

cpt = 0
for i in range(EPOCH):
    #Train
    for x, y in dataloader_train:
        cpt += 1
        #Recuperation des donnees
        x = x.to(device)
        y = y.long().to(device)
        #Selection des features
        select, select_k, g = model.Select(x, k)
        #Prediction
        y_hat   = model.Predicteur((x*select)).squeeze() #without sensitive feature
        y_hat_k = model.Predicteur((x*select_k)).squeeze() #with sensitive feature
        #Loss
        l_pred = model.loss(y_hat, y)
        l_sens = - (softmax(y_hat)*log_softmax(y_hat_k)).sum()
        #Optimisation Selecteur
        model.opti_selecteur.zero_grad()
        l_select = -((l_sens - l_pred).detach()*model.BCE(g, select.float()))* (BATCH_SIZE/lenData)
        l_select.backward()
        model.opti_selecteur.step()
        #Optimisation Predicteur
        model.opti_predicteur.zero_grad()
        l_predict = (l_pred + l_sens)*(BATCH_SIZE/lenData)
        l_predict.backward()
        model.opti_predicteur.step()

        #Logs
        if(SHOULDLOG):
            acc = (torch.argmax(y_hat.cpu(), dim = 1) == y.int().cpu()).float().mean()
            #writer.add_scalar('train_select/percent_selection' , float(select.sum() / sum(select.shape)), cpt)
            #writer.add_scalar('train_select/mean_selection' , g.mean().cpu(), cpt)
            #writer.add_scalar('train_select/std_selection'  , g.std().cpu(), cpt)
            #writer.add_scalar('train_select/Loss_selecteur' , l_select.cpu(), cpt)

            #writer.add_scalar('train_predict/Global_loss', l_predict.cpu()  , cpt)
            #writer.add_scalar('train_predict/Loss_pred', l_pred.cpu()  , cpt)
            #writer.add_scalar('train_predict/Loss_sent', l_sens.cpu()  , cpt)
            
            writer.add_scalar('train/Accuracy', acc  , cpt)

    #Test
    x, y = dataset_test
    with torch.no_grad():
        y_hat = model.Predicteur(x.to(device))
        l_predict = model.loss(y_hat,y.long().to(device))*(BATCH_SIZE/lenData)
        acc = (torch.argmax(y_hat.cpu(), dim = 1) == y.int()).float().mean() 
        if(SHOULDLOG):
            writer.add_scalar('test/Loss_predicteur', l_predict.cpu(), i)
            writer.add_scalar('test/Accuracy', acc, i)
            writer.add_scalar('test/AbsEqOppDiff', AbsEqOppDiff(x,y,y_hat,k), i)
    print("Epoch "+ str(i)+ " - AccTest : " + str(acc))