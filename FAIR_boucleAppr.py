from torch.utils.tensorboard import SummaryWriter
from FAIR_modele import *
from FAIR_dataLoader import *
from FAIR_nessMetric import *
import time

SEED = 10
SHOULDLOG = True
EPOCH = 300
model = "NBASELINE"

#Initialisation
torch.manual_seed(SEED)
if(SHOULDLOG):
    name = input('Nom enregistrement :')
    writer = SummaryWriter("logs/"+name+'-'+str(time.time()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Recuperation des donnees
dataloader_train, dataset_test, k, INPUT_SIZE, lenData = load_Adult()
#Definition du modele
if(model == "BASELINE"):
    model = Baseline(device, INPUT_SIZE)
else:
    model = FairModele(device, INPUT_SIZE)



cpt = 0
for i in range(EPOCH):
    #Train
    for x, y in dataloader_train:
        cpt += 1
        #Recuperation des donnees
        x = x.to(device)
        y = y.long().to(device)
        #Entrainement du model
        y_hat = model.predict(x, k)
        model.train(y, BATCH_SIZE/lenData)
        #Logs
        if(SHOULDLOG):
            acc = (torch.argmax(y_hat.cpu(), dim = 1) == y.int().cpu()).float().mean()
            #writer.add_scalar('train_select/percent_selection' , float(select.sum() / sum(select.shape)), cpt)
            #writer.add_scalar('train_select/mean_selection' , probaSelect.mean().cpu(), cpt)
            #writer.add_scalar('train_select/std_selection'  , probaSelect.std().cpu(), cpt)
            #writer.add_scalar('train_select/Loss_selecteur' , l_select.cpu(), cpt)

            #writer.add_scalar('train_predict/Global_loss', l_predict.cpu()  , cpt)
            #writer.add_scalar('train_predict/Loss_pred', l_pred.cpu()  , cpt)
            #writer.add_scalar('train_predict/Loss_sent', l_sens.cpu()  , cpt)
            
            writer.add_scalar('train/Accuracy', acc  , cpt)

    #Test
    x, y = dataset_test
    with torch.no_grad():
        y_hat = model.predict(x, k)
        l_predict = model.loss(y_hat,y.long().to(device))*(BATCH_SIZE/lenData)
        prediction = torch.argmax(y_hat.cpu(), dim = 1)
        acc = (prediction == y.int()).float().mean() 
        if(SHOULDLOG):
            writer.add_scalar('test/Loss_predicteur', l_predict.cpu(), i)
            writer.add_scalar('test/Accuracy', acc, i)
            writer.add_scalar('test/AbsEqOppDiff', AbsEqOppDiff(x[:,k],y,prediction), i)
            writer.add_scalar('test/AbsAvgOddsDiff', AbsAvgOddsDiff(x[:,k],y,prediction), i)
            writer.add_scalar('test/1-DispImpact', DisparateImpact(x[:,k],y,prediction), i)
    print("Epoch "+ str(i)+ " - AccTest : " + str(acc))