import torch
import torch.nn as nn

def AbsEqOppDiff(x,y,y_hat,k):
    a = x[:,k]
    indexY1 = y.int() == 1
    indexA1Y1 = (a * indexY1).bool()
    Pa1 = torch.sum(y_hat[indexA1Y1].int() == 1) / torch.sum(indexA1Y1)
    indexA0Y1 = ((1-a) * indexY1).bool()
    Pa0 = torch.sum(y_hat[indexA0Y1].int() == 1) / torch.sum(indexA0Y1)
    return torch.abs(Pa0 - Pa1)

def AbsAvgOddsDiff(x,y,y_hat,k):
    return 0.5 * (AbsEqOppDiff(x,y,y_hat,k) + AbsEqOppDiff(x,1-y,y_hat,k))

def DisparateImpact(x,y):
    #A revoir, ça depend pas du modele pour le moment...
    a = x[:,k].bool()
    num = torch.sum(y[~a].int() == 1) / torch.sum(y[~a])
    denom = torch.sum(y[a].int() == 1) / torch.sum(y[a])
    return num/denom
