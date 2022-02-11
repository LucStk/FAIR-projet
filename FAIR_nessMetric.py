import torch
import torch.nn as nn

"""
testX = torch.ones((5,5))
testX[4,0] = 0
testX[3,0] = 0
testY = torch.ones(5)
testYh = torch.ones(5)
testYh[4] = 0
testYh[2] = 0
testK = 0
print("Truth : " + str(testY))
print("Predi : " + str(testYh))
print("Prote : " + str(testX[:,testK]))"""

def AbsEqOppDiff(a,y,y_hat):
    # |TPRdiff|
    eps = 1e-4
    TP = (y.int() == 1) * (y_hat.int() == 1)
    FN = (y.int() == 1) * (y_hat.int() == 0) 
    TPp = (TP * a).sum()
    FNp = (FN * a).sum()
    TPu = (TP * (1-a)).sum()
    FNu = (FN * (1-a)).sum()
    TPRp = TPp / (TPp + FNp + eps)
    TPRu = TPu / (TPu + FNu + eps)
    return torch.abs(TPRp - TPRu)

def AbsAvgOddsDiff(a,y,y_hat):
    # (|TPRdiff| + |FPRdiff|) / 2
    return (AbsEqOppDiff(a,y,y_hat) + AbsEqOppDiff(a,1-y,y_hat)) / 2

def DisparateImpact(a,y,y_hat):
    eps = 1e-4
    indexYh = y_hat == 1
    num = ((1-a) * indexYh).sum() / (1-a + eps).sum()
    denom = (a * indexYh).sum() / (a+eps).sum()
    return 1 - num/(denom+eps)

"""
print("###TESTING")
print(AbsEqOppDiff(testX[:,testK],testY,testYh))
print(AbsAvgOddsDiff(testX[:,testK],testY,testYh))
print(DisparateImpact(testX[:,testK],testY,testYh))
raise E"""