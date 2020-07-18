from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,recall_score,roc_auc_score,roc_curve,confusion_matrix
import numpy as np
import math
from sklearn.model_selection import KFold  #For K-fold cross validation\n
import random
import copy
import math
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score, matthews_corrcoef

class cross_validation:
      def __init__(self,X_data,y_data, model,cv):
            self.X_data=X_data
            self.y_data=y_data
            self.cv=cv
            self.model=model
        
      def fit(self):
         kf = KFold(n_splits=self.cv, random_state=None, shuffle=False)
         C,tp,tn,fp,fn,pr,rc,mcc = [],[],[],[],[],[],[],[]
         for train, test in kf.split(self.X_data):
            train_predictors = (self.X_data.iloc[train,:])
            train_target = self.y_data.iloc[train]
            self.model.fit(train_predictors,train_target)
            tspred = self.model.predict(self.X_data.iloc[test,:])
            cm1=confusion_matrix(self.y_data.iloc[test], tspred)
            tp.append(cm1[1,1])
            tn.append(cm1[0,0])
            fp.append(cm1[0,1])
            fn.append(cm1[1,0])
           
         TPs=np.sum(tp)
         TNs=np.sum(tn)
         FPs=np.sum(fp)
         FNs=np.sum(fn)
         Sn=(TPs/(TPs+FNs))*100
         Sp=(TNs/(TNs+FPs))*100
         PRs=(TPs/(TPs+FPs))*100
         F1s=(2*TPs / (2*TPs + FPs + FNs))*100
         Acc=((TPs+TNs)/(TPs+TNs+FPs+FNs))*100
         print('True positive: '+str(TPs))
         print('True negative: '+str(TNs))
         print('False postive: '+str(FPs))
         print('False negative: '+str(FNs))
         print('Sensitivity: '+str(Sn))
         print('Specificity: '+str(Sp))
         print('Accuracy: '+str(Acc))
         print('Precision: '+str(PRs))
         print('F1 score: '+str(F1s))
         return TPs,TNs,FPs,FNs,Sn,Sp,Acc,F1s