from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,recall_score,roc_auc_score,roc_curve,confusion_matrix
class testset_prediction():
      def __init__(self,X,y,model):
            self.X=X
            self.y=y
            self.model=model
      def fit(self):
          ypred=self.model.predict(self.X)
          cm1=confusion_matrix(self.y,ypred)
          Sn = float(cm1[0,0])/(cm1[0,0]+cm1[0,1])
          Sp=float(cm1[1,1])/(cm1[1,0]+cm1[1,1])
          Sn=Sn*100
          Sp=Sp*100
          acc=accuracy_score(self.y,ypred)*100
          f1=f1_score(self.y,ypred)*100
          mcc=matthews_corrcoef(self.y,ypred)
          roc=roc_auc_score(self.y,ypred)
          print('True Positive: '+str(cm1[1,1]))
          print('True Negative: '+str(cm1[0,0]))
          print('False Positive '+str(cm1[0,1]))
          print('False Negative '+str(cm1[1,0]))
          print('Sensitivity: '+str(Sn))
          print('Specificity: '+str(Sp))
          print('Accuracy: '+str(acc))
          print('f1_score: '+str(f1))
          #print('Recall score: '+str(recall_score(self.y,ypred)))
          print('MCC: '+str(mcc))
          print('ROC_AUC: '+str(roc))
          return cm1[1,1],cm1[0,0],cm1[0,1],cm1[1,0],Sn,Sp,acc,f1,mcc,roc 